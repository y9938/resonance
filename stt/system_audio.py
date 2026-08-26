import abc
import mmap
import os
import struct
import sys
import time
from collections.abc import Generator

import numpy as np


class SystemAudioStrategy(abc.ABC):
    @abc.abstractmethod
    def start_capture(self) -> None:
        pass

    @abc.abstractmethod
    def stop_capture(self) -> None:
        pass

    @abc.abstractmethod
    def get_audio_stream(self) -> Generator[np.ndarray, None, None]:
        pass


class MacOSSharedMemoryStrategy(SystemAudioStrategy):
    # Semaphore names must match Swift exactly and stay ≤ 30 chars (Darwin PSHMNAMLEN).
    _SHM_PATH     = "/tmp/res_audio_shm"
    _DATA_SEM     = "/res_aud_data"
    _CMD_SEM      = "/res_aud_cmd"
    _START_TIMEOUT = 30.0  # seconds to wait for Swift to confirm CAPTURING

    # Invariants: Must exactly match IPCHeader memory layout compiled in Swift.
    # Fields: magic(4s) version(I) sampleRate(I) channels(I) framesPerSlot(I)
    #         slotCount(I) writeIndex(Q) command(I) status(I) errorCode(i) padding(5×UInt32=20x)
    # Total: 44 data bytes + 20 padding = 64 bytes (one ARM64 cache line).
    _HEADER_FMT  = '<4sIIIIIQIIi20x'
    _HEADER_SIZE = 64
    _CMD_OFFSET    = 32
    _STATUS_OFFSET = 36
    _ERROR_OFFSET  = 40

    _STATUS_IDLE     = 0
    _STATUS_STARTING = 1
    _STATUS_CAPTURING = 2
    _STATUS_FAILED   = 3

    def __init__(self):
        import posix_ipc

        try:
            fd = os.open(self._SHM_PATH, os.O_RDWR)
            self.shm = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            os.close(fd)
            self.data_sem = posix_ipc.Semaphore(self._DATA_SEM)
            self.cmd_sem  = posix_ipc.Semaphore(self._CMD_SEM)
        except FileNotFoundError:
            raise RuntimeError("System capture app is not running (shm not found).")
        except PermissionError as e:
            raise RuntimeError(f"Permission denied accessing shared memory. Try restarting the app. ({e})")
        except posix_ipc.ExistentialError:
            raise RuntimeError("System capture app is not running (semaphores not found).")

        (_, _, self.rate, _, self.frames_per_slot, self.slot_count,
         _, _, _, _) = struct.unpack(self._HEADER_FMT, bytes(self.shm[:self._HEADER_SIZE]))
        self.bytes_per_slot = self.frames_per_slot * 4
        self.read_idx = 0

    def start_capture(self) -> None:
        # Caller obligation: Swift must update header.status within _START_TIMEOUT seconds.
        struct.pack_into('<I', self.shm, self._CMD_OFFSET, 1)
        self.cmd_sem.release()

        deadline = time.monotonic() + self._START_TIMEOUT
        while time.monotonic() < deadline:
            status = struct.unpack_from('<I', self.shm, self._STATUS_OFFSET)[0]
            if status == self._STATUS_CAPTURING:
                return
            if status == self._STATUS_FAILED:
                error_code = struct.unpack_from('<i', self.shm, self._ERROR_OFFSET)[0]
                if error_code == -3801:
                    raise RuntimeError(
                        "Screen & System Audio Recording permission is required. "
                        "Grant it in System Settings → Privacy & Security, then restart Resonance."
                    )
                raise RuntimeError(f"System audio capture failed in ScreenCaptureKit (code {error_code}).")
            time.sleep(0.05)

        raise RuntimeError(f"System capture app did not confirm start within {self._START_TIMEOUT:.0f}s.")

    def stop_capture(self) -> None:
        struct.pack_into('<I', self.shm, self._CMD_OFFSET, 2)
        self.cmd_sem.release()
        # Unblock the reader thread blocked on data_sem.acquire().
        self.data_sem.release()

    def get_audio_stream(self) -> Generator[np.ndarray, None, None]:
        while True:
            self.data_sem.acquire()

            (_, _, _, _, _, _, current_write_idx,
             current_cmd, _, _) = struct.unpack(self._HEADER_FMT, bytes(self.shm[:self._HEADER_SIZE]))

            if current_cmd == 2:
                break

            # Tail-drop mitigation: STT inference is slower than real-time capture.
            if current_write_idx > self.read_idx + self.slot_count:
                self.read_idx = current_write_idx - 1

            slot_idx = self.read_idx % self.slot_count
            offset   = self._HEADER_SIZE + (slot_idx * self.bytes_per_slot)

            # Zero-copy invariant: offset must map to a contiguous SHM block.
            audio_chunk = np.ndarray(
                (self.frames_per_slot,),
                dtype=np.float32,
                buffer=self.shm,
                offset=offset,
            )

            self.read_idx += 1
            yield audio_chunk


class NativeLinuxWindowsStrategy(SystemAudioStrategy):
    def __init__(self, include_microphone: bool = False):
        import sounddevice as sd
        from queue import Queue

        self.sd = sd
        self.include_microphone = include_microphone
        self.queue = Queue()
        self.mic_queue = Queue() if include_microphone else None
        self.stream = None
        self.mic_stream = None
        self.is_active = False

        self.samplerate = 16000
        self.channels = 1
        self.blocksize = 4096

        self.device_id = None
        self.extra_settings = None

        if sys.platform.startswith("linux"):
            import subprocess
            try:
                sink = subprocess.check_output(["pactl", "get-default-sink"], timeout=1.0).decode().strip()
                if sink:
                    os.environ["PULSE_SOURCE"] = f"{sink}.monitor"
            except Exception:
                os.environ.setdefault("PULSE_SOURCE", "@DEFAULT_MONITOR@")

            devices = sd.query_devices()
            # Domain Invariant: PortAudio routes to PULSE_SOURCE via pulse/pipewire ALSA bridge.
            for name in ("pulse", "pipewire", "default"):
                idx = next((i for i, d in enumerate(devices) if d["name"] == name and d["max_input_channels"] > 0), None)
                if idx is not None:
                    self.device_id = idx
                    break

    def _audio_callback(self, indata, frames, time, status):
        if self.is_active:
            self.queue.put(indata.copy().flatten())

    def _mic_audio_callback(self, indata, frames, time, status):
        if self.is_active and self.mic_queue is not None:
            self.mic_queue.put(indata.copy().flatten())

    def start_capture(self) -> None:
        if self.stream is not None:
            return

        if sys.platform.startswith("linux"):
            import subprocess
            try:
                sink = subprocess.check_output(["pactl", "get-default-sink"], timeout=1.0).decode().strip()
                if sink:
                    os.environ["PULSE_SOURCE"] = f"{sink}.monitor"
            except Exception:
                pass

        self.is_active = True
        while not self.queue.empty():
            self.queue.get_nowait()
        if self.mic_queue is not None:
            while not self.mic_queue.empty():
                self.mic_queue.get_nowait()

        self.stream = self.sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device_id,
            channels=self.channels,
            dtype=np.float32,
            extra_settings=self.extra_settings,
            callback=self._audio_callback
        )
        self.stream.start()

        if self.include_microphone:
            try:
                # Assumes: device=None selects the system default hardware recording input
                self.mic_stream = self.sd.InputStream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    device=None,
                    channels=self.channels,
                    dtype=np.float32,
                    callback=self._mic_audio_callback
                )
                self.mic_stream.start()
            except Exception:
                self.mic_stream = None

    def stop_capture(self) -> None:
        self.is_active = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None

    def get_audio_stream(self) -> Generator[np.ndarray, None, None]:
        from queue import Empty
        while self.is_active or not self.queue.empty():
            try:
                # Assumes: Callback produces chunks faster than timeout; timeout implies stream death
                chunk_sys = self.queue.get(timeout=2.0)
                if self.include_microphone and self.mic_stream and self.mic_queue is not None:
                    try:
                        chunk_mic = self.mic_queue.get(timeout=0.2)
                        min_len = min(len(chunk_sys), len(chunk_mic))
                        # Domain Invariant: Balanced downmix with clipping prevention
                        yield np.clip(chunk_sys[:min_len] * 0.7 + chunk_mic[:min_len] * 0.7, -1.0, 1.0)
                        continue
                    except Empty:
                        pass
                yield chunk_sys
            except Empty:
                if not self.is_active:
                    break


class WindowsWasapiStrategy(SystemAudioStrategy):
    """Captures Windows desktop audio via WASAPI Loopback Client and optional hardware microphone."""

    def __init__(self, include_microphone: bool = False):
        from queue import Queue

        self.include_microphone = include_microphone
        self.queue = Queue()
        self.mic_queue = Queue() if include_microphone else None
        self.is_active = False
        self.samplerate = 16000
        self.blocksize = 4096
        self.sys_thread = None
        self.mic_thread = None

    def _sys_worker(self) -> None:
        import logging
        log = logging.getLogger("resonance")
        hr = -1
        try:
            import soundcard as sc
            import soundcard.mediafoundation as mf
            hr = mf._ole32.CoInitializeEx(mf._ffi.NULL, 0)
        except Exception as e:
            log.debug(f"Windows system audio COM init note: {e}")

        try:
            speaker = sc.default_speaker()
            loopback_mic = sc.get_microphone(id=str(speaker.id), include_loopback=True)
            with loopback_mic.recorder(samplerate=self.samplerate) as rec:
                while self.is_active:
                    # Windows WASAPI loopback streams native stereo float32.
                    data = rec.record(numframes=self.blocksize)
                    if data is not None and len(data) > 0:
                        # Downmix stereo to mono float32 for STT pipeline.
                        mono = data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
                        self.queue.put(mono)
        except Exception as e:
            log.warning(f"Windows system audio capture worker failed: {e}")
        finally:
            if hr == 0:
                try:
                    mf._ole32.CoUninitialize()
                except Exception as e:
                    log.debug(f"Windows system audio COM uninit note: {e}")

    def _mic_worker(self) -> None:
        import logging
        log = logging.getLogger("resonance")
        hr = -1
        try:
            import soundcard as sc
            import soundcard.mediafoundation as mf
            hr = mf._ole32.CoInitializeEx(mf._ffi.NULL, 0)
        except Exception as e:
            log.debug(f"Windows microphone COM init note: {e}")

        try:
            mic = sc.default_microphone()
            with mic.recorder(samplerate=self.samplerate) as rec:
                while self.is_active:
                    data = rec.record(numframes=self.blocksize)
                    if data is not None and len(data) > 0:
                        mono = data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
                        self.mic_queue.put(mono)
        except Exception as e:
            log.warning(f"Windows microphone capture worker failed: {e}")
        finally:
            if hr == 0:
                try:
                    mf._ole32.CoUninitialize()
                except Exception as e:
                    log.debug(f"Windows microphone COM uninit note: {e}")

    def start_capture(self) -> None:
        if self.is_active:
            return
        import threading

        self.is_active = True
        while not self.queue.empty():
            self.queue.get_nowait()
        if self.mic_queue is not None:
            while not self.mic_queue.empty():
                self.mic_queue.get_nowait()

        self.sys_thread = threading.Thread(target=self._sys_worker, daemon=True)
        self.sys_thread.start()
        if self.include_microphone:
            self.mic_thread = threading.Thread(target=self._mic_worker, daemon=True)
            self.mic_thread.start()

    def stop_capture(self) -> None:
        self.is_active = False
        if self.sys_thread is not None:
            self.sys_thread.join(timeout=2.0)
            self.sys_thread = None
        if self.mic_thread is not None:
            self.mic_thread.join(timeout=2.0)
            self.mic_thread = None

    def get_audio_stream(self) -> Generator[tuple[str, np.ndarray], None, None]:
        from queue import Empty
        import time

        while (
            self.is_active
            or (self.sys_thread is not None and self.sys_thread.is_alive())
            or (self.mic_thread is not None and self.mic_thread.is_alive())
            or not self.queue.empty()
            or (self.mic_queue is not None and not self.mic_queue.empty())
        ):
            sys_drained = False
            try:
                while True:
                    yield ("sys", self.queue.get_nowait())
                    sys_drained = True
            except Empty:
                pass

            mic_drained = False
            if self.include_microphone and self.mic_queue is not None:
                try:
                    while True:
                        yield ("mic", self.mic_queue.get_nowait())
                        mic_drained = True
                except Empty:
                    pass

            if not sys_drained and not mic_drained:
                is_threads_alive = (
                    (self.sys_thread is not None and self.sys_thread.is_alive())
                    or (self.mic_thread is not None and self.mic_thread.is_alive())
                )
                if not self.is_active and not is_threads_alive:
                    break
                time.sleep(0.01)


def get_system_audio_capture(include_microphone: bool = False) -> SystemAudioStrategy:
    if sys.platform == "darwin":
        return MacOSSharedMemoryStrategy()
    elif sys.platform == "win32":
        return WindowsWasapiStrategy(include_microphone=include_microphone)
    else:
        return NativeLinuxWindowsStrategy(include_microphone=include_microphone)
