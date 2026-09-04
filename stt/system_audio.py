from __future__ import annotations

import abc
import logging
import mmap
import os
import shutil
import struct
import sys
import time
from collections.abc import Generator
from typing import Any

import numpy as np

log = logging.getLogger("resonance.stt.system_audio")


class SystemAudioStrategy(abc.ABC):
    @abc.abstractmethod
    def start_capture(self) -> None:
        pass

    @abc.abstractmethod
    def stop_capture(self) -> None:
        pass

    @abc.abstractmethod
    def get_audio_stream(self) -> Generator[tuple[str, np.ndarray], None, None]:
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

    def __init__(self, include_microphone: bool = False):
        import posix_ipc

        self.include_microphone = include_microphone

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

        (_, _, self.rate, self.channels, self.frames_per_slot, self.slot_count,
         _, _, _, _) = struct.unpack(self._HEADER_FMT, bytes(self.shm[:self._HEADER_SIZE]))
        self.bytes_per_slot = self.frames_per_slot * (self.channels or 1) * 4
        self.read_idx = 0

    def start_capture(self) -> None:
        # Caller obligation: Swift must update header.status within _START_TIMEOUT seconds.
        cmd_val = 3 if self.include_microphone else 1
        struct.pack_into('<I', self.shm, self._CMD_OFFSET, cmd_val)
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

    def get_audio_stream(self) -> Generator[tuple[str, np.ndarray], None, None]:
        while True:
            self.data_sem.acquire()

            (_, _, _, channels, frames_per_slot, slot_count, current_write_idx,
             current_cmd, _, _) = struct.unpack(self._HEADER_FMT, bytes(self.shm[:self._HEADER_SIZE]))

            if current_cmd == 2:
                break

            # Tail-drop mitigation: STT inference is slower than real-time capture.
            if current_write_idx > self.read_idx + slot_count:
                self.read_idx = current_write_idx - 1

            slot_idx = self.read_idx % slot_count
            bytes_per_slot = frames_per_slot * (channels or 1) * 4
            offset   = self._HEADER_SIZE + (slot_idx * bytes_per_slot)

            if channels == 2:
                # Invariant: Interleaved layout where even frames are system audio and odd frames are microphone.
                interleaved = np.ndarray(
                    (frames_per_slot * 2,),
                    dtype=np.float32,
                    buffer=self.shm,
                    offset=offset,
                )
                sys_chunk = interleaved[0::2].copy()
                mic_chunk = interleaved[1::2].copy()
                self.read_idx += 1
                yield ("sys", sys_chunk)
                yield ("mic", mic_chunk)
            else:
                # Zero-copy invariant: offset must map to a contiguous SHM block.
                audio_chunk = np.ndarray(
                    (frames_per_slot,),
                    dtype=np.float32,
                    buffer=self.shm,
                    offset=offset,
                )
                self.read_idx += 1
                yield ("sys", audio_chunk)


class SoundcardSystemAudioStrategy(SystemAudioStrategy):
    """Captures desktop audio via Loopback Monitor and hardware microphone using soundcard."""

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
        hr = -1
        if sys.platform.startswith("win"):
            try:
                import soundcard.mediafoundation as mf
                hr = mf._ole32.CoInitializeEx(mf._ffi.NULL, 0)
            except Exception as e:
                log.debug(f"Windows system audio COM init note: {e}")

        try:
            import soundcard as sc
            speaker = sc.default_speaker()
            loopback_id = f"{speaker.id}.monitor" if (sys.platform.startswith("linux") and not str(speaker.id).endswith(".monitor")) else str(speaker.id)
            loopback_mic = sc.get_microphone(id=loopback_id, include_loopback=True)
            with loopback_mic.recorder(samplerate=self.samplerate) as rec:
                while self.is_active:
                    data = rec.record(numframes=self.blocksize)
                    if data is not None and len(data) > 0:
                        mono = data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
                        self.queue.put(mono)
        except Exception as e:
            log.warning(f"System audio capture worker failed: {e}")
        finally:
            if hr == 0:
                try:
                    mf._ole32.CoUninitialize()
                except Exception as e:
                    log.debug(f"Windows system audio COM uninit note: {e}")

    def _mic_worker(self) -> None:
        hr = -1
        if sys.platform.startswith("win"):
            try:
                import soundcard.mediafoundation as mf
                hr = mf._ole32.CoInitializeEx(mf._ffi.NULL, 0)
            except Exception as e:
                log.debug(f"Windows microphone COM init note: {e}")

        try:
            import soundcard as sc
            mic = sc.default_microphone()
            with mic.recorder(samplerate=self.samplerate) as rec:
                while self.is_active:
                    data = rec.record(numframes=self.blocksize)
                    if data is not None and len(data) > 0:
                        mono = data.mean(axis=1).astype(np.float32) if data.ndim > 1 else data.astype(np.float32)
                        self.mic_queue.put(mono)
        except Exception as e:
            log.warning(f"Microphone capture worker failed: {e}")
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
        import time
        from queue import Empty

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


class LinuxPulseParecStrategy(SystemAudioStrategy):
    """Captures Linux desktop audio and microphone via PulseAudio/PipeWire parec streams."""

    def __init__(self, include_microphone: bool = False):
        from queue import Queue

        self.include_microphone = include_microphone
        self.queue = Queue()
        self.mic_queue = Queue() if include_microphone else None
        self.is_active = False
        self.samplerate = 16000
        self.blocksize = 4096
        self.proc_sys = None
        self.proc_mic = None
        self.threads = []

    def _reader_worker(self, proc: Any, queue: Any) -> None:
        chunk_bytes = self.blocksize * 4  # float32 = 4 bytes
        try:
            while self.is_active and proc.stdout is not None:
                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    break
                samples = np.frombuffer(raw, dtype=np.float32)
                if len(samples) > 0:
                    queue.put(samples)
        except Exception as e:
            log.debug(f"Linux parec reader note: {e}")

    def start_capture(self) -> None:
        if self.is_active:
            return

        import subprocess
        import threading

        sink = subprocess.check_output(["pactl", "get-default-sink"], text=True, timeout=2.0).strip()
        monitor = f"{sink}.monitor" if not sink.endswith(".monitor") else sink

        self.is_active = True
        while not self.queue.empty():
            self.queue.get_nowait()
        if self.mic_queue is not None:
            while not self.mic_queue.empty():
                self.mic_queue.get_nowait()

        self.proc_sys = subprocess.Popen(
            [
                "parec",
                f"--device={monitor}",
                f"--rate={self.samplerate}",
                "--channels=1",
                "--format=float32le",
                "--latency-msec=50",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        t_sys = threading.Thread(target=self._reader_worker, args=(self.proc_sys, self.queue), daemon=True)
        t_sys.start()
        self.threads.append(t_sys)

        if self.include_microphone:
            try:
                source = subprocess.check_output(["pactl", "get-default-source"], text=True, timeout=2.0).strip()
                self.proc_mic = subprocess.Popen(
                    [
                        "parec",
                        f"--device={source}",
                        f"--rate={self.samplerate}",
                        "--channels=1",
                        "--format=float32le",
                        "--latency-msec=50",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                t_mic = threading.Thread(target=self._reader_worker, args=(self.proc_mic, self.mic_queue), daemon=True)
                t_mic.start()
                self.threads.append(t_mic)
            except Exception as e:
                log.warning(f"Microphone capture start failed: {e}")
                self.proc_mic = None

    def stop_capture(self) -> None:
        self.is_active = False
        if self.proc_sys:
            try:
                self.proc_sys.terminate()
                self.proc_sys.wait(timeout=1.0)
            except Exception:
                self.proc_sys.kill()
            self.proc_sys = None

        if self.proc_mic:
            try:
                self.proc_mic.terminate()
                self.proc_mic.wait(timeout=1.0)
            except Exception:
                self.proc_mic.kill()
            self.proc_mic = None

        for t in self.threads:
            t.join(timeout=1.0)
        self.threads.clear()

    def get_audio_stream(self) -> Generator[tuple[str, np.ndarray], None, None]:
        import time
        from queue import Empty

        while (
            self.is_active
            or any(t.is_alive() for t in self.threads)
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
                if not self.is_active and not any(t.is_alive() for t in self.threads):
                    break
                time.sleep(0.01)


# Aliases for backward compatibility in tests and platform dispatch
WindowsWasapiStrategy = SoundcardSystemAudioStrategy
NativeLinuxWindowsStrategy = LinuxPulseParecStrategy


def get_system_audio_capture(include_microphone: bool = False) -> SystemAudioStrategy:
    if sys.platform == "darwin":
        return MacOSSharedMemoryStrategy(include_microphone=include_microphone)
    elif sys.platform.startswith("linux") and shutil.which("parec"):
        return LinuxPulseParecStrategy(include_microphone=include_microphone)
    return SoundcardSystemAudioStrategy(include_microphone=include_microphone)
