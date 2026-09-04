from __future__ import annotations

import numpy as np


class AudioMemoryBuffer:
    """
    Zero-I/O growable contiguous in-RAM PCM buffer.
    Backing storage: C-level bytearray with zero-copy numpy projection.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        # Invariant: Stores float32 little-endian PCM bytes (4 bytes per sample).
        self._data = bytearray()

    def append(self, chunk: np.ndarray) -> None:
        """
        Caller obligation: chunk must be 1D float32 array or contiguous buffer.
        """
        if not isinstance(chunk, np.ndarray):
            chunk = np.asarray(chunk, dtype=np.float32)
        elif chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        # C-level amortized O(1) append into contiguous buffer
        self._data.extend(chunk.tobytes())

    def as_ndarray(self) -> np.ndarray:
        """
        Returns a 1D float32 numpy view directly referencing the backing bytearray without copying.
        """
        if not self._data:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(self._data, dtype=np.float32)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        """Number of float32 samples in the buffer."""
        return len(self._data) // 4

    @property
    def duration_sec(self) -> float:
        return len(self) / self.sample_rate


def _preload_bundled_ffmpeg() -> None:
    import ctypes
    import os
    import sys
    from pathlib import Path

    base_deps = Path(__file__).parent.parent / ".deps" / "ffmpeg7"

    if sys.platform.startswith("win"):
        bin_dir = base_deps / "bin"
        if bin_dir.is_dir():
            str_bin = str(bin_dir)
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str_bin)
            if str_bin not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{str_bin};{os.environ.get('PATH', '')}"
        return

    deps_dir = base_deps / "lib"
    if not deps_dir.is_dir():
        return

    libs = [
        "libavutil.so.59",
        "libswresample.so.5",
        "libswscale.so.8",
        "libavcodec.so.61",
        "libavformat.so.61",
        "libavfilter.so.10",
        "libavdevice.so.61",
    ]
    for lib_name in libs:
        lib_path = deps_dir / lib_name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


def decode_media_bytes(
    raw_bytes: bytes,
    target_sample_rate: int = 16000,
) -> AudioMemoryBuffer:
    """
    Decodes in-memory audio/video container bytes directly into an AudioMemoryBuffer using TorchCodec.
    """
    _preload_bundled_ffmpeg()
    try:
        from torchcodec.decoders import AudioDecoder
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "TorchCodec is required for audio decoding. "
            "Ensure torchcodec is installed and FFmpeg shared libraries (4..7) are in LD_LIBRARY_PATH. "
            "Run scripts/download_ffmpeg7.sh to install a local FFmpeg 7 build."
        ) from exc

    try:
        decoder = AudioDecoder(raw_bytes, sample_rate=target_sample_rate, num_channels=1)
        samples = decoder.get_all_samples().data
        audio_1d = samples.squeeze(0).cpu().numpy().astype(np.float32)
        buf = AudioMemoryBuffer(sample_rate=target_sample_rate)
        buf.append(audio_1d)
        return buf
    except Exception as exc:
        raise ValueError(f"Failed to decode media container: {exc}") from exc

