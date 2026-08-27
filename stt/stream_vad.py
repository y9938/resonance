import subprocess
import numpy as np
import logging
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)

def stream_audio_with_overlap(
    input_path: str,
    chunk_sec: float,
    overlap_sec: float,
    sample_rate: int = 16000
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Streams audio from a file via FFmpeg stdout pipe into RAM, yielding chunks.
    Replaces O(N) iterative FFmpeg fast-seek calls with a single O(1) process.

    Args:
        input_path: Path to the media file.
        chunk_sec: Duration of each chunk in seconds.
        overlap_sec: Duration of the overlap at the end of each chunk in seconds.
        sample_rate: Target sample rate for STT models (usually 16000).

    Yields:
        Tuple of (chunk_index, chunk_array_int16)
    """
    bytes_per_sec = sample_rate * 2  # 16-bit PCM (2 bytes per sample) mono
    chunk_bytes = int(chunk_sec * bytes_per_sec)
    overlap_bytes = int(overlap_sec * bytes_per_sec)
    advance_bytes = chunk_bytes - overlap_bytes

    # FAIL-FAST: Prevent infinite loops
    assert advance_bytes > 0, "Overlap must be strictly less than chunk_sec"
    assert chunk_sec > 0, "chunk_sec must be positive"

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        raise RuntimeError(f"Failed to start FFmpeg for streaming: {e}")

    buffer = b""
    chunk_index = 0

    try:
        while True:
            needed = chunk_bytes - len(buffer)
            if needed > 0:
                data = proc.stdout.read(needed)
                if not data:
                    # EOF
                    if buffer:
                        yield chunk_index, np.frombuffer(buffer, dtype=np.int16)
                    break
                buffer += data

                # If we still didn't fill the buffer (ffmpeg pipe yielded partial), read again
                if len(buffer) < chunk_bytes:
                    continue

            # We have a full chunk
            yield chunk_index, np.frombuffer(buffer, dtype=np.int16)

            # Advance the buffer, leaving the overlap behind
            buffer = buffer[advance_bytes:]
            chunk_index += 1

        # FAIL-FAST: Ensure ffmpeg terminated successfully
        proc.wait(timeout=5)
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode().strip()
            raise RuntimeError(f"FFmpeg streaming failed (code {proc.returncode}): {stderr}")

    finally:
        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()
        proc.terminate()
