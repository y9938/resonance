from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MediaInfo:
    duration_sec: float
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    size_bytes: int | None


@dataclass(frozen=True)
class SegmentSpec:
    index: int
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def _segment_text_from_transcribe(result: Any) -> str:
    """Normalize GigaAM transcribe() output to a plain string for JSON/SSE."""
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


_DEFAULT_STT_TRANSCRIBE_MAX_SEC = 25.0


def stt_transcribe_hard_limit_sec() -> float:
    """
    GigaAM transcribe() hard cap in seconds.
    Read from gigaam.model.LONGFORM_THRESHOLD when available.
    """
    try:
        from gigaam.model import LONGFORM_THRESHOLD  # type: ignore
        from gigaam.preprocess import SAMPLE_RATE  # type: ignore

        if SAMPLE_RATE > 0:
            return float(LONGFORM_THRESHOLD) / float(SAMPLE_RATE)
    except Exception:
        pass
    return _DEFAULT_STT_TRANSCRIBE_MAX_SEC


async def save_upload_to_path(
    upload: Any,
    destination: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
    max_bytes: int = 0,
) -> int:
    total = 0
    path = Path(destination)

    with path.open("wb") as handle:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes > 0 and total > max_bytes:
                raise ValueError(f"File too large (max {max_bytes // (1024 * 1024)}MB)")
            handle.write(chunk)

    return total


def _decode_duration(input_path: str | Path) -> float:
    """Fallback for browser WebM files where MediaRecorder omits duration from the container header."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(input_path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=15.0,
    )
    matches = re.findall(r"time=(\d+:\d+:\d+\.\d+)", raw.stderr)
    if not matches:
        return 0.0
    h, m, s = matches[-1].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def probe_media(input_path: str | Path) -> MediaInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream=index,codec_name,codec_type,sample_rate,channels",
        "-of",
        "json",
        str(input_path),
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=15.0)
    payload = json.loads(raw.stdout or "{}")
    streams = payload.get("streams") or []
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if not audio_stream:
        raise ValueError("No audio stream found")

    fmt = payload.get("format") or {}
    try:
        duration_sec = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid media duration") from exc
    if duration_sec <= 0:
        duration_sec = _decode_duration(input_path)
    if duration_sec <= 0:
        raise ValueError("Audio too short or empty after probing")

    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return MediaInfo(
        duration_sec=duration_sec,
        codec_name=audio_stream.get("codec_name"),
        sample_rate=_optional_int(audio_stream.get("sample_rate")),
        channels=_optional_int(audio_stream.get("channels")),
        size_bytes=_optional_int(fmt.get("size")),
    )


from .stream_vad import stream_vad_chunks
import soundfile as sf

def run_stt_job(
    *,
    job_id: str,
    input_path: str,
    upload_root: str,
    jobs: Any,
    model: Any,
    log: Any,
    sample_rate: int,
    chunk_sec: int,
    max_duration_sec: int = 0,
    diarization: bool = False,
) -> None:
    """Sequential STT runner using RAM Streaming to avoid disk thrashing and frame drift."""
    start_time = time.time()
    segment_root = tempfile.mkdtemp(dir=upload_root)

    def cancel_requested() -> bool:
        if jobs.is_cancelled(job_id):
            elapsed = time.time() - start_time
            log.info(f"STT cancelled: {elapsed:.2f}s")
            jobs.update_event(job_id, "cancelled", {})
            return True
        return False

    try:
        if cancel_requested():
            return

        if not input_path:
            raise ValueError("No audio recorded or empty audio path")
        info = probe_media(input_path)
        if max_duration_sec > 0 and info.duration_sec > max_duration_sec:
            raise ValueError(f"Audio too long (max {max_duration_sec}s)")

        jobs.update_event(
            job_id,
            "start",
            {
                "duration": info.duration_sec,
                "total": round(info.duration_sec, 2),
            },
        )

        for chunk_index, (start_sec, end_sec, chunk_array) in enumerate(stream_vad_chunks(
            input_path=input_path,
            sample_rate=sample_rate,
            target_sec=chunk_sec,
            total_duration_sec=info.duration_sec,
        )):
            if cancel_requested():
                return

            import inspect
            sig = inspect.signature(model.transcribe)
            if "diarization" in sig.parameters:
                raw = model.transcribe(chunk_array, diarization=diarization)
            else:
                raw = model.transcribe(chunk_array)

            if cancel_requested():
                return

            # Assumes: VAD yields exact absolute timestamps.
            jobs.update_event(
                job_id,
                "progress",
                {
                    "current": round(end_sec, 2),
                    "total": round(info.duration_sec, 2),
                    "segment": {
                        "start": round(start_sec, 6),
                        "end": round(end_sec, 6),
                        "text": _segment_text_from_transcribe(raw),
                    },
                },
            )

        if cancel_requested():
            return

        jobs.update_event(job_id, "complete", {"duration": info.duration_sec})
        elapsed = time.time() - start_time
        log.info(f"STT completed: {elapsed:.2f}s")
    except Exception as exc:
        elapsed = time.time() - start_time
        log.error(f"STT failed: {exc} ({elapsed:.2f}s)")
        jobs.update_event(job_id, "error", {"message": str(exc)})
    finally:
        shutil.rmtree(upload_root, ignore_errors=True)


def run_stt_worker(
    *,
    job_id: str,
    audio_path: str,
    upload_root: str,
    semaphore: threading.BoundedSemaphore,
    jobs: Any,
    model: Any,
    log: Any,
    sample_rate: int,
    chunk_sec: int,
    max_duration_sec: int = 0,
    diarization: bool = False,
) -> None:
    with semaphore:
        model_class = model.__class__.__name__
        if model_class not in ("WhisperAdapter", "GraniteAdapter"):
            effective_chunk_sec = min(chunk_sec, int(stt_transcribe_hard_limit_sec()))
        else:
            effective_chunk_sec = chunk_sec

        run_stt_job(
            job_id=job_id,
            input_path=audio_path,
            upload_root=upload_root,
            jobs=jobs,
            model=model,
            log=log,
            sample_rate=sample_rate,
            chunk_sec=effective_chunk_sec,
            max_duration_sec=max_duration_sec,
            diarization=diarization,
        )
