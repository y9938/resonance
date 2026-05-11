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
        ["ffmpeg", "-i", str(input_path), "-f", "null", "-"],
        capture_output=True,
        text=True,
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
    raw = subprocess.run(cmd, check=True, capture_output=True, text=True)
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


def plan_segments(
    *,
    duration_sec: float,
    chunk_sec: int,
    overlap_sec: int,
) -> list[SegmentSpec]:
    if duration_sec <= 0:
        raise ValueError("Audio too short or empty after probing")
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be non-negative")
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    step_sec = chunk_sec - overlap_sec
    total_segments = max(1, math.ceil(max(duration_sec - chunk_sec, 0.0) / step_sec) + 1)
    segments: list[SegmentSpec] = []
    for index in range(total_segments):
        start_sec = index * step_sec
        if start_sec >= duration_sec:
            break
        end_sec = min(start_sec + chunk_sec, duration_sec)
        segments.append(
            SegmentSpec(
                index=index + 1,
                start_sec=round(start_sec, 6),
                end_sec=round(end_sec, 6),
            )
        )
    if not segments:
        raise ValueError("Audio too short or empty after probing")
    return segments


def extract_segment_ffmpeg(
    input_path: str | Path,
    segment: SegmentSpec,
    output_path: str | Path,
    *,
    sample_rate: int,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{segment.start_sec:.6f}",
        "-i",
        str(input_path),
        "-t",
        f"{segment.duration_sec:.6f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


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
    overlap_sec: int,
    max_duration_sec: int = 0,
) -> None:
    """Sequential STT runner with one on-disk segment at a time."""
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

        info = probe_media(input_path)
        if max_duration_sec > 0 and info.duration_sec > max_duration_sec:
            raise ValueError(f"Audio too long (max {max_duration_sec}s)")

        segments = plan_segments(
            duration_sec=info.duration_sec,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
        )

        if cancel_requested():
            return

        jobs.update_event(
            job_id,
            "start",
            {"total": len(segments), "duration": info.duration_sec},
        )

        for segment in segments:
            if cancel_requested():
                return

            segment_path = os.path.join(segment_root, f"segment_{segment.index:06d}.wav")
            try:
                extract_segment_ffmpeg(
                    input_path,
                    segment,
                    segment_path,
                    sample_rate=sample_rate,
                )
                if cancel_requested():
                    return

                raw = model.transcribe(segment_path)
                if cancel_requested():
                    return
            finally:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(segment_path)

            jobs.update_event(
                job_id,
                "progress",
                {
                    "current": segment.index,
                    "total": len(segments),
                    "segment": {
                        "start": segment.start_sec,
                        "end": segment.end_sec,
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
    overlap_sec: int,
    max_duration_sec: int = 0,
) -> None:
    with semaphore:
        run_stt_job(
            job_id=job_id,
            input_path=audio_path,
            upload_root=upload_root,
            jobs=jobs,
            model=model,
            log=log,
            sample_rate=sample_rate,
            chunk_sec=min(chunk_sec, int(stt_transcribe_hard_limit_sec())),
            overlap_sec=overlap_sec,
            max_duration_sec=max_duration_sec,
        )
