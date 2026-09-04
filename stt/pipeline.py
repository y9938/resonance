from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.context import session_context_manager


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
    except (ImportError, AttributeError, ValueError):
        return _DEFAULT_STT_TRANSCRIBE_MAX_SEC
    return _DEFAULT_STT_TRANSCRIBE_MAX_SEC


def _decode_duration(input_path: str | Path) -> float:
    """Fallback for browser WebM files where MediaRecorder omits duration from the container header."""
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", str(input_path), "-f", "null", "-"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=15.0,
        check=False,
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


from .buffer import AudioMemoryBuffer
from .stream_vad import pack_array_vad_chunks, stream_vad_chunks


def run_stt_job(
    *,
    job_id: str,
    input_paths: str | dict[str, Any],
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
    cancelled_logged = False

    def cancel_requested() -> bool:
        nonlocal cancelled_logged
        if jobs.is_cancelled(job_id):
            if not cancelled_logged:
                cancelled_logged = True
                elapsed = time.time() - start_time
                log.info(f"STT cancelled: {elapsed:.2f}s")
                jobs.update_event(job_id, "cancelled", {})
            return True
        return False

    try:
        if cancel_requested():
            return

        raw_inputs = input_paths if isinstance(input_paths, dict) else {"default": input_paths}
        if not raw_inputs or not any(raw_inputs.values()):
            raise ValueError("No audio recorded or empty audio stream")

        # Input can be an on-disk path, an in-memory AudioMemoryBuffer, or a raw np.ndarray
        first_input = next(iter(raw_inputs.values()))
        if isinstance(first_input, AudioMemoryBuffer):
            total_duration_sec = first_input.duration_sec
        elif isinstance(first_input, np.ndarray):
            total_duration_sec = len(first_input) / sample_rate
        else:
            info = probe_media(first_input)
            total_duration_sec = info.duration_sec

        if max_duration_sec > 0 and total_duration_sec > max_duration_sec:
            raise ValueError(f"Audio too long (max {max_duration_sec}s)")

        model_class = model.__class__.__name__
        is_diarizing = bool(diarization and model_class != "GraniteAdapter")
        jobs.update_event(
            job_id,
            "start",
            {
                "duration": total_duration_sec,
                "total": round(total_duration_sec, 2),
                "stage": "diarization" if is_diarizing else "transcription",
            },
        )

        speaker_intervals_by_stream = {}
        if is_diarizing:
            for stream_id, audio_source in raw_inputs.items():
                if stream_id == "mic":
                    continue  # Domain Invariant: Microphone stream is always a single known speaker (Me).

                if cancel_requested():
                    return
                try:
                    from .diarization import diarize_audio, match_speaker_tag

                    if isinstance(audio_source, AudioMemoryBuffer):
                        full_audio = audio_source.as_ndarray()
                    elif isinstance(audio_source, np.ndarray):
                        full_audio = audio_source
                    else:
                        full_raw = subprocess.check_output([
                            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                            "-i", str(audio_source), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"
                        ], timeout=min(600.0, max(30.0, total_duration_sec * 2)))
                        full_audio = np.frombuffer(full_raw, dtype=np.int16).astype(np.float32) / 32768.0

                    if cancel_requested():
                        return
                    speaker_intervals_by_stream[stream_id] = diarize_audio(full_audio, cancel_check=cancel_requested)
                except subprocess.CalledProcessError as exc:
                    if cancel_requested() or exc.returncode in (255, 130, -2):
                        return
                    log.warning(f"Diarization failed for stream {stream_id}: {exc}")
                except Exception as e:
                    if cancel_requested():
                        return
                    log.warning(f"Diarization failed for stream {stream_id}: {e}")

        if cancel_requested():
            return

        import heapq

        generators = {}
        for sid, audio_source in raw_inputs.items():
            if isinstance(audio_source, AudioMemoryBuffer):
                generators[sid] = pack_array_vad_chunks(
                    audio_source.as_ndarray(),
                    sample_rate=sample_rate,
                    target_sec=chunk_sec,
                )
            elif isinstance(audio_source, np.ndarray):
                generators[sid] = pack_array_vad_chunks(
                    audio_source,
                    sample_rate=sample_rate,
                    target_sec=chunk_sec,
                )
            else:
                generators[sid] = stream_vad_chunks(
                    input_path=audio_source,
                    sample_rate=sample_rate,
                    target_sec=chunk_sec,
                    total_duration_sec=total_duration_sec,
                )

        heap = []
        for sid, gen in generators.items():
            try:
                item = next(gen)
                heapq.heappush(heap, (item[0], item[1], sid, item, gen))
            except StopIteration:
                pass

        chunk_index = 0
        while heap:
            if cancel_requested():
                return

            start_sec, end_sec, stream_id, item, gen = heapq.heappop(heap)
            _, _, chunk_array = item

            try:
                next_item = next(gen)
                heapq.heappush(heap, (next_item[0], next_item[1], stream_id, next_item, gen))
            except StopIteration:
                pass

            import inspect
            sig = inspect.signature(model.transcribe)
            if "diarization" in sig.parameters and model_class == "GraniteAdapter":
                raw = model.transcribe(chunk_array, diarization=diarization)
            else:
                raw = model.transcribe(chunk_array)

            if cancel_requested():
                return

            text = _segment_text_from_transcribe(raw)

            if stream_id == "mic":
                text = f"[SOURCE:MIC]: {text}"
            elif diarization and not text.startswith("[Speaker"):
                intervals = speaker_intervals_by_stream.get(stream_id, [])
                if intervals:
                    from .diarization import match_speaker_tag
                    tag = match_speaker_tag(start_sec, end_sec, intervals)
                    text = f"{tag}{text}"
                elif "mic" in raw_inputs:
                    text = f"[SOURCE:SYS]: {text}"
            elif "mic" in raw_inputs or (len(raw_inputs) > 1 and stream_id == "sys"):
                text = f"[SOURCE:SYS]: {text}"

            # Assumes: VAD yields exact absolute timestamps.
            if hasattr(jobs, "get_status"):
                status = jobs.get_status(job_id)
                if status and "session_id" in status:
                    session_context_manager.append(
                        session_id=status["session_id"],
                        text=text,
                        start_sec=start_sec,
                        end_sec=end_sec,
                    )

            jobs.update_event(
                job_id,
                "progress",
                {
                    "current": round(end_sec, 2),
                    "total": round(total_duration_sec, 2),
                    "segment": {
                        "start": round(start_sec, 6),
                        "end": round(end_sec, 6),
                        "text": text,
                        "source": stream_id,
                    },
                },
            )
            chunk_index += 1

        if cancel_requested():
            return

        jobs.update_event(job_id, "complete", {"duration": total_duration_sec})
        elapsed = time.time() - start_time
        log.info(f"STT completed: {elapsed:.2f}s")
    except Exception as exc:
        if cancel_requested():
            return
        elapsed = time.time() - start_time
        log.error(f"STT failed: {exc} ({elapsed:.2f}s)")
        jobs.update_event(job_id, "error", {"message": str(exc)})


def run_stt_worker(
    *,
    job_id: str,
    audio_path: str | dict[str, str] | dict[str, AudioMemoryBuffer],
    semaphore: threading.BoundedSemaphore | None = None,
    jobs: Any,
    model: Any,
    log: Any,
    sample_rate: int,
    chunk_sec: int,
    max_duration_sec: int = 0,
    diarization: bool = False,
) -> None:
    # Workaround: Optional semaphore allows interactive real-time jobs to bypass batch queue throttling.
    sync_context = semaphore if semaphore is not None else nullcontext()
    with sync_context:
        model_class = model.__class__.__name__
        if model_class not in ("WhisperAdapter", "GraniteAdapter"):
            effective_chunk_sec = min(chunk_sec, int(stt_transcribe_hard_limit_sec()))
        else:
            effective_chunk_sec = chunk_sec

        run_stt_job(
            job_id=job_id,
            input_paths=audio_path,
            jobs=jobs,
            model=model,
            log=log,
            sample_rate=sample_rate,
            chunk_sec=effective_chunk_sec,
            max_duration_sec=max_duration_sec,
            diarization=diarization,
        )
