from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing, nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from core.context import session_context_manager
from stt.inference import (
    transcribe_and_detect_language_serialized,
    transcribe_batch_serialized,
    transcription_text,
)
from stt.media import (
    DecodedSampleLimitExceeded,
    DecodeStats,
    EncodedMedia,
    iter_media_frames,
    iter_media_pcm,
    media_duration,
)
from stt.models.base import STTModelAdapter


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


@dataclass(frozen=True)
class STTChunk:
    sequence: int
    source: str
    start_sec: float
    end_sec: float
    pcm: np.ndarray


class EmptyDecodedAudioError(ValueError):
    """A media input produced no PCM in a completed decode pass."""


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


def probe_media(input_path: str | Path) -> MediaInfo:
    try:
        source = Path(input_path)
        if not source.is_file():
            raise ValueError()
        with source.open("rb"):
            pass
    except (OSError, ValueError) as exc:
        raise ValueError("Local file is missing, unreadable, or not a regular file") from exc
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist", "file",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream=index,codec_name,codec_type,sample_rate,channels",
        "-of",
        "json",
        str(input_path),
    ]
    try:
        raw = subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=15.0)
    except (subprocess.SubprocessError, OSError) as exc:
        raise ValueError("Cannot probe local media; check that the file is still accessible and contains valid audio") from exc
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
    except (TypeError, ValueError):
        duration_sec = 0.0
    if not math.isfinite(duration_sec) or duration_sec < 0:
        duration_sec = 0.0

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
from .sequence_vad import get_sequence_vad_engine
from .stream_vad import (
    iter_file_pcm,
    pack_array_vad_chunks,
    pack_utterances_into_chunks,
    segment_scored_frames,
    stream_vad_chunks,
)


def iter_stt_chunks(
    raw_inputs: dict[str, Any],
    *,
    sample_rate: int,
    chunk_sec: int,
    total_duration_sec: float,
    max_samples: int | None = None,
    decode_stats: DecodeStats | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[STTChunk]:
    """Merge the existing per-source VAD streams in timestamp order."""
    import heapq

    generators = {}
    for source, audio in raw_inputs.items():
        if isinstance(audio, AudioMemoryBuffer):
            gen = pack_array_vad_chunks(audio.as_ndarray(), sample_rate=sample_rate, target_sec=chunk_sec)
        elif isinstance(audio, np.ndarray):
            gen = pack_array_vad_chunks(audio, sample_rate=sample_rate, target_sec=chunk_sec)
        elif isinstance(audio, EncodedMedia):
            frames = iter_media_frames(
                audio, sample_rate=sample_rate,
                max_samples=max_samples, stats=decode_stats,
            )
            scored_frames = get_sequence_vad_engine().new_stream().score_frames(frames)
            utterances = segment_scored_frames(
                scored_frames, sample_rate=sample_rate,
            )
            gen = pack_utterances_into_chunks(
                utterances, sample_rate=sample_rate, target_sec=chunk_sec,
                total_duration_sec=total_duration_sec,
            )
        else:
            gen = stream_vad_chunks(
                input_path=audio, sample_rate=sample_rate, target_sec=chunk_sec,
                total_duration_sec=total_duration_sec, max_samples=max_samples,
                decode_stats=decode_stats, cancel_check=cancel_check,
            )
        generators[source] = gen

    try:
        heap = []
        for source, gen in generators.items():
            try:
                item = next(gen)
                heapq.heappush(heap, (item[0], item[1], source, item, gen))
            except StopIteration:
                pass

        sequence = 0
        while heap:
            start, end, source, item, gen = heapq.heappop(heap)
            try:
                next_item = next(gen)
                heapq.heappush(heap, (next_item[0], next_item[1], source, next_item, gen))
            except StopIteration:
                pass
            yield STTChunk(sequence, source, start, end, item[2])
            sequence += 1
    finally:
        for gen in generators.values():
            gen.close()


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
    batch_size: int = 1,
    language: str | None = None,
    detect_language: bool = False,
) -> None:
    """Ordered STT runner with bounded model batches and source-specific publication."""
    start_time = time.time()
    cancelled_logged = False
    chunks = None

    def cancel_requested() -> bool:
        nonlocal cancelled_logged
        if jobs.is_cancelled(job_id):
            if not cancelled_logged:
                cancelled_logged = True
                elapsed = time.time() - start_time
                log.info(f"STT cancelled: job_id={job_id}; {elapsed:.2f}s")
                jobs.update_event(job_id, "cancelled", {})
            return True
        return False

    try:
        if cancel_requested():
            return

        raw_inputs = input_paths if isinstance(input_paths, dict) else {"default": input_paths}
        if not raw_inputs or not any(value is not None for value in raw_inputs.values()):
            raise ValueError("No audio recorded or empty audio stream")

        # Input can be an on-disk path, an in-memory AudioMemoryBuffer, or a raw np.ndarray
        first_input = next(iter(raw_inputs.values()))
        if isinstance(first_input, AudioMemoryBuffer):
            total_duration_sec = first_input.duration_sec
        elif isinstance(first_input, np.ndarray):
            total_duration_sec = len(first_input) / sample_rate
        elif isinstance(first_input, EncodedMedia):
            total_duration_sec = media_duration(first_input) or 0.0
        else:
            info = probe_media(first_input)
            total_duration_sec = info.duration_sec

        if max_duration_sec > 0 and total_duration_sec > max_duration_sec:
            raise ValueError(f"Audio too long (max {max_duration_sec}s)")

        # Both public batch modes decode one source; live multi-source jobs keep their existing policy.
        single_media_input = len(raw_inputs) == 1 and isinstance(first_input, (EncodedMedia, str, Path))
        max_samples = max_duration_sec * sample_rate if single_media_input and max_duration_sec > 0 else None
        stt_stats = DecodeStats() if single_media_input else None

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
                        # Diarization still retains complete decoded PCM, bounded by MAX_DURATION when set.
                        full_buffer = AudioMemoryBuffer(sample_rate=sample_rate)
                        diarization_stats = DecodeStats()
                        if isinstance(audio_source, EncodedMedia):
                            decoded = iter_media_pcm(
                                audio_source, sample_rate=sample_rate,
                                max_samples=max_samples, stats=diarization_stats,
                            )
                        else:
                            decoded = iter_file_pcm(
                                audio_source, sample_rate=sample_rate,
                                max_samples=max_samples, stats=diarization_stats,
                                cancel_check=cancel_requested,
                            )
                        with closing(decoded):
                            for pcm in decoded:
                                if cancel_requested():
                                    return
                                full_buffer.append(pcm)
                        if cancel_requested():
                            return
                        if diarization_stats.decoded_samples == 0:
                            raise EmptyDecodedAudioError("Audio too short or empty after decoding")
                        full_audio = full_buffer.as_ndarray()

                    if cancel_requested():
                        return
                    speaker_intervals_by_stream[stream_id] = diarize_audio(full_audio, cancel_check=cancel_requested)
                except subprocess.CalledProcessError as exc:
                    if cancel_requested() or exc.returncode in (255, 130, -2):
                        return
                    log.warning(f"Diarization failed for stream {stream_id}: {exc}")
                except (DecodedSampleLimitExceeded, EmptyDecodedAudioError):
                    raise
                except Exception as e:
                    if cancel_requested():
                        return
                    log.warning(f"Diarization failed for stream {stream_id}: {e}")

        if cancel_requested():
            return

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        chunks = iter_stt_chunks(
            raw_inputs, sample_rate=sample_rate, chunk_sec=chunk_sec,
            total_duration_sec=total_duration_sec, max_samples=max_samples,
            decode_stats=stt_stats, cancel_check=cancel_requested,
        )
        detected_language = None
        while True:
            if cancel_requested():
                return
            group = tuple(islice(chunks, 1 if detect_language and detected_language is None else batch_size))
            if not group:
                break
            if detect_language and detected_language is None:
                first_text, detected_language = transcribe_and_detect_language_serialized(
                    model, group[0].pcm
                )
                texts = [first_text]
            else:
                options = {"diarization": diarization} if isinstance(model, STTModelAdapter) else {}
                if language is not None or detect_language:
                    options["language"] = detected_language if detect_language else language
                texts = transcribe_batch_serialized(model, tuple(chunk.pcm for chunk in group), **options)

            if cancel_requested():
                return

            for chunk, raw in zip(group, texts, strict=True):
                if cancel_requested():
                    return
                start_sec, end_sec, stream_id = chunk.start_sec, chunk.end_sec, chunk.source
                text = transcription_text(raw)

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

                if hasattr(jobs, "get_status"):
                    status = jobs.get_status(job_id)
                    if status and "session_id" in status:
                        session_context_manager.append_session(
                            session_id=status["session_id"], text=text,
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

        if cancel_requested():
            return

        if stt_stats is not None and stt_stats.decoded_samples == 0:
            raise EmptyDecodedAudioError("Audio too short or empty after decoding")
        completed_duration_sec = (
            stt_stats.decoded_samples / sample_rate
            if stt_stats is not None and total_duration_sec == 0
            else total_duration_sec
        )
        jobs.update_event(job_id, "complete", {"duration": completed_duration_sec})
        elapsed = time.time() - start_time
        log.info(f"STT completed: job_id={job_id}; {elapsed:.2f}s")
    except Exception as exc:
        if cancel_requested():
            return
        elapsed = time.time() - start_time
        message = (
            f"Audio too long (max {max_duration_sec}s)"
            if isinstance(exc, DecodedSampleLimitExceeded)
            else str(exc)
        )
        if isinstance(exc, (DecodedSampleLimitExceeded, EmptyDecodedAudioError)):
            log.warning(f"STT input rejected: job_id={job_id}; {message}")
        else:
            log.exception(f"STT failed: job_id={job_id}; {message} ({elapsed:.2f}s)")
        jobs.update_event(job_id, "error", {"message": message})
    finally:
        if chunks is not None:
            chunks.close()


def run_stt_worker(
    *,
    job_id: str,
    audio_path: str | EncodedMedia | dict[str, str | AudioMemoryBuffer | EncodedMedia],
    semaphore: threading.BoundedSemaphore | None = None,
    jobs: Any,
    model: Any,
    log: Any,
    sample_rate: int,
    chunk_sec: int,
    max_duration_sec: int = 0,
    diarization: bool = False,
    batch_size: int = 1,
    language: str | None = None,
    detect_language: bool = False,
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
            batch_size=batch_size,
            language=language,
            detect_language=detect_language,
        )
