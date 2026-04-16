#!/usr/bin/env python3
"""
Resonance API Server - Unified STT/TTS Service

Architecture:
- FastAPI for HTTP API
- asyncio.to_thread for blocking model inference (STT/TTS)
- SSE streaming via StreamBridge
- Graceful shutdown with resource cleanup
"""

from __future__ import annotations

import os
import re
import sys
import json
import shutil
import queue
import copy
import logging
import secrets
import tempfile
import subprocess
import threading
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Any
from contextlib import asynccontextmanager, suppress

from dotenv import load_dotenv

import torch
import torchaudio
import torchaudio.transforms as T
from scipy.io import wavfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class Config:
    SR: int = int(os.getenv("RESONANCE_SR", "16000"))
    CHUNK_SEC: int = int(os.getenv("RESONANCE_CHUNK_SEC", "20"))
    OVERLAP_SEC: int = int(os.getenv("RESONANCE_OVERLAP_SEC", "2"))
    TTS_SR: int = int(os.getenv("RESONANCE_TTS_SR", "48000"))
    TTS_SPEAKER: str = os.getenv("RESONANCE_TTS_SPEAKER", "ru_roman")
    TTS_MAX_CHARS: int = int(os.getenv("RESONANCE_TTS_MAX_CHARS", "600"))
    TTS_MAX_INPUT_CHARS: int = int(os.getenv("RESONANCE_TTS_MAX_INPUT_CHARS", "0"))
    MAX_WORKERS: int = int(os.getenv("RESONANCE_MAX_WORKERS", "2"))
    UPLOAD_LIMIT_MB: int = int(os.getenv("RESONANCE_UPLOAD_LIMIT_MB", "0"))
    TTS_FILE_TTL_SEC: int = int(os.getenv("RESONANCE_TTS_FILE_TTL_SEC", "5400"))
    TTS_SWEEP_INTERVAL_SEC: int = int(
        os.getenv("RESONANCE_TTS_SWEEP_INTERVAL_SEC", "900")
    )
TTS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "resonance-tts"


def cors_allow_origins() -> list[str]:
    raw = os.getenv("RESONANCE_CORS_ORIGINS", "http://localhost:8000").strip()
    if not raw:
        return ["http://localhost:8000"]
    if raw == "*":
        return ["*"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins if origins else ["http://localhost:8000"]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("resonance")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)

    logger.propagate = False
    return logger


log = setup_logging()

# Disable uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").disabled = True


# -----------------------------------------------------------------------------
# Model Manager
# -----------------------------------------------------------------------------


class ModelManager:
    """Lazy model loader. Assumes: single-threaded init, thread-safe after."""

    def __init__(self) -> None:
        self._stt: Any | None = None
        self._tts: Any | None = None
        self._lock = threading.Lock()

    @property
    def stt_loaded(self) -> bool:
        return self._stt is not None

    @property
    def tts_loaded(self) -> bool:
        return self._tts is not None

    def stt(self) -> Any:
        with self._lock:
            if self._stt is None:
                self._stt = _load_stt()
            return self._stt

    def tts(self) -> Any:
        with self._lock:
            if self._tts is None:
                self._tts = _load_tts()
            return self._tts


def _load_stt() -> Any:
    log.info("Loading STT model (GigaAM-v3)...")
    import gigaam

    device = os.getenv("DEVICE", "cpu")
    model = gigaam.load_model("v3_e2e_ctc", device=device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"STT model loaded: {params:.1f}M parameters")
    return model


def _load_tts() -> Any:
    log.info("Loading TTS model (Silero v5_cis_base)...")

    device = os.getenv("DEVICE", "cpu")

    model, _ = torch.hub.load(
        "snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_cis_base",
        trust_repo=True,
    )
    model.to(torch.device(device))
    log.info("TTS model loaded")
    return model


models = ModelManager()


# -----------------------------------------------------------------------------
# Audio Processing
# -----------------------------------------------------------------------------


def extract_audio_ffmpeg(input_path: str, output_path: str) -> None:
    """
    Caller must ensure: ffmpeg installed and in PATH.
    Raises: subprocess.CalledProcessError on failure.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(Config.SR),
        "-ac",
        "1",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def load_audio(input_path: str) -> tuple[torch.Tensor, int]:
    """
    Returns: (waveform, sample_rate)
    Note: Output always mono, resampled to Config.SR.
    """
    try:
        wav, sr = torchaudio.load_with_torchcodec(input_path)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            extract_audio_ffmpeg(input_path, tmp_path)
            wav, sr = torchaudio.load_with_torchcodec(tmp_path)
        finally:
            os.unlink(tmp_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != Config.SR:
        resampler = T.Resample(sr, Config.SR)
        wav = resampler(wav)

    return wav, Config.SR


def split_audio_chunks(
    wav: torch.Tensor, sr: int
) -> list[tuple[float, float, torch.Tensor]]:
    """
    Assumes: wav is mono, already at target sample rate.
    Returns: List of (start_sec, end_sec, chunk_tensor).
    """
    chunk_samples = int(Config.CHUNK_SEC * sr)
    overlap_samples = int(Config.OVERLAP_SEC * sr)
    step = chunk_samples - overlap_samples
    total = wav.shape[1]

    chunks: list[tuple[float, float, torch.Tensor]] = []
    i = 0
    while i < total:
        end = min(i + chunk_samples, total)
        chunk = wav[:, i:end]
        if chunk.shape[1] < 1000:
            break
        chunks.append((i / sr, end / sr, chunk))
        i += step

    return chunks


# -----------------------------------------------------------------------------
# Text Processing
# -----------------------------------------------------------------------------


def clean_tts_text(text: str) -> str:
    if text.startswith("\ufeff"):
        text = text[1:]

    lines = [ln for ln in text.split("\n") if not re.search(r"https?://", ln)]
    text = "\n".join(lines)

    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    return text.strip()


def _split_long_token(token: str, max_len: int) -> list[str]:
    """Assumes: len(token) > max_len"""
    return [token[i : i + max_len] for i in range(0, len(token), max_len)]


def split_tts_text(text: str, max_len: int = Config.TTS_MAX_CHARS) -> list[str]:
    """Returns: List of text chunks, each <= max_len."""
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def append_text(text: str) -> None:
        nonlocal current
        if not current:
            current = text
        elif len(current) + 1 + len(text) <= max_len:
            current = current + " " + text
        else:
            flush()
            current = text

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) <= max_len:
            append_text(sent)
            continue

        flush()
        for chunk in _split_long_sentence(sent, max_len):
            if len(chunk) <= max_len:
                append_text(chunk)
            else:
                flush()
                chunks.extend(_split_long_token(chunk, max_len))
    flush()

    return chunks if chunks else [text[:max_len]]


def _split_long_sentence(sent: str, max_len: int) -> list[str]:
    """Assumes: len(sent) > max_len. Split by clauses."""
    clauses = re.split(r"(?<=[,;—])\s+", sent)
    result: list[str] = []
    current = ""

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_len:
            if not current:
                current = clause
            elif len(current) + 1 + len(clause) <= max_len:
                current = current + " " + clause
            else:
                result.append(current)
                current = clause
        else:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_by_words(clause, max_len))
    if current:
        result.append(current)
    return result


def _split_by_words(text: str, max_len: int) -> list[str]:
    """Split long clause by words."""
    words = text.split()
    result: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_len:
            if current:
                result.append(current)
                current = ""
            result.extend(_split_long_token(word, max_len))
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= max_len:
            current = current + " " + word
        else:
            result.append(current)
            current = word
    if current:
        result.append(current)
    return result


# -----------------------------------------------------------------------------
# SSE Streaming
# -----------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Assumes: Event types are mutually exclusive."""

    type: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        payload = {"type": self.type, **self.data}
        return f"data: {json.dumps(payload)}\n\n"


class StreamBridge:
    """
    Bridges sync worker to async SSE stream.
    Caller must ensure: put() called from worker thread, done() called exactly once.
    """

    __slots__ = ("_queue", "_done", "_exception")

    def __init__(self) -> None:
        self._queue: queue.Queue[StreamEvent | None] = queue.Queue(maxsize=16)
        self._done = threading.Event()
        self._exception: Exception | None = None

    def put(self, event: StreamEvent) -> None:
        self._queue.put(event)

    def done(self, exception: Exception | None = None) -> None:
        self._exception = exception
        self._done.set()
        self._queue.put(None)

    async def __aiter__(self) -> AsyncIterator[str]:
        while not self._done.is_set() or not self._queue.empty():
            try:
                event = self._queue.get(timeout=0.05)
                if event is None:
                    break
                yield event.to_sse()
            except queue.Empty:
                await asyncio.sleep(0.01)

        if self._exception:
            yield StreamEvent("error", {"message": str(self._exception)}).to_sse()


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    job_type: str
    state: str
    created_at: float
    updated_at: float
    progress_current: int
    progress_total: int
    error: str | None
    result: dict[str, Any]
    cancelled: bool
    events: list[dict[str, Any]]
    next_seq: int


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(
        self,
        job_type: str,
        session_id: str,
        initial_result: dict[str, Any] | None = None,
    ) -> JobRecord:
        now = time.time()
        rec = JobRecord(
            job_id=secrets.token_urlsafe(24),
            session_id=session_id,
            job_type=job_type,
            state="queued",
            created_at=now,
            updated_at=now,
            progress_current=0,
            progress_total=0,
            error=None,
            result=copy.deepcopy(initial_result or {}),
            cancelled=False,
            events=[],
            next_seq=1,
        )
        with self._lock:
            self._jobs[rec.job_id] = rec
        return copy.deepcopy(rec)

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def belongs_to_session(self, job_id: str, session_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            return bool(rec and rec.session_id == session_id)

    def mark_cancelled(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return False
            rec.cancelled = True
            if rec.state in {"queued", "running"}:
                rec.state = "cancelled"
                rec.updated_at = time.time()
                self._append_event_locked(rec, "cancelled", {})
            return True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            return bool(rec and rec.cancelled)

    def update_event(self, job_id: str, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            now = time.time()
            rec.updated_at = now
            if event_type == "start":
                rec.state = "running"
                rec.progress_total = int(data.get("total", 0))
                rec.progress_current = 0
            elif event_type == "progress":
                rec.state = "running"
                rec.progress_current = int(data.get("current", rec.progress_current))
                rec.progress_total = int(data.get("total", rec.progress_total))
                if rec.job_type == "stt" and "segment" in data:
                    segs = rec.result.setdefault("segments", [])
                    segs.append(data["segment"])
                if rec.job_type == "tts":
                    rec.result["chunks"] = rec.progress_total
            elif event_type == "complete":
                rec.state = "completed"
                if rec.progress_total > 0:
                    rec.progress_current = rec.progress_total
                if rec.job_type == "tts":
                    for key in ("download_url", "duration", "chunks", "filename"):
                        if key in data:
                            rec.result[key] = data[key]
            elif event_type == "error":
                rec.state = "failed"
                rec.error = str(data.get("message", "Job failed"))
            elif event_type == "cancelled":
                rec.state = "cancelled"
            self._append_event_locked(rec, event_type, data)

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return {
                "job_id": rec.job_id,
                "session_id": rec.session_id,
                "job_type": rec.job_type,
                "state": rec.state,
                "progress_current": rec.progress_current,
                "progress_total": rec.progress_total,
                "error": rec.error,
                "result": copy.deepcopy(rec.result),
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
            }

    def events_after(self, job_id: str, after_seq: int) -> list[dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return []
            return [copy.deepcopy(ev) for ev in rec.events if int(ev.get("seq", 0)) > after_seq]

    def list_for_session(
        self, session_id: str, limit: int, offset: int = 0
    ) -> dict[str, Any]:
        with self._lock:
            rows = [
                rec for rec in self._jobs.values()
                if rec.session_id == session_id
            ]
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            total = len(rows)
            start = max(0, int(offset))
            lim = max(1, int(limit))
            slice_rows = rows[start : start + lim]
            out: list[dict[str, Any]] = []
            for rec in slice_rows:
                out.append(
                    {
                        "job_id": rec.job_id,
                        "job_type": rec.job_type,
                        "state": rec.state,
                        "progress_current": rec.progress_current,
                        "progress_total": rec.progress_total,
                        "error": rec.error,
                        "created_at": rec.created_at,
                        "updated_at": rec.updated_at,
                    }
                )
            next_offset = start + len(out)
            has_more = next_offset < total
            return {
                "jobs": out,
                "has_more": has_more,
                "next_offset": next_offset,
            }

    def _append_event_locked(self, rec: JobRecord, event_type: str, data: dict[str, Any]) -> None:
        payload = {"seq": rec.next_seq, "type": event_type, **copy.deepcopy(data)}
        rec.next_seq += 1
        rec.events.append(payload)
        if len(rec.events) > 5000:
            rec.events = rec.events[-5000:]


jobs = JobRegistry()


SESSION_COOKIE = "resonance_session_id"


def get_or_set_session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        return sid
    sid = secrets.token_urlsafe(24)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=60 * 60 * 24 * 30,
    )
    return sid


def _segment_text_from_transcribe(result: Any) -> str:
    """Normalize GigaAM transcribe() output to a plain string for JSON/SSE."""
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    return str(result)


_DEFAULT_STT_TRANSCRIBE_MAX_SEC = 25.0


def _stt_transcribe_hard_limit_sec() -> float:
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


def _stt_single_pass_max_sec() -> float:
    return _stt_transcribe_hard_limit_sec()


def _stt_worker(job_id: str, audio_path: str, upload_root: str) -> None:
    """Worker thread for STT streaming. Removes upload_root when finished."""
    start_time = time.time()
    try:
        model = models.stt()

        wav, sr = load_audio(audio_path)
        total_samples = wav.shape[1]
        duration_sec = total_samples / sr
        max_sec = _stt_single_pass_max_sec()
        use_single_pass = duration_sec <= max_sec and total_samples > 0

        if use_single_pass:
            tmp = tempfile.mkdtemp()
            try:
                path = os.path.join(tmp, "full.wav")
                wavfile.write(
                    path, sr, (wav.squeeze(0).numpy() * 32767).astype("int16")
                )
                jobs.update_event(job_id, "start", {"total": 1})
                raw = model.transcribe(path)
                if jobs.is_cancelled(job_id):
                    jobs.update_event(job_id, "cancelled", {})
                    return
                segment_text = _segment_text_from_transcribe(raw)
                jobs.update_event(
                    job_id,
                    "progress",
                    {
                        "current": 1,
                        "total": 1,
                        "segment": {
                            "start": 0.0,
                            "end": duration_sec,
                            "text": segment_text,
                        },
                    },
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            chunks = split_audio_chunks(wav, sr)
            if not chunks:
                raise ValueError("Audio too short or empty after loading")

            jobs.update_event(job_id, "start", {"total": len(chunks)})

            tmp = tempfile.mkdtemp()
            try:
                for idx, (start, end, chunk) in enumerate(chunks, 1):
                    path = os.path.join(tmp, f"c_{idx}.wav")
                    wavfile.write(
                        path, sr, (chunk.squeeze(0).numpy() * 32767).astype("int16")
                    )
                    raw = model.transcribe(path)
                    if jobs.is_cancelled(job_id):
                        jobs.update_event(job_id, "cancelled", {})
                        return
                    segment_text = _segment_text_from_transcribe(raw)
                    jobs.update_event(
                        job_id,
                        "progress",
                        {
                            "current": idx,
                            "total": len(chunks),
                            "segment": {
                                "start": start,
                                "end": end,
                                "text": segment_text,
                            },
                        },
                    )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        if jobs.is_cancelled(job_id):
            jobs.update_event(job_id, "cancelled", {})
            return
        jobs.update_event(job_id, "complete", {})
        elapsed = time.time() - start_time
        log.info(f"STT completed: {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"STT failed: {e} ({elapsed:.2f}s)")
        jobs.update_event(job_id, "error", {"message": str(e)})
    finally:
        shutil.rmtree(upload_root, ignore_errors=True)


def _tts_worker(job_id: str, text: str, speaker: str, filename: str | None = None) -> None:
    """Worker thread for TTS streaming."""
    start_time = time.time()
    try:
        model = models.tts()

        clean = clean_tts_text(text)
        chunks = split_tts_text(clean)
        total = len(chunks)

        jobs.update_event(job_id, "start", {"total": total})

        audio_parts: list[torch.Tensor] = []
        for idx, chunk_text in enumerate(chunks, 1):
            try:
                audio = model.apply_tts(
                    text=chunk_text, speaker=speaker, sample_rate=Config.TTS_SR
                )
                audio_parts.append(audio)
            except Exception as e:
                log.warning(f"TTS chunk {idx} failed: {e}")
                continue
            if jobs.is_cancelled(job_id):
                jobs.update_event(job_id, "cancelled", {})
                return
            jobs.update_event(job_id, "progress", {"current": idx, "total": total})

        if not audio_parts:
            raise RuntimeError("All TTS chunks failed")

        full_audio = torch.cat(audio_parts, dim=0)
        TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_name = f"{secrets.token_urlsafe(24)}.wav"
        output_path = str(TTS_OUTPUT_DIR / out_name)
        wavfile.write(
            output_path, Config.TTS_SR, (full_audio.numpy() * 32767).astype("int16")
        )

        download_url = f"/api/stream/download?p={out_name}"
        if filename:
            download_url += f"&filename={filename}"

        if jobs.is_cancelled(job_id):
            jobs.update_event(job_id, "cancelled", {})
            return
        jobs.update_event(
            job_id,
            "complete",
            {
                "download_url": download_url,
                "duration": len(full_audio) / Config.TTS_SR,
                "chunks": total,
                "filename": filename,
            },
        )
        elapsed = time.time() - start_time
        log.info(f"TTS completed: {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"TTS failed: {e} ({elapsed:.2f}s)")
        jobs.update_event(job_id, "error", {"message": str(e)})


def _sweep_stale_tts_files(max_age_sec: int) -> None:
    if not TTS_OUTPUT_DIR.is_dir():
        return
    now = time.time()
    for path in TTS_OUTPUT_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        try:
            age = now - path.stat().st_mtime
        except OSError:
            continue
        if age > max_age_sec:
            try:
                path.unlink()
            except OSError:
                pass


async def _tts_file_sweeper() -> None:
    while True:
        await asyncio.sleep(Config.TTS_SWEEP_INTERVAL_SEC)
        await asyncio.to_thread(_sweep_stale_tts_files, Config.TTS_FILE_TTL_SEC)


async def _preload_models() -> None:
    """Warm up models in background; never fail application startup."""
    log.info("Pre-loading models in background...")
    try:
        await asyncio.to_thread(models.stt)
        await asyncio.to_thread(models.tts)
        log.info("Models pre-loaded")
    except asyncio.CancelledError:
        log.info("Model pre-loading cancelled")
        raise
    except Exception as exc:
        # Keep startup quiet and allow lazy model loading on first request.
        log.warning(f"Model pre-loading skipped: {exc}")


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = os.getenv("DEVICE", "cpu")
    log.info(f"Using device: {device}")

    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        f"TTS output dir: {TTS_OUTPUT_DIR} "
        f"(TTL {Config.TTS_FILE_TTL_SEC}s, sweep every {Config.TTS_SWEEP_INTERVAL_SEC}s)"
    )
    sweep_task = asyncio.create_task(_tts_file_sweeper())
    preload_task = asyncio.create_task(_preload_models())

    await asyncio.to_thread(_sweep_stale_tts_files, Config.TTS_FILE_TTL_SEC)
    log.info("Server ready.")
    try:
        yield
    finally:
        preload_task.cancel()
        with suppress(asyncio.CancelledError):
            await preload_task
        sweep_task.cancel()
        with suppress(asyncio.CancelledError):
            await sweep_task
        log.info("Shutting down...")


app = FastAPI(
    title="Resonance API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

VALID_SPEAKERS = frozenset([
    "ru_alexandr", "ru_alfia", "ru_alfia2", "ru_bogdan", "ru_dmitriy",
    "ru_ekaterina", "ru_vika", "ru_gamat", "ru_igor", "ru_karina",
    "ru_kejilgan", "ru_kermen", "ru_marat", "ru_miyau", "ru_nurgul",
    "ru_oksana", "ru_onaoy", "ru_ramilia", "ru_roman", "ru_safarhuja",
    "ru_saida", "ru_sibday", "ru_zara", "ru_zhadyra", "ru_zhazira",
    "ru_zinaida", "ru_eduard",
])


@app.get("/api/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    return {
        "stt": {"name": "GigaAM-v3", "loaded": models.stt_loaded},
        "tts": {"name": "Silero v5_cis_base", "loaded": models.tts_loaded},
        "speakers": list(VALID_SPEAKERS),
    }


@app.post("/api/jobs/stt")
async def start_stt_job(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    content = await file.read()

    if (
        Config.UPLOAD_LIMIT_MB > 0
        and len(content) > Config.UPLOAD_LIMIT_MB * 1024 * 1024
    ):
        raise HTTPException(413, f"File too large (max {Config.UPLOAD_LIMIT_MB}MB)")

    size_kb = len(content) / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
    log.info(f"STT started: {file.filename or 'unknown'} ({size_str})")

    session_id = get_or_set_session_id(request, response)
    rec = jobs.create("stt", session_id, {"filename": file.filename or None})
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "input")
    with open(audio_path, "wb") as f:
        f.write(content)

    asyncio.create_task(
        asyncio.to_thread(_stt_worker, rec.job_id, audio_path, tmp_dir)
    )
    return {"job_id": rec.job_id}


@app.post("/api/jobs/tts")
async def start_tts_job(
    request: Request,
    response: Response,
    text: str = Query(..., min_length=1),
    speaker: str = Query(default=Config.TTS_SPEAKER),
    filename: str | None = Query(default=None),
) -> dict[str, Any]:
    # Validation at boundary (cold path)
    if Config.TTS_MAX_INPUT_CHARS > 0 and len(text) > Config.TTS_MAX_INPUT_CHARS:
        raise HTTPException(
            413, f"Text too long (max {Config.TTS_MAX_INPUT_CHARS} chars)"
        )
    if speaker not in VALID_SPEAKERS:
        raise HTTPException(400, f"Invalid speaker. Use: {list(VALID_SPEAKERS)}")

    log.info(f"TTS started: {len(text)} chars, speaker={speaker}")

    session_id = get_or_set_session_id(request, response)
    rec = jobs.create("tts", session_id, {"filename": filename})
    asyncio.create_task(asyncio.to_thread(_tts_worker, rec.job_id, text, speaker, filename))
    return {"job_id": rec.job_id}


@app.get("/api/jobs")
async def list_jobs(
    request: Request,
    response: Response,
    limit: int = Query(default=60, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    session_id = get_or_set_session_id(request, response)
    return jobs.list_for_session(session_id, limit, offset)


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str, request: Request, response: Response) -> dict[str, Any]:
    session_id = get_or_set_session_id(request, response)
    if not jobs.belongs_to_session(job_id, session_id):
        raise HTTPException(404, "Job not found")
    status = jobs.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(
    job_id: str,
    request: Request,
    response: Response,
    after: int = Query(default=0),
) -> StreamingResponse:
    session_id = get_or_set_session_id(request, response)
    if not jobs.exists(job_id) or not jobs.belongs_to_session(job_id, session_id):
        raise HTTPException(404, "Job not found")

    async def gen() -> AsyncIterator[str]:
        cursor = after
        while True:
            evs = jobs.events_after(job_id, cursor)
            for ev in evs:
                cursor = max(cursor, int(ev.get("seq", cursor)))
                yield f"data: {json.dumps(ev)}\n\n"
            status = jobs.get_status(job_id)
            if not status:
                break
            if status["state"] in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request, response: Response) -> dict[str, Any]:
    session_id = get_or_set_session_id(request, response)
    if not jobs.belongs_to_session(job_id, session_id):
        raise HTTPException(404, "Job not found")
    ok = jobs.mark_cancelled(job_id)
    if not ok:
        raise HTTPException(404, "Job not found")
    return {"ok": True}


@app.get("/api/stream/download")
async def stream_download(p: str, filename: str | None = None) -> FileResponse:
    """Download streamed TTS audio file."""
    file_basename = os.path.basename(p)
    if not file_basename.endswith(".wav") or ".." in file_basename:
        raise HTTPException(400, "Invalid file type")

    root = TTS_OUTPUT_DIR.resolve()
    try:
        candidate = (root / file_basename).resolve()
    except (OSError, RuntimeError):
        raise HTTPException(400, "Invalid path") from None
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(400, "Invalid path") from None
    if not candidate.is_file():
        raise HTTPException(404, "Audio file not found")

    download_name = filename if filename else "resonance_tts.wav"
    if not download_name.endswith(".wav"):
        download_name += ".wav"

    return FileResponse(
        str(candidate), media_type="audio/wav", filename=download_name
    )


# -----------------------------------------------------------------------------
# Public Config Endpoint
# -----------------------------------------------------------------------------


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "upload_limit_mb": Config.UPLOAD_LIMIT_MB,
        "tts_max_chars": Config.TTS_MAX_CHARS,
        "tts_max_input_chars": Config.TTS_MAX_INPUT_CHARS,
        "tts_speakers": list(VALID_SPEAKERS),
        "tts_default_speaker": Config.TTS_SPEAKER,
    }


# -----------------------------------------------------------------------------
# Static Files
# -----------------------------------------------------------------------------

public_dir = Path(__file__).parent / "public"
if public_dir.exists():
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")
else:

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {"message": "Resonance API. Create public/index.html for web UI."}
