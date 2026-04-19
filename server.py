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
import sys
import json
import shutil
import queue
import copy
import logging
import secrets
import tempfile
import threading
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Any
from contextlib import asynccontextmanager, suppress

from dotenv import load_dotenv

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from stt.pipeline import (
    run_stt_worker,
    save_upload_to_path,
)
from tts.service import TtsService

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class Config:
    SR: int = int(os.getenv("RESONANCE_SR", "16000"))
    CHUNK_SEC: int = int(os.getenv("RESONANCE_CHUNK_SEC", "20"))
    OVERLAP_SEC: int = int(os.getenv("RESONANCE_OVERLAP_SEC", "2"))
    TTS_SR: int = int(os.getenv("RESONANCE_TTS_SR", "48000"))
    TTS_VOICE_ID: str = os.getenv("RESONANCE_TTS_VOICE_ID", "ru_roman")
    TTS_MAX_CHARS: int = int(os.getenv("RESONANCE_TTS_MAX_CHARS", "600"))
    TTS_MAX_INPUT_CHARS: int = int(os.getenv("RESONANCE_TTS_MAX_INPUT_CHARS", "0"))
    MAX_WORKERS: int = int(os.getenv("RESONANCE_MAX_WORKERS", "2"))
    STT_MAX_CONCURRENT_JOBS: int = int(
        os.getenv("RESONANCE_STT_MAX_CONCURRENT_JOBS", str(MAX_WORKERS))
    )
    STT_MAX_DURATION_SEC: int = int(os.getenv("RESONANCE_STT_MAX_DURATION_SEC", "0"))
    UPLOAD_LIMIT_MB: int = int(os.getenv("RESONANCE_UPLOAD_LIMIT_MB", "0"))
    TTS_FILE_TTL_SEC: int = int(os.getenv("RESONANCE_TTS_FILE_TTL_SEC", "5400"))
    TTS_SWEEP_INTERVAL_SEC: int = int(
        os.getenv("RESONANCE_TTS_SWEEP_INTERVAL_SEC", "900")
    )
TTS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "resonance-tts"
STT_WORKER_SEMAPHORE = threading.BoundedSemaphore(
    max(1, Config.STT_MAX_CONCURRENT_JOBS)
)


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
tts_service = TtsService(
    config=Config,
    get_model=models.tts,
    is_model_loaded=lambda: models.tts_loaded,
    log=log,
    output_dir=TTS_OUTPUT_DIR,
)


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
                if rec.job_type == "stt" and "duration" in data:
                    try:
                        rec.result["duration"] = float(data["duration"])
                    except Exception:
                        pass
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
                if rec.job_type == "stt":
                    # Keep duration available in job summaries without requiring /api/jobs/{id}.
                    if "duration" in data:
                        try:
                            rec.result["duration"] = float(data["duration"])
                        except Exception:
                            pass
                    elif "duration" not in rec.result and isinstance(rec.result.get("segments"), list):
                        max_end = 0.0
                        for s in rec.result.get("segments") or []:
                            try:
                                end = float(s.get("end", 0.0)) if isinstance(s, dict) else 0.0
                            except Exception:
                                end = 0.0
                            if end > max_end:
                                max_end = end
                        if max_end > 0:
                            rec.result["duration"] = max_end
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
                duration_out: float | None = None
                duration = rec.result.get("duration")
                if duration is not None:
                    try:
                        duration_out = float(duration)
                    except Exception:
                        duration_out = None
                if duration_out is None and rec.job_type == "stt" and isinstance(rec.result.get("segments"), list):
                    max_end = 0.0
                    for s in rec.result.get("segments") or []:
                        try:
                            end = float(s.get("end", 0.0)) if isinstance(s, dict) else 0.0
                        except Exception:
                            end = 0.0
                        if end > max_end:
                            max_end = end
                    if max_end > 0:
                        duration_out = max_end
                out.append(
                    {
                        "job_id": rec.job_id,
                        "job_type": rec.job_type,
                        "state": rec.state,
                        "progress_current": rec.progress_current,
                        "progress_total": rec.progress_total,
                        "error": rec.error,
                        "duration": duration_out,
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


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = os.getenv("DEVICE", "cpu")
    log.info(f"Using device: {device}")

    tts_service.output_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        f"TTS output dir: {tts_service.output_dir} "
        f"(TTL {Config.TTS_FILE_TTL_SEC}s, sweep every {Config.TTS_SWEEP_INTERVAL_SEC}s)"
    )
    sweep_task = asyncio.create_task(
        tts_service.run_file_sweeper(
            Config.TTS_SWEEP_INTERVAL_SEC,
            Config.TTS_FILE_TTL_SEC,
        )
    )

    await asyncio.to_thread(tts_service.sweep_stale_files, Config.TTS_FILE_TTL_SEC)
    log.info("Server ready.")
    try:
        yield
    finally:
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

@app.get("/api/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("OK")


@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    primary_backend = tts_service.backends["silero_ru"]
    return {
        "stt": {"name": "GigaAM-v3", "loaded": models.stt_loaded},
        "tts": {"name": primary_backend.name, "loaded": primary_backend.loaded},
        "tts_catalog": tts_service.serialize_catalog(),
        "tts_backends": [
            {
                "id": backend_id,
                "name": backend.name,
                "loaded": backend.loaded,
            }
            for backend_id, backend in tts_service.backends.items()
        ],
    }


@app.post("/api/jobs/stt")
async def start_stt_job(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "input")
    try:
        size_bytes = await save_upload_to_path(
            file,
            audio_path,
            max_bytes=Config.UPLOAD_LIMIT_MB * 1024 * 1024,
        )
    except ValueError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(413, str(exc)) from exc
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    size_kb = size_bytes / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
    log.info(f"STT started: {file.filename or 'unknown'} ({size_str})")

    session_id = get_or_set_session_id(request, response)
    rec = jobs.create("stt", session_id, {"filename": file.filename or None})

    asyncio.create_task(
        asyncio.to_thread(
            run_stt_worker,
            job_id=rec.job_id,
            audio_path=audio_path,
            upload_root=tmp_dir,
            semaphore=STT_WORKER_SEMAPHORE,
            jobs=jobs,
            model=models.stt(),
            log=log,
            sample_rate=Config.SR,
            chunk_sec=Config.CHUNK_SEC,
            overlap_sec=Config.OVERLAP_SEC,
            max_duration_sec=Config.STT_MAX_DURATION_SEC,
        )
    )
    return {"job_id": rec.job_id}


@app.post("/api/jobs/tts")
async def start_tts_job(
    request: Request,
    response: Response,
    text: str = Query(..., min_length=1),
    language: str | None = Query(default=None),
    voice_id: str | None = Query(default=None),
    filename: str | None = Query(default=None),
) -> dict[str, Any]:
    # Validation at boundary (cold path)
    if Config.TTS_MAX_INPUT_CHARS > 0 and len(text) > Config.TTS_MAX_INPUT_CHARS:
        raise HTTPException(
            413, f"Text too long (max {Config.TTS_MAX_INPUT_CHARS} chars)"
        )
    resolved_voice_id = voice_id or tts_service.default_voice_id()
    resolved_language = language or tts_service.get_voice_or_400(resolved_voice_id).language
    tts_service.validate_language_voice(resolved_language, resolved_voice_id)

    log.info(
        f"TTS started: {len(text)} chars, language={resolved_language}, voice_id={resolved_voice_id}"
    )

    session_id = get_or_set_session_id(request, response)
    rec = jobs.create("tts", session_id, {"filename": filename})
    asyncio.create_task(
        asyncio.to_thread(
            tts_service.run_job,
            job_id=rec.job_id,
            text=text,
            voice_id=resolved_voice_id,
            jobs=jobs,
            filename=filename,
        )
    )
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
    status = jobs.get_status(job_id)
    ok = jobs.mark_cancelled(job_id)
    if not ok:
        raise HTTPException(404, "Job not found")
    job_type = str((status or {}).get("job_type", "job")).upper()
    log.info(f"{job_type} cancel requested: job_id={job_id}")
    return {"ok": True}


@app.get("/api/stream/download")
async def stream_download(p: str, filename: str | None = None) -> FileResponse:
    """Download streamed TTS audio file."""
    file_basename = os.path.basename(p)
    if not file_basename.endswith(".wav") or ".." in file_basename:
        raise HTTPException(400, "Invalid file type")

    root = tts_service.output_dir.resolve()
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
        "tts": tts_service.serialize_catalog(),
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
