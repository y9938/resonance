"""
Resonance API Server - Unified STT/TTS Service

Architecture:
- FastAPI for HTTP API
- asyncio.to_thread for blocking model inference (STT/TTS)
- Server-Sent Events (SSE) streaming for real-time progress
- Graceful shutdown with resource cleanup
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from core.jobs import JobRegistry
from core.logging import setup_logging
from stt.buffer import decode_media_bytes
from stt.live import LiveSTTSession
from stt.live_preview import LivePreviewBroker
from stt.live_transport import (
    LiveSequenceAhead,
    LiveSessionFinishing,
    LiveSessionHandle,
)
from stt.models import ModelManager, resolve_stt_model
from stt.pipeline import run_stt_worker
from stt.system_audio import get_system_audio_capture
from tts.service import TtsService

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class Config:
    SR: int = int(os.getenv("RESONANCE_SR", "16000"))
    CHUNK_SEC: int = int(os.getenv("RESONANCE_CHUNK_SEC", "20"))
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
    ENABLE_SYSTEM_AUDIO: bool = os.getenv(
        "RESONANCE_ENABLE_SYSTEM_AUDIO", "true"
    ).lower() in {"true", "1", "yes"}
    SYSTEM_CAPTURE_STOP_TIMEOUT_SEC: float = float(
        os.getenv("RESONANCE_SYSTEM_CAPTURE_STOP_TIMEOUT_SEC", "5")
    )
    LIVE_STT_IDLE_TIMEOUT_SEC: float = float(
        os.getenv("RESONANCE_LIVE_STT_IDLE_TIMEOUT_SEC", "15")
    )
    LOG_LEVEL: str = os.getenv(
        "RESONANCE_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")
    ).upper()
    LOG_TO_FILE: bool = os.getenv(
        "RESONANCE_LOG_TO_FILE", "0"
    ).lower() in {"1", "true", "yes"}
    LOG_FILE: str | None = os.getenv("RESONANCE_LOG_FILE")
TTS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "resonance-tts"
if Config.LIVE_STT_IDLE_TIMEOUT_SEC < 0:
    raise ValueError("RESONANCE_LIVE_STT_IDLE_TIMEOUT_SEC must be non-negative")
STT_WORKER_SEMAPHORE = threading.BoundedSemaphore(
    max(1, Config.STT_MAX_CONCURRENT_JOBS)
)


def cors_allow_origins() -> list[str]:
    default_origin = f"http://localhost:{os.getenv('RESONANCE_PORT', '8000')}"
    raw = os.getenv("RESONANCE_CORS_ORIGINS", default_origin).strip()
    if not raw:
        return [default_origin]
    if raw == "*":
        return ["*"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins if origins else [default_origin]


setup_logging()
log = logging.getLogger("resonance.server")

models = ModelManager()
tts_service = TtsService(
    config=Config,
    get_model=models.tts,
    is_model_loaded=lambda: models.tts_loaded,
    log=log,
    output_dir=TTS_OUTPUT_DIR,
)

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

    from core.ipc import create_local_ipc_server
    ipc_server = create_local_ipc_server()
    try:
        await ipc_server.start()
    except Exception as exc:
        log.warning(f"Failed to start local IPC server: {exc}")
        ipc_server = None

    log.info("Server ready.")
    try:
        yield
    finally:
        if ipc_server is not None:
            try:
                await ipc_server.stop()
            except Exception as exc:
                log.debug(f"Error stopping IPC server: {exc}")

        with system_capture_lock:
            for cap in active_system_captures.values():
                try:
                    cap["engine"].stop_capture()
                except Exception as e:
                    log.debug(f"Error stopping capture on shutdown: {e}")
            active_system_captures.clear()

        jobs.cancel_all()
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
        "stt": {
            "gigaam": {"name": "GigaAM-v3", "loaded": models.stt_gigaam_loaded},
            "whisper": {"name": "Distil-Whisper-v3", "loaded": models.stt_whisper_loaded},
            "granite": {"name": "IBM Granite Speech 4.1 Plus", "loaded": models.stt_granite_loaded},
            "languages": {"ru": "gigaam", "en": "whisper"},
        },
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



active_system_captures: dict[str, Any] = {}
active_live_sessions: dict[str, LiveSessionHandle] = {}
live_preview_broker = LivePreviewBroker()
system_capture_lock = threading.Lock()


async def _expire_live_session(
    job_id: str,
    handle: LiveSessionHandle,
    session: LiveSTTSession,
) -> None:
    try:
        await handle.wait_for_drain()
        await asyncio.to_thread(session.flush, "mic")
        jobs.update_event(
            job_id,
            "complete",
            {
                "duration": session.duration_sec,
                "completion_reason": "idle_timeout",
                "partial": True,
            },
        )
        log.info(f"Live STT job expired: job_id={job_id}")
    except Exception as exc:
        session.cancel()
        jobs.update_event(job_id, "error", {"message": f"Live STT timeout cleanup failed: {exc}"})
        log.exception(f"Live STT timeout cleanup failed: job_id={job_id}")
    finally:
        live_preview_broker.discard(job_id)
        if active_live_sessions.get(job_id) is handle:
            active_live_sessions.pop(job_id, None)


def _system_capture_live_loop(audio_engine, live_session: LiveSTTSession) -> Exception:
    try:
        for stream_id, chunk in audio_engine.get_audio_stream():
            live_session.process_pcm_chunk(chunk, source=stream_id)
    except Exception as e:
        return e
    return RuntimeError("System audio capture producer exited unexpectedly")


async def _supervise_system_capture(
    job_id: str,
    audio_engine: Any,
    live_session: LiveSTTSession,
) -> None:
    producer_error = await asyncio.to_thread(
        _system_capture_live_loop, audio_engine, live_session
    )
    with system_capture_lock:
        capture = active_system_captures.get(job_id)
        if capture is None or capture["live_session"] is not live_session:
            return
        if capture["stop_requested"]:
            return
        active_system_captures.pop(job_id, None)

    live_session.cancel()
    jobs.update_event(
        job_id,
        "error",
        {"message": f"System audio capture stopped unexpectedly: {producer_error}"},
    )
    log.error(f"System audio capture failed: job_id={job_id}; error={producer_error}")


def _fail_system_capture(job_id: str, capture: dict[str, Any], message: str) -> None:
    capture["live_session"].cancel()
    with system_capture_lock:
        if active_system_captures.get(job_id) is capture:
            active_system_captures.pop(job_id, None)
    jobs.update_event(job_id, "error", {"message": message})
    log.error(f"System audio capture failed: job_id={job_id}; error={message}")


@app.post("/api/system-audio/start")
async def start_system_audio(
    request: Request,
    response: Response,
    language: str | None = Query(default=None),
    model: str | None = Query(default=None),
    include_microphone: bool = Query(default=False),
) -> dict[str, Any]:
    if not Config.ENABLE_SYSTEM_AUDIO:
        raise HTTPException(
            status_code=403,
            detail="System audio capture is disabled on this server environment.",
        )

    try:
        resolved_language, model_name = resolve_stt_model(language, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = get_or_set_session_id(request, response)
    capture_config = (resolved_language, model_name, include_microphone)

    with system_capture_lock:
        if active_system_captures:
            active_job_id = next(iter(active_system_captures))
            active_capture = active_system_captures[active_job_id]
            if active_capture["config"] != capture_config:
                raise HTTPException(
                    status_code=409,
                    detail="System audio capture is already active with a different configuration",
                )
            existing_status = jobs.get_status(active_job_id)
            log.info(f"System Audio Capture already active: returning existing job_id={active_job_id}")
            return {
                "job_id": active_job_id,
                "resumed": True,
                "started_at": existing_status["started_at"] if existing_status else None,
            }

        try:
            resolved_model = models.get_stt_model(model_name)
            audio_engine = get_system_audio_capture(include_microphone=include_microphone)
        except Exception as exc:
            log.error(f"System Audio Capture setup failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Capture setup failed: {exc}") from exc

        rec = jobs.create(
            "stt",
            session_id,
            {"filename": "System Audio Capture.wav", "source": "system_audio"},
            language=resolved_language,
            model=model_name,
        )
        job_id = rec.job_id

        capture_started = False
        try:
            live_session = LiveSTTSession(
                job_id=job_id,
                session_id=session_id,
                model=resolved_model,
                jobs=jobs,
                sample_rate=Config.SR,
                source="sys",
                dual_stream=include_microphone,
                preview_broker=live_preview_broker,
            )
            audio_engine.start_capture()
            capture_started = True
            capture = {
                "engine": audio_engine,
                "live_session": live_session,
                "session_id": session_id,
                "config": capture_config,
                "stop_requested": False,
            }
            active_system_captures[job_id] = capture
            supervisor = _supervise_system_capture(job_id, audio_engine, live_session)
            try:
                capture["task"] = asyncio.create_task(supervisor)
            except Exception:
                supervisor.close()
                raise
            jobs.update_event(job_id, "start", {"stage": "capturing", "total": 0})
        except Exception as exc:
            active_system_captures.pop(job_id, None)
            if capture_started:
                try:
                    audio_engine.stop_capture()
                except Exception as stop_exc:
                    log.warning(f"Failed to stop rolled-back capture: {stop_exc}")
            jobs.update_event(job_id, "error", {"message": f"Capture start failed: {exc}"})
            raise HTTPException(status_code=500, detail=f"Capture start failed: {exc}") from exc

    log.info(f"System Audio Live Capture started: job_id={job_id}")
    status = jobs.get_status(job_id)
    return {"job_id": job_id, "started_at": status["started_at"] if status else None}


@app.post("/api/system-audio/stop")
async def stop_system_audio(
    request: Request,
    response: Response,
    job_id: str = Query(...),
) -> dict[str, Any]:
    if not Config.ENABLE_SYSTEM_AUDIO:
        raise HTTPException(
            status_code=403,
            detail="System audio capture is disabled on this server environment.",
        )

    with system_capture_lock:
        if job_id not in active_system_captures:
            raise HTTPException(status_code=404, detail="Job ID not found or already stopped")
        capture = active_system_captures[job_id]
        capture["stop_requested"] = True

    engine = capture["engine"]
    try:
        engine.stop_capture()
    except Exception as exc:
        _fail_system_capture(job_id, capture, f"System audio stop failed: {exc}")
        raise HTTPException(status_code=500, detail="System audio stop failed") from exc

    try:
        await asyncio.wait_for(
            asyncio.shield(capture["task"]),
            timeout=Config.SYSTEM_CAPTURE_STOP_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError as exc:
        _fail_system_capture(
            job_id,
            capture,
            "System audio capture did not stop before the shutdown timeout",
        )
        raise HTTPException(status_code=504, detail="System audio capture stop timed out") from exc
    except Exception as exc:
        _fail_system_capture(job_id, capture, f"System audio capture stop failed: {exc}")
        raise HTTPException(status_code=500, detail="System audio capture stop failed") from exc

    live_session = capture["live_session"]
    try:
        await asyncio.to_thread(live_session.flush)
    except Exception as exc:
        _fail_system_capture(job_id, capture, f"System audio flush failed: {exc}")
        raise HTTPException(status_code=500, detail="System audio flush failed") from exc

    with system_capture_lock:
        if active_system_captures.get(job_id) is capture:
            active_system_captures.pop(job_id, None)
    jobs.update_event(job_id, "complete", {"duration": live_session.duration_sec})
    live_preview_broker.discard(job_id)
    log.debug("System live STT metrics: job_id=%s metrics=%s", job_id, live_session.inference_metrics)
    log.info(f"System Audio Live Capture stopped: job_id={job_id}")
    return {"job_id": job_id}


@app.post("/api/jobs/live/start")
async def start_live_job(
    request: Request,
    response: Response,
    language: str | None = Query(default=None),
    model: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        resolved_language, model_name = resolve_stt_model(language, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        resolved_model = models.get_stt_model(model_name)
    except Exception as exc:
        log.error(f"Live STT model setup failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Live STT setup failed: {exc}") from exc

    session_id = get_or_set_session_id(request, response)
    rec = jobs.create(
        "stt",
        session_id,
        {"filename": "Live Recording.wav", "source": "mic_live"},
        language=resolved_language,
        model=model_name,
    )
    job_id = rec.job_id
    try:
        live_session = LiveSTTSession(
            job_id=job_id,
            session_id=session_id,
            model=resolved_model,
            jobs=jobs,
            sample_rate=Config.SR,
            source="mic",
            preview_broker=live_preview_broker,
        )
        handle = LiveSessionHandle(live_session, Config.LIVE_STT_IDLE_TIMEOUT_SEC)
        active_live_sessions[job_id] = handle
        jobs.update_event(job_id, "start", {"stage": "streaming", "total": 0})
        handle.start_expiry_task(
            lambda session: _expire_live_session(job_id, handle, session)
        )
    except Exception as exc:
        active_live_sessions.pop(job_id, None)
        jobs.update_event(job_id, "error", {"message": f"Live STT start failed: {exc}"})
        raise HTTPException(status_code=500, detail=f"Live STT start failed: {exc}") from exc

    log.info(f"Live STT job started: job_id={job_id}")
    status = jobs.get_status(job_id)
    return {"job_id": job_id, "started_at": status["started_at"] if status else None}


@app.post("/api/jobs/live/{job_id}/chunk", response_model=None)
async def append_live_chunk(
    job_id: str,
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    sequence: int = Query(..., ge=1),
) -> Any:
    session_id = get_or_set_session_id(request, response)
    if not jobs.belongs_to_session(job_id, session_id):
        raise HTTPException(status_code=404, detail="Live job not found or completed")
    handle = active_live_sessions.get(job_id)
    if not handle:
        raise HTTPException(status_code=404, detail="Live job not found or completed")

    try:
        async with handle.claim_chunk() as session:
            file_bytes = await file.read()
            if not file_bytes:
                raise HTTPException(status_code=400, detail="Live audio chunk is empty")
            async with handle.serialise_sequence(sequence) as duplicate_ack:
                if duplicate_ack is not None:
                    await handle.touch_activity()
                    return {"emitted": 0, "ack_sequence": duplicate_ack}

                await handle.touch_activity()
                try:
                    def decode_chunk() -> Any:
                        audio_buffer = decode_media_bytes(file_bytes, target_sample_rate=Config.SR)
                        return audio_buffer.as_ndarray()

                    pcm_chunk = await asyncio.to_thread(decode_chunk)
                except Exception as exc:
                    log.warning(f"Invalid live audio chunk: {exc}")
                    raise HTTPException(status_code=400, detail="Invalid live audio chunk") from exc

                emitted = await asyncio.to_thread(session.process_pcm_chunk, pcm_chunk, "mic")
                handle.commit_sequence(sequence)
                return {"emitted": len(emitted), "ack_sequence": sequence}
    except HTTPException:
        raise
    except LiveSessionFinishing as exc:
        raise HTTPException(status_code=404, detail="Live job not found or completed") from exc
    except LiveSequenceAhead as exc:
        return JSONResponse(
            status_code=409,
            content={"expected_sequence": exc.expected_sequence},
        )
    except Exception as exc:
        finish_session = await handle.claim_finish()
        if finish_session is not None:
            finish_session.cancel()
            live_preview_broker.discard(job_id)
            handle.cancel_expiry_task()
            if active_live_sessions.get(job_id) is handle:
                active_live_sessions.pop(job_id, None)
            jobs.update_event(job_id, "error", {"message": f"Live STT processing failed: {exc}"})
        log.exception(f"Live STT processing failed: job_id={job_id}")
        raise HTTPException(status_code=500, detail="Live STT processing failed") from exc


@app.post("/api/jobs/live/{job_id}/stop")
async def stop_live_job(
    job_id: str,
    request: Request,
    response: Response,
) -> dict[str, Any]:
    session_id = get_or_set_session_id(request, response)
    if not jobs.belongs_to_session(job_id, session_id):
        raise HTTPException(status_code=404, detail="Live job not found or already stopped")
    handle = active_live_sessions.get(job_id)
    if not handle:
        raise HTTPException(status_code=404, detail="Live job not found or already stopped")
    session = await handle.claim_finish()
    if not session:
        raise HTTPException(status_code=404, detail="Live job not found or already stopped")

    try:
        handle.cancel_expiry_task()
        await handle.wait_for_drain()
        await asyncio.to_thread(session.flush, "mic")
        jobs.update_event(
            job_id,
            "complete",
            {
                "duration": session.duration_sec,
                "completion_reason": "user_stop",
                "partial": False,
            },
        )
    except Exception as exc:
        session.cancel()
        jobs.update_event(job_id, "error", {"message": f"Live STT stop failed: {exc}"})
        log.exception(f"Live STT stop failed: job_id={job_id}")
        raise HTTPException(status_code=500, detail="Live STT stop failed") from exc
    finally:
        live_preview_broker.discard(job_id)
        if active_live_sessions.get(job_id) is handle:
            active_live_sessions.pop(job_id, None)
    log.info(f"Live STT job stopped: job_id={job_id}")
    log.debug("Browser live STT metrics: job_id=%s metrics=%s", job_id, session.inference_metrics)
    return {"job_id": job_id}


@app.post("/api/jobs/stt")
async def start_stt_job(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    language: str | None = Query(default=None),
    model: str | None = Query(default=None),
    diarization: bool = Query(default=False),
    batch_id: str | None = Query(default=None),
    batch_index: int | None = Query(default=None, ge=1),
    batch_total: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    try:
        resolved_language, model_name = resolve_stt_model(language, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_model = models.get_stt_model(model_name)
    max_bytes = Config.UPLOAD_LIMIT_MB * 1024 * 1024
    file_bytes = await file.read()
    if max_bytes > 0 and len(file_bytes) > max_bytes:
        raise HTTPException(413, f"File too large (max {Config.UPLOAD_LIMIT_MB}MB)")

    try:
        audio_buffer = decode_media_bytes(file_bytes, target_sample_rate=Config.SR)
    except Exception as exc:
        raise HTTPException(400, f"Failed to decode audio file: {exc}") from exc

    size_kb = len(file_bytes) / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
    log.info(f"STT started: {file.filename or 'unknown'} ({size_str})")

    session_id = get_or_set_session_id(request, response)
    initial_result: dict[str, Any] = {"filename": file.filename or None}
    if diarization:
        initial_result["diarization"] = True
    if batch_id:
        initial_result["batch_id"] = batch_id
        if batch_index is not None:
            initial_result["batch_index"] = batch_index
        if batch_total is not None:
            initial_result["batch_total"] = batch_total
    rec = jobs.create(
        "stt",
        session_id,
        initial_result,
        language=resolved_language,
        model=model_name,
    )

    asyncio.create_task(
        asyncio.to_thread(
            run_stt_worker,
            job_id=rec.job_id,
            audio_path=audio_buffer,
            semaphore=STT_WORKER_SEMAPHORE,
            jobs=jobs,
            model=resolved_model,
            log=log,
            sample_rate=Config.SR,
            chunk_sec=Config.CHUNK_SEC,
            max_duration_sec=Config.STT_MAX_DURATION_SEC,
            diarization=diarization,
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
        preview_subscription = live_preview_broker.subscribe(job_id)
        try:
            while True:
                evs = jobs.events_after(job_id, cursor)
                for ev in evs:
                    cursor = max(cursor, int(ev.get("seq", cursor)))
                    yield f"data: {json.dumps(ev)}\n\n"
                # Preview events are intentionally outside JobRegistry: no sequence,
                # no persistence and no replay after a reconnect.
                for preview in live_preview_broker.take(preview_subscription):
                    yield f"data: {json.dumps(preview)}\n\n"
                status = jobs.get_status(job_id)
                if not status:
                    break
                if status["state"] in {"completed", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.2)
        except (asyncio.CancelledError, GeneratorExit):
            return
        finally:
            live_preview_broker.unsubscribe(preview_subscription)

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
    live_handle = active_live_sessions.get(job_id)
    live_session = await live_handle.claim_finish() if live_handle else None
    if live_handle is not None and live_session is None:
        raise HTTPException(404, "Job not found")

    ok = jobs.mark_cancelled(job_id)
    if not ok:
        if live_session is not None:
            live_session.cancel()
            if active_live_sessions.get(job_id) is live_handle:
                active_live_sessions.pop(job_id, None)
        raise HTTPException(404, "Job not found")
    job_type = str((status or {}).get("job_type", "job")).upper()
    log.info(f"{job_type} cancel requested: job_id={job_id}")

    if live_session is not None:
        live_session.cancel()
        live_preview_broker.discard(job_id)
        live_handle.cancel_expiry_task()
        if active_live_sessions.get(job_id) is live_handle:
            active_live_sessions.pop(job_id, None)
        return {"ok": True}

    with system_capture_lock:
        if job_id in active_system_captures:
            capture = active_system_captures.pop(job_id)
            try:
                capture["live_session"].cancel()
                live_preview_broker.discard(job_id)
                capture["engine"].stop_capture()
                capture["task"].cancel()
            except Exception as e:
                log.debug(f"Error terminating cancelled capture engine: {e}")

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
# Active Session Context Tail Endpoint
# -----------------------------------------------------------------------------


@app.get("/api/context/tail")
async def get_context_tail(
    request: Request,
    response: Response,
    lines: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    session_id = get_or_set_session_id(request, response)
    from core.context import session_context_manager

    tail_lines = session_context_manager.get_tail(session_id=session_id, lines=lines)
    return {
        "lines": tail_lines,
        "combined": " ".join(tail_lines),
        "count": len(tail_lines),
    }


# -----------------------------------------------------------------------------
# Public Config Endpoint
# -----------------------------------------------------------------------------


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return {
        "upload_limit_mb": Config.UPLOAD_LIMIT_MB,
        "tts_max_chars": Config.TTS_MAX_CHARS,
        "tts_max_input_chars": Config.TTS_MAX_INPUT_CHARS,
        "system_audio_enabled": Config.ENABLE_SYSTEM_AUDIO,
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
