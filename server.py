#!/usr/bin/env python3
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
import shutil
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.jobs import JobRegistry, StreamEvent
from core.logging import setup_logging
from stt.models import ModelManager
from stt.pipeline import (
    run_stt_worker,
    save_upload_to_path,
)
from stt.system_audio import get_system_audio_capture
from tts.service import TtsService

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
    ENABLE_SYSTEM_AUDIO: bool = os.getenv(
        "RESONANCE_ENABLE_SYSTEM_AUDIO", "true"
    ).lower() in {"true", "1", "yes"}
    LOG_LEVEL: str = os.getenv(
        "RESONANCE_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")
    ).upper()
    LOG_TO_FILE: bool = os.getenv(
        "RESONANCE_LOG_TO_FILE", "0"
    ).lower() in {"1", "true", "yes"}
    LOG_FILE: str | None = os.getenv("RESONANCE_LOG_FILE")
TTS_OUTPUT_DIR = Path(tempfile.gettempdir()) / "resonance-tts"
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
    log.info("Server ready.")
    try:
        yield
    finally:
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



active_system_captures = {}

def _system_capture_write_loop(audio_engine, tmp_dir: str):
    import os
    import soundfile as sf
    files = {}
    try:
        for stream_id, chunk in audio_engine.get_audio_stream():
            if stream_id not in files:
                path = os.path.join(tmp_dir, f"{stream_id}.wav")
                files[stream_id] = sf.SoundFile(path, mode='w', samplerate=16000, channels=1, subtype='PCM_16')
            files[stream_id].write(chunk)
    except Exception as e:
        pass
    finally:
        for f in files.values():
            f.close()

@app.post("/api/system-audio/start")
async def start_system_audio(
    request: Request,
    response: Response,
    language: str | None = Query(default=None),
    model: str | None = Query(default=None),
    diarization: bool = Query(default=False),
    include_microphone: bool = Query(default=False),
) -> dict[str, Any]:
    if not Config.ENABLE_SYSTEM_AUDIO:
        raise HTTPException(
            status_code=403,
            detail="System audio capture is disabled on this server environment.",
        )

    resolved_language = language or "ru"
    if resolved_language not in {"ru", "en"}:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {resolved_language}")

    if model:
        if model not in {"gigaam", "whisper", "granite"}:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
        if resolved_language == "ru" and model != "gigaam":
            raise HTTPException(status_code=400, detail="Russian language only supports gigaam model")
        if resolved_language == "en" and model not in {"whisper", "granite"}:
            raise HTTPException(status_code=400, detail=f"English language does not support {model} model")
        model_name = model
    else:
        model_name = "gigaam" if resolved_language == "ru" else "whisper"

    capture_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_engine = get_system_audio_capture(include_microphone=include_microphone)
        audio_engine.start_capture()
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Capture failed: {e}")

    task = asyncio.create_task(asyncio.to_thread(_system_capture_write_loop, audio_engine, tmp_dir))

    active_system_captures[capture_id] = {
        "engine": audio_engine,
        "task": task,
        "tmp_dir": tmp_dir,
        "language": resolved_language,
        "model_name": model_name,
        "diarization": diarization,
    }

    log.info(f"System Audio Capture started: {capture_id}")
    return {"capture_id": capture_id}

@app.post("/api/system-audio/stop")
async def stop_system_audio(
    request: Request,
    response: Response,
    capture_id: str = Query(...),
) -> dict[str, Any]:
    if not Config.ENABLE_SYSTEM_AUDIO:
        raise HTTPException(
            status_code=403,
            detail="System audio capture is disabled on this server environment.",
        )

    if capture_id not in active_system_captures:
        raise HTTPException(status_code=404, detail="Capture ID not found")

    capture = active_system_captures.pop(capture_id)
    engine = capture["engine"]

    engine.stop_capture()

    try:
        await asyncio.wait_for(capture["task"], timeout=5.0)
    except Exception as e:
        log.warning(f"Error while stopping capture task: {e}")

    session_id = get_or_set_session_id(request, response)
    initial_result: dict[str, Any] = {"filename": "System Audio Capture.wav"}
    if capture["diarization"]:
        initial_result["diarization"] = True

    rec = jobs.create(
        "stt",
        session_id,
        initial_result,
        language=capture["language"],
        model=capture["model_name"],
    )

    resolved_model = models.get_stt_model(capture["model_name"])

    # Find all generated wav files in tmp_dir
    import glob
    wav_files = glob.glob(os.path.join(capture["tmp_dir"], "*.wav"))
    input_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in wav_files}

    asyncio.create_task(
        asyncio.to_thread(
            run_stt_worker,
            job_id=rec.job_id,
            audio_path=input_paths,
            upload_root=capture["tmp_dir"],
            semaphore=STT_WORKER_SEMAPHORE,
            jobs=jobs,
            model=resolved_model,
            log=log,
            sample_rate=Config.SR,
            chunk_sec=Config.CHUNK_SEC,

            max_duration_sec=0,
            diarization=capture["diarization"],
        )
    )

    log.info(f"System Audio Capture {capture_id} stopped. Spawned job {rec.job_id}.")
    return {"job_id": rec.job_id}


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
    resolved_language = language or "ru"
    if resolved_language not in {"ru", "en"}:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {resolved_language}")

    if model:
        if model not in {"gigaam", "whisper", "granite"}:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
        if resolved_language == "ru" and model != "gigaam":
            raise HTTPException(status_code=400, detail="Russian language only supports gigaam model")
        if resolved_language == "en" and model not in {"whisper", "granite"}:
            raise HTTPException(status_code=400, detail=f"English language does not support {model} model")
        model_name = model
    else:
        if resolved_language == "ru":
            model_name = "gigaam"
        else:
            model_name = "whisper"

    resolved_model = models.get_stt_model(model_name)

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
            audio_path=audio_path,
            upload_root=tmp_dir,
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
        try:
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
        except (asyncio.CancelledError, GeneratorExit):
            return

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
