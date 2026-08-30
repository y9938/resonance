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

import asyncio
import copy
import json
import logging
import os
import queue
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Any

from dotenv import load_dotenv

load_dotenv()

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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


from core.logging import setup_logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

setup_logging()
log = logging.getLogger("resonance.server")


# -----------------------------------------------------------------------------
# Model Manager
# -----------------------------------------------------------------------------


class ModelManager:
    """Lazy model loader. Assumes: single-threaded init, thread-safe after."""

    def __init__(self) -> None:
        self._stt_gigaam: Any | None = None
        self._stt_whisper: Any | None = None
        self._stt_granite: Any | None = None
        self._tts: Any | None = None
        self._lock = threading.Lock()

    @property
    def stt_gigaam_loaded(self) -> bool:
        return self._stt_gigaam is not None

    @property
    def stt_whisper_loaded(self) -> bool:
        return self._stt_whisper is not None

    @property
    def stt_granite_loaded(self) -> bool:
        return self._stt_granite is not None

    @property
    def tts_loaded(self) -> bool:
        return self._tts is not None

    def stt_gigaam(self) -> Any:
        with self._lock:
            if self._stt_gigaam is None:
                self._stt_gigaam = _load_stt()
            return self._stt_gigaam

    def stt_whisper(self) -> Any:
        with self._lock:
            if self._stt_whisper is None:
                self._stt_whisper = _load_whisper()
            return self._stt_whisper

    def stt_granite(self) -> Any:
        with self._lock:
            if self._stt_granite is None:
                self._stt_granite = _load_granite()
            return self._stt_granite

    def tts(self) -> Any:
        with self._lock:
            if self._tts is None:
                self._tts = _load_tts()
            return self._tts


class GigaAMAdapter:
    """Wraps GigaAM for True Zero-I/O in-RAM tensor inference without disk operations."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def transcribe(self, audio: Any) -> str:
        import torch
        import numpy as np

        # Domain Invariant: True Zero-I/O RAM pipeline bypassing file creation and wrapper threshold
        if isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio).to(self._model._device).to(self._model._dtype)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            length = torch.tensor([wav.shape[-1]], device=self._model._device)
        else:
            wav, length = self._model.prepare_wav(str(audio))

        with torch.inference_mode():
            encoded, encoded_len = self._model.forward(wav, length)
            text, _ = self._model._decode(encoded, encoded_len, length, False)[0]
        return text


def _load_stt() -> GigaAMAdapter:
    log.info("Loading STT model (GigaAM-v3)...")
    import gigaam

    device = os.getenv("DEVICE", "cpu")
    model = gigaam.load_model("v3_e2e_ctc", device=device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"STT model loaded: {params:.1f}M parameters")
    return GigaAMAdapter(model)


class WhisperAdapter:
    """Wraps WhisperModel to match GigaAM's transcribe(path) -> str interface."""

    def __init__(self, model: Any, beam_size: int = 5) -> None:
        self._model = model
        self._beam_size = beam_size

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, beam_size=self._beam_size)
        return " ".join(seg.text.strip() for seg in segments).strip()


def _load_whisper() -> WhisperAdapter:
    log.info("Loading STT model (Distil-Whisper-v3)...")
    from faster_whisper import WhisperModel

    device = os.getenv("DEVICE", "cpu")
    if device.startswith("cuda"):
        ct2_device = "cuda"
        device_index = 0
        if ":" in device:
            try:
                device_index = int(device.split(":")[1])
            except ValueError:
                pass
        compute_type = "float16"
    else:
        ct2_device = "cpu"
        device_index = 0
        compute_type = "int8"

    kwargs = {
        "device": ct2_device,
        "compute_type": compute_type,
    }
    if device_index > 0:
        kwargs["device_index"] = device_index

    model = WhisperModel(
        "Systran/faster-distil-whisper-large-v3",
        **kwargs
    )
    log.info(f"Whisper model loaded: device={ct2_device}, device_index={device_index}, compute_type={compute_type}")
    return WhisperAdapter(model)


class GraniteAdapter:
    """Wraps ibm-granite/granite-speech-4.1-2b-plus model for inference."""

    def __init__(self, model: Any, processor: Any, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def transcribe(self, audio: np.ndarray, diarization: bool = False) -> str:
        import torch

        waveform = torch.from_numpy(audio)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        audio_input = waveform.squeeze().numpy()

        if diarization:
            prompt = "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
        else:
            prompt = "<|audio|> can you transcribe the speech into a written format?"

        chat = [{"role": "user", "content": prompt}]
        prompt_text = self._processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = self._processor(
            text=prompt_text,
            audio=audio_input,
            sampling_rate=16000,
            return_tensors="pt"
        )
        inputs = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self._model.dtype)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2000
            )

        if "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_len:]
        else:
            new_tokens = generated_ids[0]

        transcription = self._processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return transcription.strip()


def _load_granite() -> GraniteAdapter:
    log.info("Loading STT model (IBM Granite Speech 4.1 Plus)...")
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch

    device = os.getenv("DEVICE", "cpu")
    model_id = "ibm-granite/granite-speech-4.1-2b-plus"

    processor = AutoProcessor.from_pretrained(model_id)

    if device.startswith("cuda"):
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch_dtype,
    ).to(device)

    log.info(f"Granite model loaded: device={device}, dtype={torch_dtype}")
    return GraniteAdapter(model, processor, device)


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
    language: str | None = None
    model: str | None = None


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(
        self,
        job_type: str,
        session_id: str,
        initial_result: dict[str, Any] | None = None,
        language: str | None = None,
        model: str | None = None,
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
            language=language,
            model=model,
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

    def cancel_all(self) -> None:
        with self._lock:
            for rec in self._jobs.values():
                rec.cancelled = True
                if rec.state in {"queued", "running"}:
                    rec.state = "cancelled"
                    rec.updated_at = time.time()
                    self._append_event_locked(rec, "cancelled", {})

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
                "language": rec.language,
                "model": rec.model,
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
                item = {
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
                if rec.language is not None:
                    item["language"] = rec.language
                if rec.model is not None:
                    item["model"] = rec.model
                for key in ("filename", "batch_id", "batch_index", "batch_total"):
                    if key in rec.result:
                        item[key] = copy.deepcopy(rec.result[key])
                out.append(item)
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

    if capture["model_name"] == "granite":
        resolved_model = models.stt_granite()
    elif capture["model_name"] == "whisper":
        resolved_model = models.stt_whisper()
    else:
        resolved_model = models.stt_gigaam()

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

    if model_name == "granite":
        resolved_model = models.stt_granite()
    elif model_name == "whisper":
        resolved_model = models.stt_whisper()
    else:
        resolved_model = models.stt_gigaam()

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
