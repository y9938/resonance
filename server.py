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
import logging
import tempfile
import subprocess
import threading
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv

import torch
import torchaudio
import torchaudio.transforms as T
from scipy.io import wavfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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


def _stt_worker(bridge: StreamBridge, audio_path: str) -> None:
    """Worker thread for STT streaming."""
    tmp_dir = tempfile.mkdtemp()
    start_time = time.time()
    try:
        model = models.stt()

        wav, sr = load_audio(audio_path)
        chunks = split_audio_chunks(wav, sr)

        bridge.put(StreamEvent("start", {"total": len(chunks)}))

        tmp = tempfile.mkdtemp()
        try:
            for idx, (start, end, chunk) in enumerate(chunks, 1):
                path = os.path.join(tmp, f"c_{idx}.wav")
                wavfile.write(
                    path, sr, (chunk.squeeze(0).numpy() * 32767).astype("int16")
                )
                text = model.transcribe(path)
                bridge.put(
                    StreamEvent(
                        "progress",
                        {
                            "current": idx,
                            "total": len(chunks),
                            "segment": {"start": start, "end": end, "text": text},
                        },
                    )
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        bridge.put(StreamEvent("complete", {}))
        elapsed = time.time() - start_time
        log.info(f"STT completed: {elapsed:.2f}s")
        bridge.done()
    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"STT failed: {e} ({elapsed:.2f}s)")
        bridge.done(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _tts_worker(bridge: StreamBridge, text: str, speaker: str, filename: str | None = None) -> None:
    """Worker thread for TTS streaming."""
    start_time = time.time()
    try:
        model = models.tts()

        clean = clean_tts_text(text)
        chunks = split_tts_text(clean)
        total = len(chunks)

        bridge.put(StreamEvent("start", {"total": total}))

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
            bridge.put(StreamEvent("progress", {"current": idx, "total": total}))

        if not audio_parts:
            raise RuntimeError("All TTS chunks failed")

        full_audio = torch.cat(audio_parts, dim=0)
        output_path = os.path.join(tempfile.gettempdir(), f"tts_{time.time()}.wav")
        wavfile.write(
            output_path, Config.TTS_SR, (full_audio.numpy() * 32767).astype("int16")
        )

        download_url = f"/api/stream/download?p={os.path.basename(output_path)}"
        if filename:
            download_url += f"&filename={filename}"

        bridge.put(
            StreamEvent(
                "complete",
                {
                    "download_url": download_url,
                    "duration": len(full_audio) / Config.TTS_SR,
                    "chunks": total,
                },
            )
        )
        elapsed = time.time() - start_time
        log.info(f"TTS completed: {elapsed:.2f}s")
        bridge.done()
    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"TTS failed: {e} ({elapsed:.2f}s)")
        bridge.done(e)


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    device = os.getenv("DEVICE", "cpu")
    log.info(f"Using device: {device}")

    log.info("Pre-loading models...")
    await asyncio.to_thread(models.stt)
    await asyncio.to_thread(models.tts)
    log.info("Models loaded. Server ready.")
    yield
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
    allow_origins=["*"],
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


@app.post("/api/stream/stt")
async def stream_stt(file: UploadFile = File(...)) -> StreamingResponse:
    content = await file.read()

    if (
        Config.UPLOAD_LIMIT_MB > 0
        and len(content) > Config.UPLOAD_LIMIT_MB * 1024 * 1024
    ):
        raise HTTPException(413, f"File too large (max {Config.UPLOAD_LIMIT_MB}MB)")

    size_kb = len(content) / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
    log.info(f"STT started: {file.filename or 'unknown'} ({size_str})")

    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "input")
    with open(audio_path, "wb") as f:
        f.write(content)

    bridge = StreamBridge()
    asyncio.create_task(asyncio.to_thread(_stt_worker, bridge, audio_path))

    return StreamingResponse(
        bridge,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/stream/tts")
async def stream_tts(
    text: str = Query(..., min_length=1),
    speaker: str = Query(default=Config.TTS_SPEAKER),
    filename: str | None = Query(default=None),
) -> StreamingResponse:
    # Validation at boundary (cold path)
    if Config.TTS_MAX_INPUT_CHARS > 0 and len(text) > Config.TTS_MAX_INPUT_CHARS:
        raise HTTPException(
            413, f"Text too long (max {Config.TTS_MAX_INPUT_CHARS} chars)"
        )
    if speaker not in VALID_SPEAKERS:
        raise HTTPException(400, f"Invalid speaker. Use: {list(VALID_SPEAKERS)}")

    log.info(f"TTS started: {len(text)} chars, speaker={speaker}")

    bridge = StreamBridge()
    asyncio.create_task(asyncio.to_thread(_tts_worker, bridge, text, speaker, filename))

    return StreamingResponse(
        bridge,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/stream/download")
async def stream_download(p: str, filename: str | None = None) -> FileResponse:
    """Download streamed TTS audio file."""
    file_basename = os.path.basename(p)
    if not file_basename.endswith(".wav"):
        raise HTTPException(400, "Invalid file type")

    path = os.path.join(tempfile.gettempdir(), file_basename)
    if not os.path.exists(path):
        raise HTTPException(404, "Audio file not found")

    download_name = filename if filename else "resonance_tts.wav"
    if not download_name.endswith(".wav"):
        download_name += ".wav"

    return FileResponse(path, media_type="audio/wav", filename=download_name)


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
