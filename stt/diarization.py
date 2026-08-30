from __future__ import annotations

import logging
import os
import shutil
import ssl
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

try:
    import certifi

    _SSL_CONTEXT: ssl.SSLContext | None = ssl.create_default_context(
        cafile=certifi.where()
    )
except Exception:
    _SSL_CONTEXT = None

log = logging.getLogger("resonance.stt.diarization")

# Domain Invariant: Both models are commercially permissible (MIT and Apache 2.0).
SEGMENTATION_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
# Workaround: eres2net_base zh-cn produces 3-6 false clusters on Russian speech (VoxCeleb multilingual
# coverage is sufficient; empirically verified on 2-speaker Russian audio: threshold=0.55 → exactly 2).
EMBEDDING_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
EMBEDDING_FILENAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"
# Workaround: 0.55 is the empirically determined threshold for wespeaker_resnet34_LM;
# below 0.55 the model oversplits (3+ clusters on 2-speaker audio).
CLUSTERING_THRESHOLD = float(os.getenv("RESONANCE_DIARIZATION_THRESHOLD", "0.55"))


@dataclass(frozen=True)
class SpeakerInterval:
    start_sec: float
    end_sec: float
    speaker_id: int


def _download_file(url: str, dest_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Resonance/0.1.0"})
    with urllib.request.urlopen(req, context=_SSL_CONTEXT) as response, open(
        dest_path, "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)


def get_sherpa_cache_dir() -> Path:
    # Domain Invariant: Explicit override takes precedence before standard XDG cache location.
    if path := (os.getenv("SHERPA_HOME") or os.getenv("RESONANCE_CACHE_DIR")):
        return Path(path)
    base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "resonance" / "sherpa"


def _ensure_models() -> tuple[Path, Path]:
    cache_dir = get_sherpa_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    seg_model = cache_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
    emb_model = cache_dir / EMBEDDING_FILENAME

    if not seg_model.exists():
        log.info(f"Downloading PyAnnote ONNX segmentation model to {cache_dir}...")
        archive = cache_dir / "seg.tar.bz2"
        _download_file(SEGMENTATION_URL, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(cache_dir)
        archive.unlink(missing_ok=True)

    if not emb_model.exists():
        log.info(f"Downloading speaker embedding model to {cache_dir}...")
        _download_file(EMBEDDING_URL, emb_model)

    assert seg_model.exists(), f"Fail-fast: Missing segmentation model at {seg_model}"
    assert emb_model.exists(), f"Fail-fast: Missing embedding model at {emb_model}"
    return seg_model, emb_model


_DIARIZER: sherpa_onnx.OfflineSpeakerDiarization | None = None


def get_diarizer() -> sherpa_onnx.OfflineSpeakerDiarization:
    global _DIARIZER
    if _DIARIZER is None:
        log.info("Loading Diarization model (PyAnnote Segmentation 3.0 + WeSpeaker VoxCeleb ResNet34)...")
        seg_model, emb_model = _ensure_models()
        threads = min(4, os.cpu_count() or 1)
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(seg_model),
                    # Domain Invariant: 0.25 (2.5s shift) gives 4x speedup over 0.1 with identical clustering accuracy.
                    window_shift_ratio=0.25,
                ),
                num_threads=threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(emb_model),
                num_threads=threads,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(threshold=CLUSTERING_THRESHOLD),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        assert config.validate(), "Fail-fast: sherpa-onnx diarization config validation failed"
        _DIARIZER = sherpa_onnx.OfflineSpeakerDiarization(config)
        log.info(f"Diarization model loaded (threads={threads}).")
    return _DIARIZER


def diarize_audio(
    audio_16k: np.ndarray,
    cancel_check: Callable[[], bool] | None = None,
) -> list[SpeakerInterval]:
    # Assumes: audio_16k is 1D float32 normalized [-1.0, 1.0] at 16000 Hz.
    assert audio_16k.ndim == 1, f"Expected 1D audio array, got shape {audio_16k.shape}"
    assert audio_16k.dtype == np.float32, f"Expected float32 dtype, got {audio_16k.dtype}"

    # Domain Invariant: Offline diarization requires at least 0.5s to extract meaningful embeddings.
    if len(audio_16k) < 16000 * 0.5:
        return []

    diarizer = get_diarizer()

    def progress_callback(processed: int, total: int) -> int:
        if cancel_check and cancel_check():
            raise RuntimeError("STT job cancelled")
        return 0

    segments = diarizer.process(audio_16k, callback=progress_callback).sort_by_start_time()

    return [
        SpeakerInterval(start_sec=s.start, end_sec=s.end, speaker_id=s.speaker)
        for s in segments
    ]


def match_speaker_tag(
    start_sec: float,
    end_sec: float,
    intervals: list[SpeakerInterval],
) -> str:
    """
    Matches the dominant speaker ID for a given time window using temporal overlap.
    """
    if not intervals:
        return ""

    best_speaker = None
    max_overlap = 0.0

    for iv in intervals:
        overlap = max(0.0, min(end_sec, iv.end_sec) - max(start_sec, iv.start_sec))
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = iv.speaker_id

    # Domain Invariant: Require at least 100ms overlap to assign speaker tag
    if best_speaker is not None and max_overlap > 0.1:
        return f"[Speaker {best_speaker + 1}]: "
    return ""
