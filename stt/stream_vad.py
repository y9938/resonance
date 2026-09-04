from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
import torch

# Workaround: silero_vad package import mutates global PyTorch state via torch.set_num_threads(1).
# We preserve PyTorch's self-calibrated thread count to maintain optimal multi-core performance for ASR.
# Assumes: Silero VAD (ONNX) preserves single-threaded execution via internal SessionOptions.
_INITIAL_PYTORCH_THREADS = torch.get_num_threads()
import silero_vad  # noqa: I001
torch.set_num_threads(_INITIAL_PYTORCH_THREADS)

logger = logging.getLogger("resonance.stt.stream_vad")

# Assumes: Silero VAD strictly requires 16000 Hz.
_SAMPLE_RATE = 16000
_VAD_WINDOW_SAMPLES = 512
_WINDOW_BYTES = _VAD_WINDOW_SAMPLES * 2  # s16le: 2 bytes per sample
_CONTEXT_SIZE = 64  # 4ms context prefix at 16kHz for Silero CNN boundary alignment

_SHARED_VAD_ENGINE: StatelessSileroVAD | None = None


@dataclass
class VADStreamState:
    # Invariant: Recurrent tensor layout strictly matching Silero ONNX input: (2, batch_size=1, 128)
    state: np.ndarray = field(default_factory=lambda: np.zeros((2, 1, 128), dtype=np.float32))
    context: np.ndarray = field(default_factory=lambda: np.zeros((1, _CONTEXT_SIZE), dtype=np.float32))

    def reset(self) -> None:
        self.state.fill(0)
        self.context.fill(0)


class StatelessSileroVAD:
    """Thread-safe, stateless Silero VAD wrapper sharing a single C++ ONNX Runtime session."""

    def __init__(self, onnx_path: str) -> None:
        # Assumes: SessionOptions single-threaded per session to prevent CPU thread contention under multi-stream load.
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"], sess_options=opts)
        self._sr = np.array(16000, dtype=np.int64)

    def process_frame(self, stream_state: VADStreamState, frame_512: np.ndarray) -> float:
        """
        Caller obligation: frame_512 must be a 1D float32 array with strictly 512 samples.
        Thread-safety: InferenceSession.run() is reentrant; all mutations are strictly confined to stream_state.
        """
        x = np.concatenate([stream_state.context, frame_512.reshape(1, _VAD_WINDOW_SAMPLES)], axis=1)
        ort_inputs = {"input": x, "state": stream_state.state, "sr": self._sr}
        out, new_state = self._session.run(None, ort_inputs)

        stream_state.state = new_state
        stream_state.context = x[:, -_CONTEXT_SIZE:]
        return float(out[0, 0])


def get_shared_vad_engine() -> StatelessSileroVAD:
    global _SHARED_VAD_ENGINE
    if _SHARED_VAD_ENGINE is None:
        logger.info("Initializing Stateless Silero VAD (ONNX)...")
        onnx_path = os.path.join(os.path.dirname(silero_vad.__file__), "data", "silero_vad.onnx")
        _SHARED_VAD_ENGINE = StatelessSileroVAD(onnx_path)
    return _SHARED_VAD_ENGINE


@dataclass
class Utterance:
    start_sample: int
    end_sample: int
    pcm: np.ndarray


def segment_vad_frames(
    frames_iterator: Iterator[np.ndarray],
    sample_rate: int = 16000,
    silence_threshold: float = 0.5,
    min_silence_duration: float = 0.128,
    max_speech_duration: float = 24.0,
    vad_engine: StatelessSileroVAD | None = None,
) -> Iterator[Utterance]:
    """
    Pure in-RAM streaming speech detector.
    Caller obligation: frames_iterator must yield 1D float32 chunks of strictly 512 samples.
    """
    engine = vad_engine or get_shared_vad_engine()
    stream_state = VADStreamState()

    min_silence_windows = int(min_silence_duration * sample_rate / _VAD_WINDOW_SAMPLES)
    max_speech_windows = int(max_speech_duration * sample_rate / _VAD_WINDOW_SAMPLES)
    lead_in_windows = 8  # 256ms lead-in ring cushion

    cushion_windows: list[np.ndarray] = []
    current_speech_windows: list[np.ndarray] = []
    is_speech = False
    speech_start_sample = 0
    silence_window_count = 0
    total_samples_read = 0

    for window_f32 in frames_iterator:
        total_samples_read += _VAD_WINDOW_SAMPLES
        prob = engine.process_frame(stream_state, window_f32)

        if prob >= silence_threshold:
            if not is_speech:
                is_speech = True
                speech_start_sample = total_samples_read - _VAD_WINDOW_SAMPLES - (len(cushion_windows) * _VAD_WINDOW_SAMPLES)
                current_speech_windows = list(cushion_windows)
                cushion_windows = []
            current_speech_windows.append(window_f32)
            silence_window_count = 0

            if len(current_speech_windows) >= max_speech_windows:
                speech_pcm = np.concatenate(current_speech_windows)
                yield Utterance(
                    start_sample=max(0, speech_start_sample),
                    end_sample=total_samples_read,
                    pcm=speech_pcm,
                )
                current_speech_windows = []
                speech_start_sample = total_samples_read
        else:
            if is_speech:
                current_speech_windows.append(window_f32)
                silence_window_count += 1
                if silence_window_count >= min_silence_windows:
                    is_speech = False
                    speech_pcm = np.concatenate(current_speech_windows)
                    yield Utterance(
                        start_sample=max(0, speech_start_sample),
                        end_sample=total_samples_read,
                        pcm=speech_pcm,
                    )
                    current_speech_windows = []
                    cushion_windows = []
                    silence_window_count = 0
            else:
                cushion_windows.append(window_f32)
                if len(cushion_windows) > lead_in_windows:
                    cushion_windows.pop(0)

    if is_speech and current_speech_windows:
        speech_pcm = np.concatenate(current_speech_windows)
        yield Utterance(
            start_sample=max(0, speech_start_sample),
            end_sample=total_samples_read,
            pcm=speech_pcm,
        )


def vad_segment_array(
    audio: np.ndarray,
    sample_rate: int = 16000,
    silence_threshold: float = 0.5,
    min_silence_duration: float = 0.128,
    max_speech_duration: float = 24.0,
    vad_engine: StatelessSileroVAD | None = None,
) -> Iterator[Utterance]:
    """
    Zero-I/O VAD segmentation directly on in-memory 1D float32 audio without FFmpeg.
    """
    assert audio.ndim == 1, "Audio array must be 1D"
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    def _frame_generator() -> Iterator[np.ndarray]:
        for i in range(0, len(audio) - _VAD_WINDOW_SAMPLES + 1, _VAD_WINDOW_SAMPLES):
            yield audio[i : i + _VAD_WINDOW_SAMPLES]

    return segment_vad_frames(
        _frame_generator(),
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        min_silence_duration=min_silence_duration,
        max_speech_duration=max_speech_duration,
        vad_engine=vad_engine,
    )


def _stream_vad_utterances(
    input_path: str,
    sample_rate: int = 16000,
    silence_threshold: float = 0.5,
    min_silence_duration: float = 0.128,
    max_speech_duration: float = 24.0,
    vad_engine: StatelessSileroVAD | None = None,
) -> Iterator[Utterance]:
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        raise RuntimeError(f"Failed to start FFmpeg: {exc}") from exc

    def _ffmpeg_frame_generator() -> Iterator[np.ndarray]:
        raw_carry = b""
        try:
            while True:
                raw = proc.stdout.read(4096)
                if not raw:
                    break
                raw_carry += raw
                while len(raw_carry) >= _WINDOW_BYTES:
                    window_s16 = np.frombuffer(raw_carry[:_WINDOW_BYTES], dtype=np.int16)
                    raw_carry = raw_carry[_WINDOW_BYTES:]
                    yield window_s16.astype(np.float32) / 32768.0

            proc.wait(timeout=15.0)
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode().strip()
                raise RuntimeError(f"FFmpeg failed (code {proc.returncode}): {stderr}")
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.terminate()

    return segment_vad_frames(
        _ffmpeg_frame_generator(),
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        min_silence_duration=min_silence_duration,
        max_speech_duration=max_speech_duration,
        vad_engine=vad_engine,
    )


def pack_utterances_into_chunks(
    utterances: Iterator[Utterance],
    sample_rate: int = 16000,
    target_sec: float = 18.0,
    max_sec: float = 24.0,
    max_gap_sec: float = 3.0,
    total_duration_sec: float = 0.0,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """
    Workaround: Greedy Utterance Packing eliminates intra-word cuts and time drift.
    Assumes: sample_rate is 16000 (Silero ONNX strict contract).
    """
    assert sample_rate == _SAMPLE_RATE, f"Silero VAD ONNX requires {_SAMPLE_RATE} Hz, got {sample_rate}"
    assert target_sec < max_sec, "target_sec must be less than max_sec"

    current_batch: list[Utterance] = []

    def _flush_batch() -> tuple[float, float, np.ndarray] | None:
        nonlocal current_batch
        if not current_batch:
            return None
        start_sec = current_batch[0].start_sample / sample_rate
        end_sec = current_batch[-1].end_sample / sample_rate

        total_samples = current_batch[-1].end_sample - current_batch[0].start_sample
        pcm = np.zeros(total_samples, dtype=np.float32)
        base_sample = current_batch[0].start_sample
        for u in current_batch:
            offset = u.start_sample - base_sample
            pcm[offset : offset + len(u.pcm)] = u.pcm
        current_batch = []
        return start_sec, end_sec, pcm

    for utterance in utterances:
        if not current_batch:
            current_batch.append(utterance)
        else:
            gap_sec = (utterance.start_sample - current_batch[-1].end_sample) / sample_rate
            projected_dur = (utterance.end_sample - current_batch[0].start_sample) / sample_rate

            if gap_sec > max_gap_sec or projected_dur > max_sec:
                chunk = _flush_batch()
                if chunk:
                    yield chunk
                current_batch.append(utterance)
            elif projected_dur >= target_sec:
                remaining_sec = total_duration_sec - (utterance.end_sample / sample_rate) if total_duration_sec > 0 else 999.0
                if remaining_sec <= 5.0 and (projected_dur + remaining_sec) <= max_sec:
                    current_batch.append(utterance)
                else:
                    current_batch.append(utterance)
                    chunk = _flush_batch()
                    if chunk:
                        yield chunk
            else:
                current_batch.append(utterance)

    if current_batch:
        chunk = _flush_batch()
        if chunk:
            yield chunk


def pack_array_vad_chunks(
    audio: np.ndarray,
    sample_rate: int = 16000,
    target_sec: float = 18.0,
    max_sec: float = 24.0,
    max_gap_sec: float = 3.0,
    silence_threshold: float = 0.5,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """
    Zero-I/O VAD chunker directly on 1D float32 audio array.
    """
    total_sec = len(audio) / sample_rate
    utterances = vad_segment_array(
        audio,
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        max_speech_duration=max_sec,
    )
    return pack_utterances_into_chunks(
        utterances,
        sample_rate=sample_rate,
        target_sec=target_sec,
        max_sec=max_sec,
        max_gap_sec=max_gap_sec,
        total_duration_sec=total_sec,
    )


def stream_vad_chunks(
    input_path: str,
    sample_rate: int = 16000,
    target_sec: float = 18.0,
    max_sec: float = 24.0,
    max_gap_sec: float = 3.0,
    silence_threshold: float = 0.5,
    total_duration_sec: float = 0.0,
) -> Iterator[tuple[float, float, np.ndarray]]:
    utterances = _stream_vad_utterances(
        input_path,
        sample_rate=sample_rate,
        silence_threshold=silence_threshold,
        min_silence_duration=0.128,
        max_speech_duration=max_sec,
    )
    return pack_utterances_into_chunks(
        utterances,
        sample_rate=sample_rate,
        target_sec=target_sec,
        max_sec=max_sec,
        max_gap_sec=max_gap_sec,
        total_duration_sec=total_duration_sec,
    )
