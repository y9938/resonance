from __future__ import annotations

import logging
import subprocess
from typing import Iterator, Tuple

import numpy as np
import torch

# Workaround: silero_vad package import mutates global PyTorch state via torch.set_num_threads(1).
# We preserve PyTorch's self-calibrated thread count to maintain optimal multi-core performance for ASR.
# Assumes: Silero VAD (ONNX) preserves single-threaded execution via internal SessionOptions.
_INITIAL_PYTORCH_THREADS = torch.get_num_threads()
from silero_vad import load_silero_vad
torch.set_num_threads(_INITIAL_PYTORCH_THREADS)



logger = logging.getLogger(__name__)

# Assumes: Silero VAD strictly requires 16000 Hz.
_SAMPLE_RATE = 16000
_VAD_WINDOW_SAMPLES = 512
_WINDOW_BYTES = _VAD_WINDOW_SAMPLES * 2  # s16le: 2 bytes per sample

_VAD_MODEL = None


def _get_vad_model():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        logger.info("Initializing Silero VAD (ONNX)...")
        _VAD_MODEL = load_silero_vad(onnx=True)
    return _VAD_MODEL


from dataclasses import dataclass

@dataclass
class Utterance:
    start_sample: int
    end_sample: int
    pcm: np.ndarray


def _stream_vad_utterances(
    input_path: str,
    sample_rate: int = 16000,
    silence_threshold: float = 0.5,
    min_silence_duration: float = 0.128,
    max_speech_duration: float = 24.0,
) -> Iterator[Utterance]:
    model = _get_vad_model()
    model.reset_states()

    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:
        raise RuntimeError(f"Failed to start FFmpeg: {exc}") from exc

    min_silence_windows = int(min_silence_duration * sample_rate / _VAD_WINDOW_SAMPLES)
    max_speech_windows = int(max_speech_duration * sample_rate / _VAD_WINDOW_SAMPLES)
    lead_in_windows = 8  # 256ms lead-in ring cushion

    cushion_windows: list[np.ndarray] = []
    current_speech_windows: list[np.ndarray] = []
    is_speech = False
    speech_start_sample = 0
    silence_window_count = 0
    total_samples_read = 0
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
                total_samples_read += _VAD_WINDOW_SAMPLES

                window_f32 = window_s16.astype(np.float32) / 32768.0
                prob = model(torch.from_numpy(window_f32).unsqueeze(0), sample_rate).item()

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


def stream_vad_chunks(
    input_path: str,
    sample_rate: int = 16000,
    target_sec: float = 18.0,
    max_sec: float = 24.0,
    max_gap_sec: float = 3.0,
    silence_threshold: float = 0.5,
    total_duration_sec: float = 0.0,
) -> Iterator[Tuple[float, float, np.ndarray]]:
    """
    Workaround: Greedy Utterance Packing eliminates intra-word cuts and time drift.
    Assumes: sample_rate is 16000 (Silero ONNX strict contract).
    """
    assert sample_rate == _SAMPLE_RATE, f"Silero VAD ONNX requires {_SAMPLE_RATE} Hz, got {sample_rate}"
    assert target_sec < max_sec, "target_sec must be less than max_sec"

    current_batch: list[Utterance] = []

    def _flush_batch() -> Tuple[float, float, np.ndarray] | None:
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

    for utterance in _stream_vad_utterances(
        input_path, sample_rate, silence_threshold, min_silence_duration=0.128, max_speech_duration=max_sec
    ):
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
