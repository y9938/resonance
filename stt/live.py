from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np

from core.context import session_context_manager
from stt.inference import transcribe_serialized
from stt.stream_vad import _VAD_WINDOW_SAMPLES, VADStreamState, get_shared_vad_engine


class CommitDecision(Enum):
    KEEP_BUFFERING = auto()
    COMMIT_LONG_PAUSE = auto()
    COMMIT_SOFT_BOUNDARY = auto()
    COMMIT_HARD_LIMIT = auto()
    COMMIT_FLUSH = auto()


@dataclass(frozen=True)
class ASRCommitPolicy:
    """Pure timing policy; deliberately independent of transport and transcript state."""

    sample_rate: int
    long_pause_sec: float = 1.0
    soft_context_sec: float = 12.0
    hard_context_sec: float = 20.0

    def decide(self, *, samples: int, natural_boundary: bool, pause_samples: int, flush: bool) -> CommitDecision:
        if flush:
            return CommitDecision.COMMIT_FLUSH
        if samples >= self.hard_context_sec * self.sample_rate:
            return CommitDecision.COMMIT_HARD_LIMIT
        if natural_boundary and pause_samples >= self.long_pause_sec * self.sample_rate:
            return CommitDecision.COMMIT_LONG_PAUSE
        if natural_boundary and samples >= self.soft_context_sec * self.sample_rate:
            return CommitDecision.COMMIT_SOFT_BOUNDARY
        return CommitDecision.KEEP_BUFFERING


@dataclass(frozen=True)
class LivePreviewSubscription:
    """Opaque identity for one transient SSE consumer."""

    job_id: str
    token: int


class ASRCommitBuffer:
    """The sole owner of uncommitted contiguous PCM for one source."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.start_sample: int | None = None
        self._windows: list[np.ndarray] = []
        self.speech_samples = 0
        self.generation = 1
        self.last_preview_samples = 0

    @property
    def samples(self) -> int:
        return sum(len(window) for window in self._windows)

    @property
    def has_speech(self) -> bool:
        return self.speech_samples > 0

    def begin(self, start_sample: int, lead_in: list[np.ndarray]) -> None:
        if self._windows:
            return
        self.start_sample = max(0, start_sample)
        self._windows.extend(lead_in)

    def append(self, window: np.ndarray, is_speech: bool) -> None:
        if self.start_sample is None:
            raise RuntimeError("Cannot append PCM before commit buffer begins")
        self._windows.append(window)
        if is_speech:
            self.speech_samples += len(window)

    def snapshot(self) -> np.ndarray:
        return np.concatenate(self._windows) if self._windows else np.empty(0, dtype=np.float32)

    def take(self) -> tuple[int, np.ndarray, int]:
        if self.start_sample is None:
            raise RuntimeError("Cannot commit an empty buffer")
        start, pcm, generation = self.start_sample, self.snapshot(), self.generation
        self.start_sample = None
        self._windows.clear()
        self.speech_samples = 0
        self.last_preview_samples = 0
        self.generation += 1
        return start, pcm, generation


class LivePreviewBroker:
    """Ephemeral, non-replayable preview transport. It never touches JobRegistry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
        self._revisions: dict[str, int] = {}
        self._subscriptions: dict[int, tuple[str, int]] = {}
        self._job_tokens: dict[str, set[int]] = {}
        self._next_token = 1

    def subscribe(self, job_id: str) -> LivePreviewSubscription:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            # A reconnect starts fresh: this subscriber intentionally does not
            # receive previews published before it subscribed. Existing
            # subscribers retain their own observation cursors.
            self._subscriptions[token] = (job_id, self._revisions.get(job_id, 0))
            self._job_tokens.setdefault(job_id, set()).add(token)
            return LivePreviewSubscription(job_id, token)

    def unsubscribe(self, subscription: LivePreviewSubscription) -> None:
        with self._lock:
            state = self._subscriptions.get(subscription.token)
            if state is None or state[0] != subscription.job_id:
                return
            del self._subscriptions[subscription.token]
            tokens = self._job_tokens[subscription.job_id]
            tokens.discard(subscription.token)
            if not tokens:
                self._job_tokens.pop(subscription.job_id, None)
                self._latest.pop(subscription.job_id, None)
                self._revisions.pop(subscription.job_id, None)

    def publish(self, job_id: str, source: str, generation: int, text: str) -> None:
        with self._lock:
            if job_id not in self._job_tokens:
                return
            # Preview is a mutable tail: a slow client only needs the most
            # recent value per source, never an unbounded edit history. Each
            # subscriber tracks its own cursor, so one consumer cannot consume
            # or unsubscribe another consumer's latest value.
            revision = self._revisions.get(job_id, 0) + 1
            self._revisions[job_id] = revision
            event = {
                "type": "transcript_preview", "source": source,
                "generation": generation, "text": text,
            }
            self._latest.setdefault(job_id, {})[source] = (revision, event)

    def take(self, subscription: LivePreviewSubscription) -> list[dict[str, Any]]:
        with self._lock:
            state = self._subscriptions.get(subscription.token)
            if state is None or state[0] != subscription.job_id:
                return []
            _, seen_revision = state
            previews = [
                (revision, event)
                for revision, event in self._latest.get(subscription.job_id, {}).values()
                if revision > seen_revision
            ]
            if previews:
                self._subscriptions[subscription.token] = (
                    subscription.job_id,
                    max(revision for revision, _ in previews),
                )
            return [event for _, event in sorted(previews)]

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._latest.pop(job_id, None)
            self._revisions.pop(job_id, None)


class LiveSTTSession:
    """VAD endpointing plus per-source ASR aggregation and durable final publication."""

    def __init__(self, job_id: str, session_id: str, model: Any, jobs: Any, sample_rate: int = 16000,
                 source: str = "mic", dual_stream: bool = False, preview_broker: LivePreviewBroker | None = None) -> None:
        self.job_id, self.session_id, self.model, self.jobs = job_id, session_id, model, jobs
        self.sample_rate, self.source, self.dual_stream, self.preview_broker = sample_rate, source, dual_stream, preview_broker
        self._lock, self._cancelled, self._vad_engine = threading.RLock(), threading.Event(), get_shared_vad_engine()
        self._streams: dict[str, dict[str, Any]] = {}
        self._inference_stats = {"preview_calls": 0, "final_calls": 0, "preview_ms": [], "final_ms": [], "max_buffer_samples": 0}
        self._min_silence_windows = max(1, int(0.35 * sample_rate / _VAD_WINDOW_SAMPLES))
        self._lead_in_windows, self._preview_new_samples = int(0.15 * sample_rate / _VAD_WINDOW_SAMPLES), int(2.0 * sample_rate)

    def _get_or_create_stream(self, src: str) -> dict[str, Any]:
        if src not in self._streams:
            self._streams[src] = {"vad_state": VADStreamState(), "samples_read": 0, "is_speech": False,
                "silence_windows": 0, "cushion": [], "raw_carry": np.empty(0, dtype=np.float32),
                "buffer": ASRCommitBuffer(self.sample_rate), "policy": ASRCommitPolicy(self.sample_rate)}
        return self._streams[src]

    def process_pcm_chunk(self, pcm_chunk: np.ndarray, source: str | None = None) -> list[dict[str, Any]]:
        if self._cancelled.is_set():
            return []
        audio, src, emitted = np.asarray(pcm_chunk, dtype=np.float32).reshape(-1), source or self.source, []
        with self._lock:
            st = self._get_or_create_stream(src)
            audio = np.concatenate((st["raw_carry"], audio)) if len(st["raw_carry"]) else audio
            consumed = len(audio) // _VAD_WINDOW_SAMPLES * _VAD_WINDOW_SAMPLES
            st["raw_carry"] = audio[consumed:]
            for offset in range(0, consumed, _VAD_WINDOW_SAMPLES):
                window = audio[offset:offset + _VAD_WINDOW_SAMPLES]
                st["samples_read"] += len(window)
                speech = self._vad_engine.process_frame(st["vad_state"], window) >= 0.5
                buffer: ASRCommitBuffer = st["buffer"]
                natural_boundary = False
                if speech:
                    if not st["is_speech"]:
                        st["is_speech"] = True
                        if not buffer.samples:
                            buffer.begin(st["samples_read"] - len(window) - len(st["cushion"]) * _VAD_WINDOW_SAMPLES, st["cushion"])
                            st["cushion"] = []
                    buffer.append(window, True)
                    st["silence_windows"] = 0
                elif st["is_speech"]:
                    buffer.append(window, False)  # retain real pauses inside ASR context
                    st["silence_windows"] += 1
                    if st["silence_windows"] >= self._min_silence_windows:
                        st["is_speech"], natural_boundary = False, True
                elif buffer.has_speech:
                    # A local endpoint is not a commit: preserve continuing silence
                    # so a later long-pause decision has exact contiguous PCM.
                    buffer.append(window, False)
                    st["silence_windows"] += 1
                    natural_boundary = True
                else:
                    st["cushion"].append(window)
                    if len(st["cushion"]) > self._lead_in_windows:
                        st["cushion"].pop(0)
                decision = st["policy"].decide(samples=buffer.samples, natural_boundary=natural_boundary,
                    pause_samples=st["silence_windows"] * _VAD_WINDOW_SAMPLES, flush=False)
                self._inference_stats["max_buffer_samples"] = max(self._inference_stats["max_buffer_samples"], buffer.samples)
                if decision is not CommitDecision.KEEP_BUFFERING:
                    segment = self._commit(st, src)
                    st["is_speech"] = False
                    st["silence_windows"] = 0
                    if segment:
                        emitted.append(segment)
                elif buffer.has_speech and buffer.samples - buffer.last_preview_samples >= self._preview_new_samples:
                    self._preview(buffer, src)
        return emitted

    def flush(self, source: str | None = None) -> list[dict[str, Any]]:
        if self._cancelled.is_set():
            return []
        emitted = []
        with self._lock:
            for src in ([source] if source else list(self._streams)):
                st = self._streams.get(src)
                if st and st["buffer"].has_speech:
                    segment = self._commit(st, src)
                    if segment:
                        emitted.append(segment)
        return emitted

    @property
    def duration_sec(self) -> float:
        with self._lock:
            return max((s["samples_read"] + len(s["raw_carry"]) for s in self._streams.values()), default=0) / self.sample_rate

    def cancel(self) -> None:
        self._cancelled.set()
        with self._lock:
            for st in self._streams.values():
                if st["buffer"].samples:
                    st["buffer"].take()  # explicitly discard pending aggregate; never infer on cancel

    @property
    def inference_metrics(self) -> dict[str, float | int]:
        with self._lock:
            values = self._inference_stats
            all_ms = values["preview_ms"] + values["final_ms"]
            return {"preview_calls": values["preview_calls"], "final_calls": values["final_calls"],
                "p50_ms": float(np.percentile(all_ms, 50)) if all_ms else 0.0,
                "p95_ms": float(np.percentile(all_ms, 95)) if all_ms else 0.0,
                "max_buffer_samples": values["max_buffer_samples"]}

    def _transcribe(self, pcm: np.ndarray, kind: str) -> Any:
        started = time.perf_counter()
        try:
            return transcribe_serialized(self.model, pcm)
        finally:
            self._inference_stats[f"{kind}_calls"] += 1
            self._inference_stats[f"{kind}_ms"].append((time.perf_counter() - started) * 1000)

    def _preview(self, buffer: ASRCommitBuffer, src: str) -> None:
        if self.preview_broker is None or self._cancelled.is_set():
            return
        generation, snapshot = buffer.generation, buffer.snapshot()
        raw = self._transcribe(snapshot, "preview")
        if self._cancelled.is_set() or generation != buffer.generation:
            return
        from stt.pipeline import _segment_text_from_transcribe
        buffer.last_preview_samples = buffer.samples
        self.preview_broker.publish(self.job_id, src, generation, _segment_text_from_transcribe(raw).strip())

    def _commit(self, st: dict[str, Any], src: str) -> dict[str, Any] | None:
        buffer: ASRCommitBuffer = st["buffer"]
        if not buffer.has_speech:
            return None
        speech_samples = buffer.speech_samples
        start, pcm, generation = buffer.take()
        if speech_samples < int(0.2 * self.sample_rate):
            return None
        end, raw = start + len(pcm), self._transcribe(pcm, "final")
        if self._cancelled.is_set():
            return None
        from stt.pipeline import _segment_text_from_transcribe
        text = _segment_text_from_transcribe(raw).strip()
        if not text:
            return None
        if self.dual_stream:
            text = f"[SOURCE:{src.upper()}]: {text}"
        session_context_manager.append(self.session_id, text, start / self.sample_rate, end / self.sample_rate)
        segment = {"start": round(start / self.sample_rate, 3), "end": round(end / self.sample_rate, 3), "text": text, "source": src, "generation": generation}
        if self.jobs.update_event(self.job_id, "progress", {"current": round(end / self.sample_rate, 2), "total": round(end / self.sample_rate, 2), "segment": segment}) is False:
            raise RuntimeError("Live STT progress event was rejected")
        return segment
