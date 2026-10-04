from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from stt.inference import InferenceGate, transcribe_batch_serialized
from stt.models.base import STTModelAdapter
from stt.models.gigaam import GigaAMAdapter
from stt.pipeline import run_stt_job
from tests.test_stt_pipeline import FakeJobs, FakeLog


def test_gigaam_batch_preserves_input_order_and_real_lengths() -> None:
    model = MagicMock()
    model._device = torch.device("cpu")
    model._dtype = torch.float32

    def forward(padded, lengths):
        assert padded.shape == ((2, 4) if len(lengths) == 2 else (1, 2))
        assert lengths.tolist() == ([2, 4] if len(lengths) == 2 else [2])
        assert padded[0].tolist() == ([1, 2, 0, 0] if len(lengths) == 2 else [1, 2])
        return padded, lengths

    model.forward.side_effect = forward
    model._decode.side_effect = lambda encoded, encoded_len, lengths, timestamps: [
        (str(int(encoded[index, 0].item())), None) for index in range(len(lengths))
    ]
    adapter = GigaAMAdapter(model)
    chunks = (np.array([1, 2], dtype=np.float32), np.array([3, 4, 5, 6], dtype=np.float32))

    assert adapter.transcribe_batch(chunks) == ["1", "3"]
    assert adapter.transcribe(chunks[0]) == "1"
    assert model.forward.call_count == 2


def test_gigaam_multi_item_inference_passes_common_boundary() -> None:
    model = MagicMock()
    model._device = torch.device("cpu")
    model._dtype = torch.float32
    model.forward.side_effect = lambda padded, lengths: (padded, lengths)
    model._decode.side_effect = lambda encoded, encoded_len, lengths, timestamps: [
        (str(int(encoded[index, 0].item())), None) for index in range(len(lengths))
    ]
    chunks = (np.array([1], dtype=np.float32), np.array([2, 3], dtype=np.float32))

    assert transcribe_batch_serialized(GigaAMAdapter(model), chunks) == ["1", "2"]
    model.forward.assert_called_once()


def test_multi_item_inference_requires_explicit_native_batching() -> None:
    model = MagicMock()
    with pytest.raises(ValueError):
        transcribe_batch_serialized(model, (np.zeros(1), np.zeros(1)))
    model.transcribe_batch.assert_not_called()


def test_production_pipeline_defaults_to_single_inference_even_with_native_capability(monkeypatch) -> None:
    calls: list[int] = []

    def chunks(*args, **kwargs):
        for index in range(3):
            yield float(index), float(index + 1), np.array([index], dtype=np.float32)

    class NativeCapableModel:
        supports_native_batching = True

        def transcribe_batch(self, audio):
            calls.append(len(audio))
            return [str(int(item[0])) for item in audio]

    monkeypatch.setattr("stt.pipeline.pack_array_vad_chunks", chunks)
    run_stt_job(
        job_id="batch", input_paths=np.ones(16000, dtype=np.float32),
        jobs=FakeJobs(), model=NativeCapableModel(), log=FakeLog(),
        sample_rate=16000, chunk_sec=20,
    )

    assert calls == [1, 1, 1]


def test_inference_batch_rejects_wrong_cardinality() -> None:
    model = MagicMock()
    model.supports_native_batching = True
    model.transcribe_batch.return_value = ["only one"]
    with pytest.raises(RuntimeError):
        transcribe_batch_serialized(model, (np.zeros(1), np.zeros(1)))


def test_batch_pipeline_passes_resolved_language_to_adapter(monkeypatch) -> None:
    seen: list[str | None] = []

    class Model(STTModelAdapter):
        @classmethod
        def supported_languages(cls) -> frozenset[str]:
            return frozenset({"de"})

        def transcribe(self, audio, *, language=None, **kwargs):
            seen.append(language)
            return "Hallo"

    monkeypatch.setattr(
        "stt.pipeline.pack_array_vad_chunks",
        lambda *args, **kwargs: (chunk for chunk in [(0.0, 1.0, np.ones(16000, dtype=np.float32))]),
    )
    run_stt_job(
        job_id="german", input_paths=np.ones(16000, dtype=np.float32),
        jobs=FakeJobs(), model=Model(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, language="de",
    )
    assert seen == ["de"]


def test_batch_auto_locks_language_after_first_chunk(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class Model:
        def transcribe_and_detect_language(self, audio):
            calls.append(("detect", None))
            return "Hallo", "de"

        def transcribe(self, audio, *, language):
            calls.append(("transcribe", language))
            return "Welt"

    monkeypatch.setattr(
        "stt.pipeline.pack_array_vad_chunks",
        lambda *args, **kwargs: (chunk for chunk in [
            (0.0, 1.0, np.ones(16000, dtype=np.float32)),
            (1.0, 2.0, np.ones(16000, dtype=np.float32)),
        ]),
    )
    jobs = FakeJobs()
    run_stt_job(
        job_id="auto", input_paths=np.ones(32000, dtype=np.float32),
        jobs=jobs, model=Model(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, detect_language=True,
    )
    assert calls == [("detect", None), ("transcribe", "de")]
    assert [event for event, _ in jobs.events] == ["start", "progress", "progress", "complete"]
    assert [data["segment"]["text"] for event, data in jobs.events if event == "progress"] == [
        "Hallo", "Welt",
    ]


def test_batch_auto_with_no_speech_never_calls_model(monkeypatch) -> None:
    monkeypatch.setattr("stt.pipeline.pack_array_vad_chunks", lambda *args, **kwargs: (chunk for chunk in ()))
    model = MagicMock()
    jobs = FakeJobs()
    run_stt_job(
        job_id="silent", input_paths=np.zeros(16000, dtype=np.float32),
        jobs=jobs, model=model, log=FakeLog(), sample_rate=16000,
        chunk_sec=20, detect_language=True,
    )
    model.transcribe_and_detect_language.assert_not_called()
    assert [event for event, _ in jobs.events] == ["start", "complete"]


def test_live_waiter_runs_before_next_batch() -> None:
    gate = InferenceGate()
    first_started = threading.Event()
    release_first = threading.Event()
    order: list[str] = []

    def first_batch():
        first_started.set()
        assert release_first.wait(2)
        order.append("batch one")

    def next_call(label: str, live: bool):
        gate.call(lambda: order.append(label), live=live)

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(gate.call, first_batch, live=False)
        assert first_started.wait(2)
        live = pool.submit(next_call, "live", True)
        with gate._condition:
            assert gate._condition.wait_for(lambda: gate._live_waiters == 1, timeout=2)
        batch = pool.submit(next_call, "batch two", False)
        release_first.set()
        first.result(timeout=2)
        live.result(timeout=2)
        batch.result(timeout=2)

    assert order == ["batch one", "live", "batch two"]


def test_batch_pipeline_commits_in_order_without_prefetch(monkeypatch) -> None:
    produced: list[int] = []

    def chunks(*args, **kwargs):
        for index in range(5):
            produced.append(index)
            yield float(index), float(index + 1), np.array([index], dtype=np.float32)

    class Model:
        supports_native_batching = True

        def transcribe_batch(self, audio):
            assert len(produced) == (4 if not calls else 5)
            calls.append(len(audio))
            return [str(int(item[0])) for item in audio]

    calls: list[int] = []
    monkeypatch.setattr("stt.pipeline.pack_array_vad_chunks", chunks)
    jobs = FakeJobs()
    run_stt_job(
        job_id="batch", input_paths=np.ones(16000, dtype=np.float32),
        jobs=jobs, model=Model(), log=FakeLog(), sample_rate=16000,
        chunk_sec=20, batch_size=3,
    )

    assert calls == [3, 2]
    progress = [data["segment"] for kind, data in jobs.events if kind == "progress"]
    assert [(item["start"], item["text"]) for item in progress] == [
        (float(index), str(index)) for index in range(5)
    ]
