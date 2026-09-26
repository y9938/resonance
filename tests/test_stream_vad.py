import subprocess
import sys
import threading

import numpy as np
import pytest

from stt.stream_vad import (
    VADStreamState,
    get_shared_vad_engine,
    stream_vad_chunks,
)


def test_stateless_silero_vad_isolation() -> None:
    engine = get_shared_vad_engine()
    state_a = VADStreamState()
    state_b = VADStreamState()

    silence_frame = np.zeros(512, dtype=np.float32)
    tone_frame = (0.5 * np.sin(np.linspace(0, 50, 512, dtype=np.float32))).astype(np.float32)

    engine.process_frame(state_a, tone_frame)
    prob_b_silence = engine.process_frame(state_b, silence_frame)

    assert prob_b_silence < 0.05
    assert not np.array_equal(state_a.state, state_b.state)
    assert not np.array_equal(state_a.context, state_b.context)

    state_a.reset()
    assert np.all(state_a.state == 0)
    assert np.all(state_a.context == 0)


def test_concurrent_stream_vad_chunks_bit_exact() -> None:
    audio_file = "tests/fixtures/ru_audio.wav"

    baseline_chunks = list(stream_vad_chunks(audio_file, target_sec=5.0))
    assert len(baseline_chunks) > 0

    results: list[list | None] = [None] * 4

    def worker(idx: int) -> None:
        results[idx] = list(stream_vad_chunks(audio_file, target_sec=5.0))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for idx, thread_chunks in enumerate(results):
        assert thread_chunks is not None
        assert len(thread_chunks) == len(baseline_chunks)
        for c_idx, (b_item, c_item) in enumerate(zip(baseline_chunks, thread_chunks)):
            assert b_item[0] == c_item[0]
            assert b_item[1] == c_item[1]
            np.testing.assert_array_equal(b_item[2], c_item[2])

@pytest.mark.parametrize("thread_count", [1, 4])
@pytest.mark.parametrize("backend", ["frame", "sequence"])
def test_vad_initialization_preserves_pytorch_threads(thread_count: int, backend: str) -> None:
    module, factory = (
        ("stt.stream_vad", "get_shared_vad_engine")
        if backend == "frame"
        else ("stt.sequence_vad", "get_sequence_vad_engine")
    )
    code = (
        "import sys, torch\n"
        f"torch.set_num_threads({thread_count})\n"
        f"from {module} import {factory}\n"
        f"engine = {factory}()\n"
        f"assert torch.get_num_threads() == {thread_count}\n"
        "assert 'silero_vad' not in sys.modules\n"
        "assert engine._session.get_session_options().intra_op_num_threads == 1\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
