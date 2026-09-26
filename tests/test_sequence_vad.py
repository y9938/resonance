from __future__ import annotations

import io
import itertools
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from stt.media import EncodedMedia, iter_media_frames
from stt.sequence_vad import get_sequence_vad_engine
from stt.stream_vad import (
    VADStreamState,
    get_shared_vad_engine,
    pack_utterances_into_chunks,
    segment_scored_frames,
    segment_vad_frames,
)


def _reference_probabilities(frames: list[np.ndarray]) -> np.ndarray:
    state = VADStreamState()
    engine = get_shared_vad_engine()
    return np.asarray([engine.process_frame(state, frame) for frame in frames])


def _fixture_frames() -> list[np.ndarray]:
    media = EncodedMedia(
        Path("tests/fixtures/ru_audio.wav").read_bytes(), "ru_audio.wav"
    )
    return list(iter_media_frames(media))


@pytest.mark.parametrize("block_frames", [1, 37, 128, 512])
def test_sequence_probabilities_preserve_state_across_blocks(block_frames: int) -> None:
    fixture = _fixture_frames()
    frames = list(itertools.islice(itertools.cycle(fixture), 1031))
    expected = _reference_probabilities(frames)

    scored = list(
        get_sequence_vad_engine()
        .new_stream(block_frames=block_frames)
        .score_frames(iter(frames))
    )
    actual = np.asarray([probability for _, probability in scored])

    assert len(scored) == len(frames)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(actual >= 0.5, expected >= 0.5)


def test_sequence_streams_are_isolated_when_interleaved() -> None:
    frames_a = _fixture_frames()
    frames_b = list(reversed(frames_a))
    expected_a = _reference_probabilities(frames_a)
    expected_b = _reference_probabilities(frames_b)
    engine = get_sequence_vad_engine()
    stream_a = engine.new_stream(block_frames=37)
    stream_b = engine.new_stream(block_frames=37)
    scored_a = stream_a.score_frames(iter(frames_a))
    scored_b = stream_b.score_frames(iter(frames_b))
    actual_a: list[float] = []
    actual_b: list[float] = []
    for a, b in itertools.zip_longest(scored_a, scored_b):
        if a is not None:
            actual_a.append(a[1])
        if b is not None:
            actual_b.append(b[1])

    np.testing.assert_allclose(actual_a, expected_a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(actual_b, expected_b, rtol=0, atol=1e-6)
    with pytest.raises(RuntimeError, match="exactly one audio stream"):
        stream_a.score_frames(iter(frames_a))


def test_sequence_streams_are_isolated_across_threads() -> None:
    frames = _fixture_frames()
    expected = _reference_probabilities(frames)
    engine = get_sequence_vad_engine()

    def run() -> np.ndarray:
        return np.asarray(
            [
                probability
                for _, probability in engine.new_stream().score_frames(iter(frames))
            ]
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run(), range(4)))

    for actual in results:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_sequence_preserves_utterances_and_packed_pcm() -> None:
    frames = _fixture_frames()
    stock = list(segment_vad_frames(iter(frames)))
    sequence = list(
        segment_scored_frames(
            get_sequence_vad_engine()
            .new_stream(block_frames=37)
            .score_frames(iter(frames))
        )
    )

    assert len(stock) == len(sequence) > 0
    for reference, actual in zip(stock, sequence, strict=True):
        assert (actual.start_sample, actual.end_sample) == (
            reference.start_sample,
            reference.end_sample,
        )
        np.testing.assert_array_equal(actual.pcm, reference.pcm)

    reference_chunks = list(pack_utterances_into_chunks(iter(stock)))
    sequence_chunks = list(pack_utterances_into_chunks(iter(sequence)))
    for reference, actual in zip(reference_chunks, sequence_chunks, strict=True):
        assert actual[:2] == reference[:2]
        np.testing.assert_array_equal(actual[2], reference[2])


def test_partial_pcm_at_eof_does_not_create_sequence_frame() -> None:
    samples = np.zeros(2 * 512 + 31, dtype=np.int16)
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    frames = list(iter_media_frames(EncodedMedia(encoded.getvalue(), "partial.wav")))
    scored = list(get_sequence_vad_engine().new_stream().score_frames(iter(frames)))

    assert len(frames) == len(scored) == 2


def test_sequence_reads_at_most_one_bounded_block_ahead() -> None:
    frame = np.zeros(512, dtype=np.float32)
    pulled = 0

    def source():
        nonlocal pulled
        for _ in range(100):
            pulled += 1
            yield frame

    scored = (
        get_sequence_vad_engine().new_stream(block_frames=37).score_frames(source())
    )
    assert pulled == 0
    next(scored)
    assert pulled == 37
