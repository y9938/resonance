from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from stt.buffer import decode_media_bytes
from stt.media import (
    DecodedSampleLimitExceeded,
    DecodeStats,
    EncodedMedia,
    iter_media_frames,
    iter_media_pcm,
    media_duration,
)
from stt.pipeline import iter_stt_chunks
from stt.stream_vad import (
    pack_array_vad_chunks,
    pack_utterances_into_chunks,
    segment_vad_frames,
)
from tests.media_helpers import silent_wav


def test_sample_budget_counts_incomplete_vad_tail() -> None:
    allowed = silent_wav(16000)
    stats = DecodeStats()
    frames = list(iter_media_frames(allowed, max_samples=16000, stats=stats))
    assert len(frames) == 31
    assert stats.decoded_samples == 16000

    over_limit = silent_wav(16001)
    stats = DecodeStats()
    with pytest.raises(DecodedSampleLimitExceeded):
        list(iter_media_frames(over_limit, max_samples=16000, stats=stats))


def test_sample_budget_counts_resampler_flush_before_yield() -> None:
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams = [MagicMock(type="audio")]
    container.decode.return_value = [MagicMock()]
    converted = [MagicMock(), MagicMock()]
    converted[0].to_ndarray.return_value = np.zeros(16000, dtype=np.float32)
    converted[1].to_ndarray.return_value = np.zeros(1, dtype=np.float32)
    resampler = MagicMock()
    resampler.resample.side_effect = [[converted[0]], [converted[1]]]
    stats = DecodeStats()

    with patch("stt.media.av.open", return_value=container), patch(
        "stt.media.av.AudioResampler", return_value=resampler,
    ):
        stream = iter_media_pcm(EncodedMedia(b"encoded"), max_samples=16000, stats=stats)
        first_chunk = next(stream)
        with pytest.raises(DecodedSampleLimitExceeded):
            next(stream)

    assert len(first_chunk) == 16000
    assert stats.decoded_samples == 16001


def test_zero_metadata_duration_is_unknown() -> None:
    media = silent_wav(0)
    assert media_duration(media) is None


def test_missing_metadata_duration_is_unknown() -> None:
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams = [MagicMock(type="audio", duration=None)]
    container.duration = None
    with patch("stt.media.av.open", return_value=container):
        assert media_duration(EncodedMedia(b"encoded")) is None


@pytest.mark.parametrize("stream_duration", [0, float("nan")])
def test_unusable_stream_duration_falls_back_to_container(stream_duration) -> None:
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams = [MagicMock(type="audio", duration=stream_duration, time_base=1 / 16000)]
    container.duration = 2 * av.time_base
    with patch("stt.media.av.open", return_value=container):
        assert media_duration(EncodedMedia(b"encoded")) == 2.0


def test_lazy_decoder_matches_full_reference_and_chunk_boundaries() -> None:
    encoded = Path("tests/fixtures/ru_audio.wav").read_bytes()
    media = EncodedMedia(encoded, "ru_audio.wav")
    reference = decode_media_bytes(encoded).as_ndarray()
    frames = list(iter_media_frames(media))
    streamed = np.concatenate(frames)

    assert media_duration(media) == len(reference) / 16000
    assert all(len(frame) == 512 and frame.dtype == np.float32 for frame in frames)
    np.testing.assert_array_equal(streamed, reference[:len(streamed)])
    assert len(reference) - len(streamed) < 512

    old = [(start, end) for start, end, _ in pack_array_vad_chunks(reference)]
    new = [
        (start, end)
        for start, end, _ in pack_utterances_into_chunks(
            segment_vad_frames(iter_media_frames(media)), total_duration_sec=media_duration(media)
        )
    ]
    assert new == old


def test_encoded_batch_chunks_match_frame_vad_reference() -> None:
    media = EncodedMedia(Path("tests/fixtures/ru_audio.wav").read_bytes(), "ru_audio.wav")
    duration = media_duration(media)
    reference = list(
        pack_utterances_into_chunks(
            segment_vad_frames(iter_media_frames(media)),
            target_sec=20,
            total_duration_sec=duration,
        )
    )
    actual = list(
        iter_stt_chunks(
            {"default": media}, sample_rate=16000, chunk_sec=20, total_duration_sec=duration
        )
    )

    assert len(actual) == len(reference) > 0
    for chunk, (start, end, pcm) in zip(actual, reference, strict=True):
        assert (chunk.start_sec, chunk.end_sec, chunk.source) == (start, end, "default")
        np.testing.assert_array_equal(chunk.pcm, pcm)


def test_batch_upload_passes_encoded_bytes_to_worker_without_decoding() -> None:
    from fastapi.testclient import TestClient

    from server import app

    encoded = Path("tests/fixtures/ru_audio.wav").read_bytes()
    with (
        patch("server.models.stt_gigaam", return_value=MagicMock()),
        patch("server.run_stt_worker") as worker,
        patch("server.decode_media_bytes", side_effect=AssertionError("eager decode")),
    ):
        response = TestClient(app).post(
            "/api/jobs/stt", files={"file": ("ru_audio.wav", encoded, "audio/wav")}
        )

    assert response.status_code == 200
    media = worker.call_args.kwargs["audio_path"]
    assert isinstance(media, EncodedMedia)
    assert media.data == encoded
    assert "batch_size" not in worker.call_args.kwargs


def test_batch_upload_accepts_unknown_metadata_duration() -> None:
    from fastapi.testclient import TestClient

    from server import app

    media = silent_wav(16000)
    with (
        patch("server.models.stt_gigaam", return_value=MagicMock()),
        patch("server.media_duration", return_value=None) as duration,
        patch("server.run_stt_worker"),
    ):
        response = TestClient(app).post(
            "/api/jobs/stt", files={"file": ("silent.wav", media.data, "audio/wav")},
        )

    assert response.status_code == 200
    duration.assert_called_once()


def test_batch_upload_rejects_invalid_container_before_job_creation() -> None:
    from fastapi.testclient import TestClient

    from server import app

    with (
        patch("server.models.stt_gigaam", return_value=MagicMock()),
        patch("server.run_stt_worker") as worker,
    ):
        response = TestClient(app).post(
            "/api/jobs/stt", files={"file": ("bad.mp4", b"not media", "video/mp4")}
        )

    assert response.status_code == 400
    worker.assert_not_called()
