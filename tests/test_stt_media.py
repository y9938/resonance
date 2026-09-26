from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from stt.buffer import decode_media_bytes
from stt.media import EncodedMedia, iter_media_frames, media_duration
from stt.stream_vad import (
    pack_array_vad_chunks,
    pack_utterances_into_chunks,
    segment_vad_frames,
)


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
