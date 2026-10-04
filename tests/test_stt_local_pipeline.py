"""Real FFmpeg path decoding obeys the public batch duration/lifetime contract."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from stt import pipeline, stream_vad
from tests.media_helpers import silent_wav
from tests.test_stt_pipeline import FakeJobs, FakeLog, FakeModel, SilentVAD


@pytest.mark.parametrize("samples,hint,limit,error", [
    (16001, 0, 2, None),       # Unknown duration, including the incomplete VAD tail.
    (16001, 0.5, 1, "Audio too long"),
    (16001, 0, 1, "Audio too long"),
    (16000, 2, 1, "Audio too long"),
    (0, 0, 1, "empty after decoding"),
    (0, 0.5, 1, "empty after decoding"),
    (16000, 1, 1, None),      # Silence is valid, also exactly at the limit.
])
@pytest.mark.parametrize("diarization", [False, True])
def test_path_duration_contract(tmp_path, monkeypatch, samples, hint, limit, error, diarization):
    path = tmp_path / "audio.wav"
    path.write_bytes(silent_wav(samples).data)
    probe = pipeline.probe_media
    monkeypatch.setattr(pipeline, "probe_media", lambda path: replace(probe(path), duration_sec=hint))
    monkeypatch.setattr(stream_vad, "get_sequence_vad_engine", lambda: SilentVAD())
    diarize = MagicMock(return_value=[])
    monkeypatch.setattr("stt.diarization.diarize_audio", diarize)
    jobs = FakeJobs()
    pipeline.run_stt_job(
        job_id="path", input_paths=str(path), jobs=jobs, model=FakeModel(), log=FakeLog(),
        sample_rate=16000, chunk_sec=20, max_duration_sec=limit, diarization=diarization,
    )
    event, data = jobs.events[-1]
    if error:
        assert event == "error"
        assert error in data["message"]
        diarize.assert_not_called()
    else:
        assert event == "complete"
        assert data["duration"] == samples / 16000
        if hint == 0:
            assert jobs.events[0][1]["total"] == 0
        if diarization:
            assert len(diarize.call_args.args[0]) == samples


def test_probe_accepts_missing_duration_without_decoding(tmp_path, monkeypatch):
    path = tmp_path / "audio.webm"
    path.touch()
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: MagicMock(
        stdout='{"streams":[{"codec_type":"audio"}],"format":{"duration":"N/A"}}',
    ))
    assert pipeline.probe_media(path).duration_sec == 0


def test_cancel_during_silent_decode_reaps_ffmpeg(tmp_path, monkeypatch):
    source = tmp_path / "silence.wav"
    source.write_bytes(silent_wav(16000 * 30).data)
    jobs = FakeJobs()
    processes = []
    popen = stream_vad.subprocess.Popen

    def track_process(*args, **kwargs):
        proc = popen(*args, **kwargs)
        processes.append(proc)
        return proc

    class CancellingVAD(SilentVAD):
        def score_frames(self, frames):
            for frame in frames:
                jobs.cancelled = True
                yield frame, 0.0

    monkeypatch.setattr(stream_vad.subprocess, "Popen", track_process)
    monkeypatch.setattr(stream_vad, "get_sequence_vad_engine", CancellingVAD)
    pipeline.run_stt_job(
        job_id="cancel", input_paths=str(source), jobs=jobs, model=FakeModel(), log=FakeLog(),
        sample_rate=16000, chunk_sec=20,
    )
    assert jobs.events[-1][0] == "cancelled"
    assert processes and all(proc.poll() is not None for proc in processes)
