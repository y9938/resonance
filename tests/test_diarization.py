import numpy as np
import pytest

from stt.diarization import SpeakerInterval, diarize_audio, get_sherpa_cache_dir, match_speaker_tag


def test_match_speaker_tag_empty():
    assert match_speaker_tag(0.0, 5.0, []) == ""


def test_match_speaker_tag_dominant_overlap():
    intervals = [
        SpeakerInterval(start_sec=0.0, end_sec=3.0, speaker_id=0),
        SpeakerInterval(start_sec=3.0, end_sec=6.0, speaker_id=1),
    ]

    tag0 = match_speaker_tag(0.0, 2.5, intervals)
    assert tag0 == "[Speaker 1]: "

    tag1 = match_speaker_tag(2.8, 5.5, intervals)
    assert tag1 == "[Speaker 2]: "


def test_match_speaker_tag_below_threshold():
    intervals = [
        SpeakerInterval(start_sec=10.0, end_sec=12.0, speaker_id=0),
    ]
    tag = match_speaker_tag(0.0, 5.0, intervals)
    assert tag == ""


def test_get_sherpa_cache_dir_precedence(monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("SHERPA_HOME", "/custom/sherpa")
    assert get_sherpa_cache_dir() == Path("/custom/sherpa")

    monkeypatch.delenv("SHERPA_HOME")
    monkeypatch.setenv("RESONANCE_CACHE_DIR", "/custom/resonance")
    assert get_sherpa_cache_dir() == Path("/custom/resonance")

    monkeypatch.delenv("RESONANCE_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/xdg")
    assert get_sherpa_cache_dir() == Path("/custom/xdg/resonance/sherpa")


def test_diarize_audio_aborts_on_cancel(monkeypatch):
    class FakeDiarizer:
        def process(self, audio, callback=None):
            if callback:
                callback(1, 10)
            return []

    monkeypatch.setattr("stt.diarization.get_diarizer", lambda: FakeDiarizer())

    audio = np.zeros(16000 * 2, dtype=np.float32)
    with pytest.raises(RuntimeError, match="STT job cancelled"):
        diarize_audio(audio, cancel_check=lambda: True)
