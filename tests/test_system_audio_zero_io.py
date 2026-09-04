import asyncio
import subprocess
import tempfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from server import app
from stt.buffer import AudioMemoryBuffer
from stt.stream_vad import vad_segment_array


def test_audio_memory_buffer_append_and_view() -> None:
    buf = AudioMemoryBuffer()
    assert buf.duration_sec == 0.0
    assert len(buf) == 0

    chunk1 = np.ones(4096, dtype=np.float32) * 0.5
    chunk2 = np.ones(4096, dtype=np.float32) * -0.5
    buf.append(chunk1)
    buf.append(chunk2)

    assert len(buf) == 8192
    assert buf.duration_sec == 8192 / 16000

    view = buf.as_ndarray()
    assert view.shape == (8192,)
    assert view.dtype == np.float32
    assert view[0] == 0.5
    assert view[4096] == -0.5


def test_audio_memory_buffer_zero_disk_io(monkeypatch) -> None:
    def fail_on_disk_call(*args, **kwargs):
        pytest.fail("Disk I/O detected: temporary file or disk call made!")

    monkeypatch.setattr(tempfile, "mkdtemp", fail_on_disk_call)
    monkeypatch.setattr(sf, "SoundFile", fail_on_disk_call)

    buf = AudioMemoryBuffer()
    chunk = np.zeros(16000, dtype=np.float32)
    buf.append(chunk)

    view = buf.as_ndarray()
    assert len(view) == 16000


def _load_fixture_16k() -> np.ndarray:
    audio, sr = sf.read("tests/fixtures/ru_audio.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Assumes: Fixture is 48000 Hz, decimate to 16000 Hz without torchaudio/sox dependency
    step = sr // 16000
    return audio[::step]


def test_vad_segment_array_zero_disk_io(monkeypatch) -> None:
    def fail_call(*args, **kwargs):
        pytest.fail("Disk/subprocess call detected in in-RAM VAD!")

    monkeypatch.setattr(subprocess, "Popen", fail_call)
    monkeypatch.setattr(tempfile, "mkdtemp", fail_call)

    audio = _load_fixture_16k()
    utterances = list(vad_segment_array(audio, sample_rate=16000))

    assert len(utterances) >= 1
    assert utterances[0].start_sample >= 0
    assert len(utterances[0].pcm) > 0


def test_run_stt_job_in_memory_buffers_zero_disk() -> None:
    from stt.pipeline import run_stt_job
    from tests.test_stt_pipeline import FakeJobs, FakeLog

    sr = 16000
    audio = _load_fixture_16k()

    buf_sys = AudioMemoryBuffer()
    buf_sys.append(audio)

    class MockEchoModel:
        def transcribe(self, chunk, **kwargs):
            return "Transcribed from in-memory buffer"

    jobs = FakeJobs()
    model = MockEchoModel()
    log = FakeLog()

    run_stt_job(
        job_id="job-zero-io-test",
        input_paths={"sys": buf_sys},
        jobs=jobs,
        model=model,
        log=log,
        sample_rate=sr,
        chunk_sec=10,
        diarization=False,
    )

    complete_events = [data for event, data in jobs.events if event == "complete"]
    assert len(complete_events) == 1
    progress_events = [data for event, data in jobs.events if event == "progress"]
    assert len(progress_events) >= 1
    assert "Transcribed from in-memory buffer" in progress_events[0]["segment"]["text"]


@pytest.mark.asyncio
async def test_server_system_audio_lifecycle_pure_in_ram(monkeypatch) -> None:
    def fail_on_disk(*args, **kwargs):
        pytest.fail("Disk write detected in server system audio!")

    monkeypatch.setattr(tempfile, "mkdtemp", fail_on_disk)
    monkeypatch.setattr(sf, "SoundFile", fail_on_disk)

    class DummyCaptureStrategy:
        def __init__(self, include_microphone=False):
            self.active = False
        def start_capture(self):
            self.active = True
        def stop_capture(self):
            self.active = False
        def get_audio_stream(self):
            # Yield 1 chunk of synthetic audio
            yield ("sys", np.zeros(16000, dtype=np.float32))

    monkeypatch.setattr("server.get_system_audio_capture", lambda **kw: DummyCaptureStrategy(**kw))

    client = TestClient(app)
    # Start capture
    resp = client.post("/api/system-audio/start?language=ru&model=gigaam")
    assert resp.status_code == 200
    capture_id = resp.json()["capture_id"]

    await asyncio.sleep(0.05)

    # Stop capture
    stop_resp = client.post(f"/api/system-audio/stop?capture_id={capture_id}")
    assert stop_resp.status_code == 200
    assert "job_id" in stop_resp.json()

def test_decode_media_bytes_to_audio_memory_buffer() -> None:
    import io

    import soundfile as sf

    from stt.buffer import decode_media_bytes

    # Generate a WAV in memory
    sr = 48000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_io = io.BytesIO()
    sf.write(wav_io, tone, sr, format="WAV", subtype="FLOAT")
    encoded_bytes = wav_io.getvalue()

    buf = decode_media_bytes(encoded_bytes, target_sample_rate=16000)
    assert isinstance(buf, AudioMemoryBuffer)
    assert buf.sample_rate == 16000
    # 1 second of 16kHz audio = 16000 samples (+- margin)
    assert abs(len(buf) - 16000) <= 200
    arr = buf.as_ndarray()
    assert arr.dtype == np.float32
