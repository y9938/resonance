import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from stt.system_audio import MacOSSharedMemoryStrategy

# Native wire layout, independent of the Python reader's implementation.
HEADER_FMT = '<4sIIIIIQIIiII12x'
COMMAND_OFFSET = 32
STATUS_OFFSET = 36
ERROR_OFFSET = 40
REQUEST_OFFSET = 44
RESPONSE_OFFSET = 48


@pytest.fixture
def capture():
    # Exercise the Python protocol with RAM and real thread wakeups; no user files or devices.
    strategy = MacOSSharedMemoryStrategy.__new__(MacOSSharedMemoryStrategy)
    strategy.include_microphone = False
    strategy._stopped = threading.Event()
    strategy._request_id = None
    strategy.shm = bytearray(64 + 4096 * 16 * 2 * 4)
    strategy.data_sem = threading.Semaphore(0)
    strategy.cmd_sem = threading.Semaphore(0)
    strategy.read_idx = 0
    struct.pack_into(HEADER_FMT, strategy.shm, 0,
                     b'OSER', 2, 16000, 1, 4096, 16, 0, 0, 0, 0, 0, 0)
    return strategy


def respond(capture, status, *, request=None, error=0):
    if request is None:
        request = struct.unpack_from('<I', capture.shm, REQUEST_OFFSET)[0]
    struct.pack_into('<i', capture.shm, ERROR_OFFSET, error)
    struct.pack_into('<I', capture.shm, STATUS_OFFSET, status)
    struct.pack_into('<I', capture.shm, RESPONSE_OFFSET, request)


@pytest.mark.parametrize('previous_status', [2, 3])
def test_start_waits_for_its_response_instead_of_previous_success_or_failure(capture, monkeypatch, previous_status):
    respond(capture, previous_status, request=0, error=-3801)
    waits = []

    def complete_new_request(_seconds):
        waits.append(True)
        respond(capture, 2)

    monkeypatch.setattr('stt.system_audio.time.sleep', complete_new_request)
    capture.start_capture()

    assert waits == [True]
    assert not capture._stopped.is_set()


def test_start_timeout_invalidates_pending_request_and_sends_stop(capture):
    capture._START_TIMEOUT = 0
    with pytest.raises(RuntimeError, match='did not confirm start'):
        capture.start_capture()
    assert struct.unpack_from('<I', capture.shm, COMMAND_OFFSET)[0] == 2
    assert struct.unpack_from('<I', capture.shm, REQUEST_OFFSET)[0] == 2
    assert capture._stopped.is_set()


def test_fresh_native_failure_is_reported_and_pending_start_is_cancelled(capture, monkeypatch):
    monkeypatch.setattr('stt.system_audio.time.sleep', lambda _: respond(capture, 3, error=1003))
    with pytest.raises(RuntimeError, match='code 1003'):
        capture.start_capture()
    assert struct.unpack_from('<I', capture.shm, COMMAND_OFFSET)[0] == 2
    assert capture._stopped.is_set()


def test_stop_wakes_reader_and_finishes_even_if_native_has_cleared_command(capture, monkeypatch):
    monkeypatch.setattr('stt.system_audio.time.sleep', lambda _: respond(capture, 2))
    capture.start_capture()
    reader_entered = threading.Event()
    reader_resumes = threading.Event()
    semaphore = capture.data_sem

    class ReaderSemaphore:
        def acquire(self):
            reader_entered.set()
            semaphore.acquire()
            assert reader_resumes.wait(1)

        def release(self):
            semaphore.release()

    capture.data_sem = ReaderSemaphore()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(next, capture.get_audio_stream(), None)
        try:
            assert reader_entered.wait(1)
            capture.stop_capture()
            # Model the native ACK winning the race with the waking Python reader.
            struct.pack_into('<I', capture.shm, COMMAND_OFFSET, 0)
            respond(capture, 0)
            reader_resumes.set()
            assert result.result(timeout=1) is None
        finally:
            reader_resumes.set()
            capture.stop_capture()


def test_wakeup_without_published_audio_does_not_yield_a_slot(capture):
    respond(capture, 2)
    strategy_sem = capture.data_sem
    calls = []

    class DataArrivesAfterSpuriousWakeup:
        def acquire(self):
            calls.append(True)
            if len(calls) == 2:
                np.ndarray((4096,), np.float32, capture.shm, 64).fill(0.5)
                struct.pack_into('<Q', capture.shm, 24, 1)

        def release(self):
            strategy_sem.release()

    capture.data_sem = DataArrivesAfterSpuriousWakeup()
    source, chunk = next(capture.get_audio_stream())
    assert len(calls) == 2
    assert source == 'sys'
    assert np.all(chunk == 0.5)
    # Reusing a shared slot must not change audio already handed to inference.
    np.ndarray((4096,), np.float32, capture.shm, 64).fill(0.9)
    assert np.all(chunk == 0.5)


def test_unexpected_native_stop_error_reaches_audio_consumer(capture):
    respond(capture, 3, error=1003)
    capture.data_sem.release()
    with pytest.raises(RuntimeError, match='code 1003'):
        next(capture.get_audio_stream())


def test_cleanup_of_old_capture_does_not_stop_a_newer_capture(capture, monkeypatch):
    monkeypatch.setattr('stt.system_audio.time.sleep', lambda _: respond(capture, 2))
    capture.start_capture()
    # A newer strategy published START while the old supervisor was cleaning up.
    struct.pack_into('<I', capture.shm, REQUEST_OFFSET, 2)
    capture.stop_capture()
    assert struct.unpack_from('<I', capture.shm, REQUEST_OFFSET)[0] == 2
    assert struct.unpack_from('<I', capture.shm, COMMAND_OFFSET)[0] == 1
    assert capture._stopped.is_set()


@pytest.mark.parametrize('mode', ['system', 'dual', 'quiet_system'])
def test_capture_sources_reach_live_transcripts_without_mixing(capture, monkeypatch, mode):
    import server
    from stt.live import LiveSTTSession

    channels = 1 if mode == 'system' else 2
    system = np.full(4096, 0.25 if mode != 'quiet_system' else 0, dtype=np.float32)
    microphone = np.full(4096, -0.75, dtype=np.float32)
    slot = system if channels == 1 else np.column_stack((system, microphone)).ravel()
    struct.pack_into(HEADER_FMT, capture.shm, 0,
                     b'OSER', 2, 16000, channels, 4096, 16, 1, 0, 2, 0, 0, 0)
    np.ndarray(slot.shape, np.float32, capture.shm, 64)[:] = slot
    capture.data_sem.release()
    semaphore = capture.data_sem

    def acquire():
        assert semaphore.acquire(timeout=0.2), 'Capture did not deliver the published source'

    capture.data_sem = SimpleNamespace(acquire=acquire, release=semaphore.release)
    # Recognize signal identity deterministically; exercise the real reader, forwarding and tagging.
    def transcribe(pcm):
        signal = pcm[pcm != 0]
        assert len(signal)
        assert np.all(signal == signal[0]), 'Source samples were mixed'
        return str(float(signal[0]))

    model = MagicMock(transcribe=MagicMock(side_effect=transcribe))
    vad = SimpleNamespace(process_frame=lambda _state, pcm: float(np.any(pcm)))
    monkeypatch.setattr('stt.live.get_shared_vad_engine', lambda: vad)
    session = LiveSTTSession('capture-sources', 'capture-sources', model, MagicMock(),
                             source='sys', dual_stream=channels == 2)
    reader = capture.get_audio_stream()
    engine = SimpleNamespace(get_audio_stream=lambda: islice(reader, channels))
    try:
        error = server._system_capture_live_loop(engine, session)
        assert str(error) == 'System audio capture producer exited unexpectedly'
        segments = session.flush()
    finally:
        reader.close()
        capture.stop_capture()

    expected = {'sys': '0.25'} if channels == 1 else {'sys': '0.25', 'mic': '-0.75'}
    if mode == 'quiet_system':
        expected = {'mic': '-0.75'}
    assert len(segments) == len(expected)
    assert {segment['source'] for segment in segments} == expected.keys()
    for segment in segments:
        source = segment['source']
        prefix = f'[SOURCE:{source.upper()}]: ' if channels == 2 else ''
        assert segment['text'] == prefix + expected[source]
    assert model.transcribe.call_count == len(expected)
