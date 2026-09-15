from unittest.mock import MagicMock

import numpy as np

from core.jobs import JobRegistry
from stt.live import ASRCommitPolicy, CommitDecision, LiveSTTSession
from stt.live_preview import LivePreviewBroker
from stt.stream_vad import _VAD_WINDOW_SAMPLES


class FakeVad:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)

    def process_frame(self, _state, _window) -> float:
        return next(self.probabilities)


def make_session(probabilities: list[float], preview: LivePreviewBroker | None = None):
    model = MagicMock()
    model.transcribe.return_value = "text"
    jobs = JobRegistry()
    job = jobs.create("stt", "session", {"filename": "live"})
    session = LiveSTTSession(job.job_id, "session", model, jobs, preview_broker=preview)
    session._vad_engine = FakeVad(probabilities)
    return session, model, jobs, job.job_id


def feed(session: LiveSTTSession, windows: int, source: str = "mic") -> list[dict]:
    return session.process_pcm_chunk(np.ones(windows * _VAD_WINDOW_SAMPLES, dtype=np.float32), source)


def test_vad_endpoint_does_not_commit_and_resumes_same_contiguous_buffer() -> None:
    session, model, _, _ = make_session([1.0] * 8 + [0.0] * 20 + [1.0] * 8)
    assert feed(session, 36) == []
    assert model.transcribe.call_count == 0
    emitted = session.flush()
    assert len(emitted) == 1
    assert model.transcribe.call_args.args[0].shape[0] == 36 * _VAD_WINDOW_SAMPLES


def test_long_pause_commits_once_and_hard_limit_is_bounded() -> None:
    session, model, _, _ = make_session([1.0] * 8 + [0.0] * 32)
    emitted = feed(session, 40)
    assert len(emitted) == model.transcribe.call_count == 1

    hard_windows = int(20 * 16000 / _VAD_WINDOW_SAMPLES) + 1
    session, model, _, _ = make_session([1.0] * hard_windows)
    emitted = feed(session, hard_windows)
    assert len(emitted) == model.transcribe.call_count == 1


def test_soft_limit_waits_for_natural_boundary_and_cancel_drops_pending() -> None:
    soft_windows = int(12 * 16000 / _VAD_WINDOW_SAMPLES) + 3
    session, model, _, _ = make_session([1.0] * soft_windows + [0.0] * 11)
    assert feed(session, soft_windows) == []
    assert model.transcribe.call_count == 0
    assert len(feed(session, 11)) == 1

    session, model, _, _ = make_session([1.0] * 10)
    feed(session, 10)
    session.cancel()
    assert session.flush() == []
    model.transcribe.assert_not_called()


def test_sources_keep_independent_buffers_and_preview_is_ephemeral() -> None:
    broker = LivePreviewBroker()
    session, _model, jobs, job_id = make_session([1.0] * 128, broker)
    subscription = broker.subscribe(job_id)
    feed(session, 64, "mic")
    feed(session, 64, "sys")
    previews = broker.take(subscription)
    assert {event["source"] for event in previews} == {"mic", "sys"}
    assert jobs.events_after(job_id, 0) == []
    session.flush()
    durable = jobs.get_status(job_id)["result"]["segments"]
    assert {segment["source"] for segment in durable} == {"mic", "sys"}


def test_commit_policy_is_typed_and_flush_is_explicit() -> None:
    policy = ASRCommitPolicy(16000)
    assert policy.decide(samples=12 * 16000, natural_boundary=False, pause_samples=0, flush=False) is CommitDecision.KEEP_BUFFERING
    assert policy.decide(samples=12 * 16000, natural_boundary=True, pause_samples=0, flush=False) is CommitDecision.COMMIT_SOFT_BOUNDARY
    assert policy.decide(samples=0, natural_boundary=False, pause_samples=0, flush=True) is CommitDecision.COMMIT_FLUSH


def test_preview_broker_is_latest_value_not_an_unbounded_fifo() -> None:
    broker = LivePreviewBroker()
    subscription = broker.subscribe("job")
    broker.publish("job", "mic", 1, "P1")
    broker.publish("job", "mic", 1, "P2")
    broker.publish("job", "mic", 1, "P3")
    broker.publish("job", "sys", 2, "S1")
    assert broker.take(subscription) == [
        {"type": "transcript_preview", "source": "mic", "generation": 1, "text": "P3"},
        {"type": "transcript_preview", "source": "sys", "generation": 2, "text": "S1"},
    ]
    broker.publish("job", "mic", 2, "lost on reconnect")
    broker.unsubscribe(subscription)
    reconnect = broker.subscribe("job")
    assert broker.take(reconnect) == []


def test_preview_subscriptions_have_independent_latest_value_cursors() -> None:
    broker = LivePreviewBroker()
    tab_a = broker.subscribe("job")
    tab_b = broker.subscribe("job")

    broker.publish("job", "mic", 1, "P1")
    assert broker.take(tab_a)[0]["text"] == "P1"
    assert broker.take(tab_b)[0]["text"] == "P1"

    broker.publish("job", "mic", 1, "P2")
    broker.unsubscribe(tab_a)
    assert broker.take(tab_b) == [
        {"type": "transcript_preview", "source": "mic", "generation": 1, "text": "P2"}
    ]


def test_old_preview_subscription_cannot_unsubscribe_new_reconnect() -> None:
    broker = LivePreviewBroker()
    old_connection = broker.subscribe("job")
    new_connection = broker.subscribe("job")

    broker.unsubscribe(old_connection)
    broker.publish("job", "mic", 2, "after reconnect")

    assert broker.take(new_connection) == [
        {
            "type": "transcript_preview",
            "source": "mic",
            "generation": 2,
            "text": "after reconnect",
        }
    ]
