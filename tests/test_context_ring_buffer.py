from core.context import SessionContextManager, TextContextRingBuffer


def test_text_context_ring_buffer_capacity_and_tail():
    buf = TextContextRingBuffer(capacity=3)
    buf.append("Line 1")
    buf.append("Line 2")
    buf.append("Line 3")

    tail = buf.get_tail(lines=2)
    assert tail == ["Line 2", "Line 3"]

    buf.append("Line 4")
    tail_all = buf.get_tail(lines=5)
    assert tail_all == ["Line 2", "Line 3", "Line 4"]
    assert len(tail_all) == 3


def test_text_context_ring_buffer_ignores_empty_text():
    buf = TextContextRingBuffer(capacity=5)
    buf.append("   ")
    buf.append("")
    assert buf.get_tail(lines=5) == []


def test_text_context_ring_buffer_ttl_expiration():
    buf = TextContextRingBuffer(capacity=5)
    buf.append("Fresh line")
    assert buf.get_tail(lines=5, max_age_sec=-1.0) == []


def test_session_context_manager_tenant_isolation():
    manager = SessionContextManager(buffer_capacity=5)
    manager.append("session_a", "Session A context")
    manager.append("session_b", "Session B context")

    assert manager.get_tail("session_a", lines=5) == ["Session A context"]
    assert manager.get_tail("session_b", lines=5) == ["Session B context"]
    assert manager.get_tail("unknown_session", lines=5) == []


def test_session_context_manager_latest_tail_recency():
    manager = SessionContextManager(buffer_capacity=5)
    assert manager.get_latest_tail() == []

    manager.append("session_1", "Text from 1")
    manager.append("session_2", "Text from 2")
    assert manager.get_latest_tail() == ["Text from 2"]

    manager.append("session_1", "Second text from 1")
    assert manager.get_latest_tail() == ["Text from 1", "Second text from 1"]
