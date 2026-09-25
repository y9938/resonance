from core.context import SessionContextManager, TextContextRingBuffer


def test_context_fifo_retention() -> None:
    buffer = TextContextRingBuffer(capacity=3)
    for line in ("one", "two", "three", "four"):
        buffer.append(line)

    assert buffer.get_tail(lines=10) == ["two", "three", "four"]


def test_batch_vs_live_routing() -> None:
    manager = SessionContextManager(buffer_capacity=10)

    manager.append_session("batch", "uploaded transcript")
    manager.append_live("live", "spoken command")

    assert manager.get_session_tail("batch") == ["uploaded transcript"]
    assert manager.get_session_tail("live") == ["spoken command"]
    assert manager.get_ipc_tail() == ["spoken command"]


def test_session_isolation() -> None:
    manager = SessionContextManager()
    manager.append_live("session-a", "A")
    manager.append_live("session-b", "B")

    assert manager.get_session_tail("session-a") == ["A"]
    assert manager.get_session_tail("session-b") == ["B"]
    assert manager.get_session_tail("unknown") == []
