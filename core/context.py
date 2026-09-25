from __future__ import annotations

import threading
from collections import deque

DEFAULT_CONTEXT_CAPACITY: int = 1000


class TextContextRingBuffer:
    """Thread-safe bounded text buffer with FIFO retention."""

    def __init__(self, capacity: int = DEFAULT_CONTEXT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._lock = threading.Lock()
        self._entries: deque[str] = deque(maxlen=capacity)

    def append(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        with self._lock:
            self._entries.append(cleaned)

    def get_tail(self, lines: int = 5) -> list[str]:
        if lines < 1:
            raise ValueError(f"lines must be >= 1, got {lines}")
        with self._lock:
            return list(self._entries)[-lines:]


class SessionContextManager:
    """Stores per-session context and a process-local live transcript feed.

    Session buffers live for the process lifetime and are bounded by FIFO.
    Live commits update the session and IPC feed as one ordered operation.
    """

    def __init__(self, buffer_capacity: int = DEFAULT_CONTEXT_CAPACITY) -> None:
        if buffer_capacity < 1:
            raise ValueError(f"buffer_capacity must be >= 1, got {buffer_capacity}")
        self._capacity = buffer_capacity
        self._lock = threading.Lock()
        self._sessions: dict[str, TextContextRingBuffer] = {}
        self._ipc_feed = TextContextRingBuffer(capacity=buffer_capacity)

    def _append(self, session_id: str, text: str, *, publish_to_ipc: bool) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = TextContextRingBuffer(capacity=self._capacity)
            self._sessions[session_id].append(cleaned)
            if publish_to_ipc:
                self._ipc_feed.append(cleaned)

    def append_session(self, session_id: str, text: str) -> None:
        self._append(session_id, text, publish_to_ipc=False)

    def append_live(self, session_id: str, text: str) -> None:
        """Appends a live conversational audio segment to session buffer and local IPC feed."""
        self._append(session_id, text, publish_to_ipc=True)

    def get_session_tail(self, session_id: str, lines: int = 5) -> list[str]:
        if lines < 1:
            raise ValueError(f"lines must be >= 1, got {lines}")
        with self._lock:
            session = self._sessions.get(session_id)
            return [] if session is None else session.get_tail(lines=lines)

    def get_ipc_tail(self, lines: int = 5) -> list[str]:
        return self._ipc_feed.get_tail(lines=lines)


session_context_manager = SessionContextManager()
