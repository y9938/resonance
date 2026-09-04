from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextEntry:
    text: str
    start_sec: float
    end_sec: float
    created_at: float


class TextContextRingBuffer:
    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self._lock = threading.Lock()
        self._entries: list[ContextEntry] = []

    def append(self, text: str, start_sec: float = 0.0, end_sec: float = 0.0) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        entry = ContextEntry(
            text=cleaned,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            created_at=time.time(),
        )
        with self._lock:
            if len(self._entries) >= self.capacity:
                self._entries.pop(0)
            self._entries.append(entry)

    def get_tail(self, lines: int = 5, max_age_sec: float = 300.0) -> list[str]:
        now = time.time()
        with self._lock:
            recent = [
                e.text
                for e in self._entries[-lines:]
                if (now - e.created_at) <= max_age_sec
            ]
        return recent

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class SessionContextManager:
    """Session-isolated context store preventing cross-tenant information disclosure."""

    def __init__(self, buffer_capacity: int = 100) -> None:
        self._capacity = buffer_capacity
        self._lock = threading.Lock()
        self._sessions: dict[str, TextContextRingBuffer] = {}

    def append(self, session_id: str, text: str, start_sec: float = 0.0, end_sec: float = 0.0) -> None:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = TextContextRingBuffer(capacity=self._capacity)
            buf = self._sessions[session_id]
            self._sessions[session_id] = self._sessions.pop(session_id)
        buf.append(text, start_sec, end_sec)

    def get_tail(self, session_id: str, lines: int = 5, max_age_sec: float = 300.0) -> list[str]:
        with self._lock:
            buf = self._sessions.get(session_id)
        if buf is None:
            return []
        return buf.get_tail(lines=lines, max_age_sec=max_age_sec)

    def get_latest_tail(self, lines: int = 5, max_age_sec: float = 300.0) -> list[str]:
        with self._lock:
            if not self._sessions:
                return []
            latest_session = next(reversed(self._sessions.values()))
            return latest_session.get_tail(lines=lines, max_age_sec=max_age_sec)

    def clear(self, session_id: str) -> None:
        with self._lock:
            buf = self._sessions.get(session_id)
        if buf is not None:
            buf.clear()


session_context_manager = SessionContextManager(buffer_capacity=100)
