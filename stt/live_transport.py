from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stt.live import LiveSTTSession


class LiveSessionFinishing(Exception):
    """Raised when a chunk arrives after terminal ownership was claimed."""


class LiveSequenceAhead(Exception):
    """Raised when a client skips an uncommitted live-audio sequence."""

    def __init__(self, expected_sequence: int) -> None:
        self.expected_sequence = expected_sequence


class LiveSessionHandle:
    """Owns browser live-session lifecycle while keeping STT processing separate."""

    def __init__(self, session: LiveSTTSession, idle_timeout_sec: float) -> None:
        self.session = session
        self._phase = "ACTIVE"
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._transport_lock = asyncio.Lock()
        self._expected_sequence = 1
        self._drained = asyncio.Event()
        self._drained.set()
        self._idle_timeout_sec = idle_timeout_sec
        self._loop = asyncio.get_running_loop()
        self._deadline = (
            self._loop.time() + idle_timeout_sec if idle_timeout_sec > 0 else None
        )
        self._expiry_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def claim_chunk(self) -> AsyncIterator[LiveSTTSession]:
        async with self._lock:
            if self._phase != "ACTIVE":
                raise LiveSessionFinishing
            self._in_flight += 1
            self._drained.clear()
        try:
            yield self.session
        finally:
            async with self._lock:
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._drained.set()

    async def claim_finish(self) -> LiveSTTSession | None:
        async with self._lock:
            if self._phase != "ACTIVE":
                return None
            self._phase = "FINISHING"
            return self.session

    async def wait_for_drain(self) -> None:
        await self._drained.wait()

    @asynccontextmanager
    async def serialise_sequence(self, sequence: int) -> AsyncIterator[int | None]:
        async with self._transport_lock:
            if sequence < self._expected_sequence:
                yield self._expected_sequence - 1
                return
            if sequence > self._expected_sequence:
                raise LiveSequenceAhead(self._expected_sequence)
            yield None

    def commit_sequence(self, sequence: int) -> None:
        if sequence != self._expected_sequence:
            raise RuntimeError("Live sequence commit is out of order")
        self._expected_sequence += 1

    async def touch_activity(self) -> None:
        if self._deadline is None:
            return
        async with self._lock:
            if self._phase == "ACTIVE":
                self._deadline = self._loop.time() + self._idle_timeout_sec

    async def claim_timeout_if_expired(
        self, now: float
    ) -> tuple[LiveSTTSession | None, float | None]:
        async with self._lock:
            if self._phase != "ACTIVE" or self._deadline is None:
                return None, None
            if self._deadline > now:
                return None, self._deadline
            self._phase = "FINISHING"
            return self.session, None

    def start_expiry_task(
        self,
        on_expiry: Callable[[LiveSTTSession], Awaitable[None]],
    ) -> None:
        if self._deadline is not None:
            self._expiry_task = asyncio.create_task(self._run_expiry(on_expiry))

    def cancel_expiry_task(self) -> None:
        task = self._expiry_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _run_expiry(
        self,
        on_expiry: Callable[[LiveSTTSession], Awaitable[None]],
    ) -> None:
        try:
            while True:
                session, deadline = await self.claim_timeout_if_expired(self._loop.time())
                if session is not None:
                    await on_expiry(session)
                    return
                if deadline is None:
                    return
                await asyncio.sleep(max(0.0, deadline - self._loop.time()))
        except asyncio.CancelledError:
            return
        finally:
            if self._expiry_task is asyncio.current_task():
                self._expiry_task = None
