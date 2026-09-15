from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LivePreviewSubscription:
    """Opaque identity for one transient SSE consumer."""

    job_id: str
    token: int


class LivePreviewBroker:
    """Ephemeral, non-replayable preview transport. It never touches JobRegistry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
        self._revisions: dict[str, int] = {}
        self._subscriptions: dict[int, tuple[str, int]] = {}
        self._job_tokens: dict[str, set[int]] = {}
        self._next_token = 1

    def subscribe(self, job_id: str) -> LivePreviewSubscription:
        with self._lock:
            token = self._next_token
            self._next_token += 1
            # A reconnect starts fresh: this subscriber intentionally does not
            # receive previews published before it subscribed. Existing
            # subscribers retain their own observation cursors.
            self._subscriptions[token] = (job_id, self._revisions.get(job_id, 0))
            self._job_tokens.setdefault(job_id, set()).add(token)
            return LivePreviewSubscription(job_id, token)

    def unsubscribe(self, subscription: LivePreviewSubscription) -> None:
        with self._lock:
            state = self._subscriptions.get(subscription.token)
            if state is None or state[0] != subscription.job_id:
                return
            del self._subscriptions[subscription.token]
            tokens = self._job_tokens[subscription.job_id]
            tokens.discard(subscription.token)
            if not tokens:
                self._job_tokens.pop(subscription.job_id, None)
                self._latest.pop(subscription.job_id, None)
                self._revisions.pop(subscription.job_id, None)

    def publish(self, job_id: str, source: str, generation: int, text: str) -> None:
        with self._lock:
            if job_id not in self._job_tokens:
                return
            # Preview is a mutable tail: a slow client only needs the most
            # recent value per source, never an unbounded edit history. Each
            # subscriber tracks its own cursor, so one consumer cannot consume
            # or unsubscribe another consumer's latest value.
            revision = self._revisions.get(job_id, 0) + 1
            self._revisions[job_id] = revision
            event = {
                "type": "transcript_preview", "source": source,
                "generation": generation, "text": text,
            }
            self._latest.setdefault(job_id, {})[source] = (revision, event)

    def take(self, subscription: LivePreviewSubscription) -> list[dict[str, Any]]:
        with self._lock:
            state = self._subscriptions.get(subscription.token)
            if state is None or state[0] != subscription.job_id:
                return []
            _, seen_revision = state
            previews = [
                (revision, event)
                for revision, event in self._latest.get(subscription.job_id, {}).values()
                if revision > seen_revision
            ]
            if previews:
                self._subscriptions[subscription.token] = (
                    subscription.job_id,
                    max(revision for revision, _ in previews),
                )
            return [event for _, event in sorted(previews)]

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._latest.pop(job_id, None)
            self._revisions.pop(job_id, None)
