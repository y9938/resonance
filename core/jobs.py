from __future__ import annotations

import copy
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class StreamEvent:
    """Assumes: Event types are mutually exclusive."""

    type: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        payload = {"type": self.type, **self.data}
        return f"data: {json.dumps(payload)}\n\n"


@dataclass
class JobRecord:
    job_id: str
    session_id: str
    job_type: str
    state: str
    created_at: float
    updated_at: float
    progress_current: int
    progress_total: int
    error: str | None
    result: dict[str, Any]
    cancelled: bool
    events: list[dict[str, Any]]
    next_seq: int
    language: str | None = None
    model: str | None = None


class JobRegistry:
    """Thread-safe in-memory background job registry with event logging and status polling."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}

    def create(
        self,
        job_type: str,
        session_id: str,
        initial_result: dict[str, Any] | None = None,
        language: str | None = None,
        model: str | None = None,
    ) -> JobRecord:
        now = time.time()
        rec = JobRecord(
            job_id=secrets.token_urlsafe(24),
            session_id=session_id,
            job_type=job_type,
            state="queued",
            created_at=now,
            updated_at=now,
            progress_current=0,
            progress_total=0,
            error=None,
            result=copy.deepcopy(initial_result or {}),
            cancelled=False,
            events=[],
            next_seq=1,
            language=language,
            model=model,
        )
        with self._lock:
            self._jobs[rec.job_id] = rec
        return copy.deepcopy(rec)

    def exists(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    def belongs_to_session(self, job_id: str, session_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            return bool(rec and rec.session_id == session_id)

    def mark_cancelled(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return False
            rec.cancelled = True
            if rec.state in {"queued", "running"}:
                rec.state = "cancelled"
                rec.updated_at = time.time()
                self._append_event_locked(rec, "cancelled", {})
            return True

    def cancel_all(self) -> None:
        with self._lock:
            for rec in self._jobs.values():
                rec.cancelled = True
                if rec.state in {"queued", "running"}:
                    rec.state = "cancelled"
                    rec.updated_at = time.time()
                    self._append_event_locked(rec, "cancelled", {})

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            return bool(rec and rec.cancelled)

    def update_event(self, job_id: str, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            now = time.time()
            rec.updated_at = now
            if event_type == "start":
                rec.state = "running"
                rec.progress_total = int(data.get("total", 0))
                rec.progress_current = 0
                if rec.job_type == "stt" and "duration" in data:
                    try:
                        rec.result["duration"] = float(data["duration"])
                    except (ValueError, TypeError):
                        pass
            elif event_type == "progress":
                rec.state = "running"
                rec.progress_current = int(data.get("current", rec.progress_current))
                rec.progress_total = int(data.get("total", rec.progress_total))
                if rec.job_type == "stt" and "segment" in data:
                    segs = rec.result.setdefault("segments", [])
                    segs.append(data["segment"])
                if rec.job_type == "tts":
                    rec.result["chunks"] = rec.progress_total
            elif event_type == "complete":
                rec.state = "completed"
                if rec.progress_total > 0:
                    rec.progress_current = rec.progress_total
                if rec.job_type == "stt":
                    # Keep duration available in job summaries without requiring /api/jobs/{id}.
                    if "duration" in data:
                        try:
                            rec.result["duration"] = float(data["duration"])
                        except (ValueError, TypeError):
                            pass
                    elif "duration" not in rec.result and isinstance(rec.result.get("segments"), list):
                        max_end = 0.0
                        for s in rec.result.get("segments") or []:
                            try:
                                end = float(s.get("end", 0.0)) if isinstance(s, dict) else 0.0
                            except (ValueError, TypeError):
                                end = 0.0
                            max_end = max(max_end, end)
                        if max_end > 0:
                            rec.result["duration"] = max_end
                if rec.job_type == "tts":
                    for key in ("download_url", "duration", "chunks", "filename"):
                        if key in data:
                            rec.result[key] = data[key]
            elif event_type == "error":
                rec.state = "failed"
                rec.error = str(data.get("message", "Job failed"))
            elif event_type == "cancelled":
                rec.state = "cancelled"
            self._append_event_locked(rec, event_type, data)

    def get_status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return {
                "job_id": rec.job_id,
                "session_id": rec.session_id,
                "job_type": rec.job_type,
                "state": rec.state,
                "progress_current": rec.progress_current,
                "progress_total": rec.progress_total,
                "error": rec.error,
                "result": copy.deepcopy(rec.result),
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "language": rec.language,
                "model": rec.model,
            }

    def events_after(self, job_id: str, after_seq: int) -> list[dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return []
            return [copy.deepcopy(ev) for ev in rec.events if int(ev.get("seq", 0)) > after_seq]

    def list_for_session(
        self, session_id: str, limit: int, offset: int = 0
    ) -> dict[str, Any]:
        with self._lock:
            rows = [
                rec for rec in self._jobs.values()
                if rec.session_id == session_id
            ]
            rows.sort(key=lambda r: r.updated_at, reverse=True)
            total = len(rows)
            start = max(0, int(offset))
            lim = max(1, int(limit))
            slice_rows = rows[start : start + lim]
            out: list[dict[str, Any]] = []
            for rec in slice_rows:
                duration_out: float | None = None
                duration = rec.result.get("duration")
                if duration is not None:
                    try:
                        duration_out = float(duration)
                    except (ValueError, TypeError):
                        duration_out = None
                if duration_out is None and rec.job_type == "stt" and isinstance(rec.result.get("segments"), list):
                    max_end = 0.0
                    for s in rec.result.get("segments") or []:
                        try:
                            end = float(s.get("end", 0.0)) if isinstance(s, dict) else 0.0
                        except (ValueError, TypeError):
                            end = 0.0
                        max_end = max(max_end, end)
                    if max_end > 0:
                        duration_out = max_end
                item = {
                    "job_id": rec.job_id,
                    "job_type": rec.job_type,
                    "state": rec.state,
                    "progress_current": rec.progress_current,
                    "progress_total": rec.progress_total,
                    "error": rec.error,
                    "duration": duration_out,
                    "created_at": rec.created_at,
                    "updated_at": rec.updated_at,
                }
                if rec.language is not None:
                    item["language"] = rec.language
                if rec.model is not None:
                    item["model"] = rec.model
                for key in ("filename", "batch_id", "batch_index", "batch_total"):
                    if key in rec.result:
                        item[key] = copy.deepcopy(rec.result[key])
                out.append(item)
            next_offset = start + len(out)
            has_more = next_offset < total
            return {
                "jobs": out,
                "has_more": has_more,
                "next_offset": next_offset,
            }

    def _append_event_locked(self, rec: JobRecord, event_type: str, data: dict[str, Any]) -> None:
        payload = {"seq": rec.next_seq, "type": event_type, **copy.deepcopy(data)}
        rec.next_seq += 1
        rec.events.append(payload)
        if len(rec.events) > 5000:
            rec.events = rec.events[-5000:]
