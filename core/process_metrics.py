"""Low-cost, process-wide resource snapshots for DEBUG diagnostics."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class ProcessSnapshot:
    monotonic: float
    rss_bytes: int
    swap_bytes: int | None
    cpu_user_seconds: float
    cpu_system_seconds: float
    threads: int
    host_available_bytes: int | None
    host_logical_cpus: int | None
    host_cpu_total_seconds: float | None
    host_cpu_busy_seconds: float | None
    host_cpu_iowait_seconds: float | None
    host_cpu_steal_seconds: float | None


def process_snapshot() -> ProcessSnapshot | None:
    """Capture resident memory and cumulative CPU for this Python process."""
    try:
        process = psutil.Process(os.getpid())
        with process.oneshot():
            memory = process.memory_info()
            cpu = process.cpu_times()
            sampled_at = time.monotonic()
            threads = process.num_threads()
        try:
            host_available_bytes = psutil.virtual_memory().available
        except (OSError, psutil.Error):
            host_available_bytes = None
        try:
            host_logical_cpus = psutil.cpu_count(logical=True)
        except (OSError, psutil.Error):
            host_logical_cpus = None
        try:
            (
                host_cpu_total_seconds,
                host_cpu_busy_seconds,
                host_cpu_iowait_seconds,
                host_cpu_steal_seconds,
            ) = _host_cpu_seconds()
        except (OSError, psutil.Error):
            host_cpu_total_seconds = None
            host_cpu_busy_seconds = None
            host_cpu_iowait_seconds = None
            host_cpu_steal_seconds = None
        return ProcessSnapshot(
            monotonic=sampled_at,
            rss_bytes=memory.rss,
            swap_bytes=_process_swap_bytes(process.pid),
            cpu_user_seconds=cpu.user,
            cpu_system_seconds=cpu.system,
            threads=threads,
            host_available_bytes=host_available_bytes,
            host_logical_cpus=host_logical_cpus,
            host_cpu_total_seconds=host_cpu_total_seconds,
            host_cpu_busy_seconds=host_cpu_busy_seconds,
            host_cpu_iowait_seconds=host_cpu_iowait_seconds,
            host_cpu_steal_seconds=host_cpu_steal_seconds,
        )
    except (OSError, psutil.Error):
        return None


def _process_swap_bytes(pid: int) -> int | None:
    """Read Linux VmSwap cheaply; avoid memory_full_info()'s address-space scan."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmSwap:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _host_cpu_seconds() -> tuple[float, float, float | None, float | None]:
    """Return host total, non-idle, iowait, and steal CPU seconds.

    psutil's guest and guest_nice counters are already included in user and nice,
    so exclude them from the total to avoid double-counting on Linux. Non-idle
    includes steal time, which is reported separately for virtual-machine runs.
    """
    counters = psutil.cpu_times()._asdict()
    total = sum(
        value for name, value in counters.items()
        if name not in {"guest", "guest_nice"}
    )
    iowait = counters.get("iowait")
    steal = counters.get("steal")
    idle = counters.get("idle", 0.0) + (iowait or 0.0)
    return (
        total,
        max(0.0, total - idle),
        float(iowait) if iowait is not None else None,
        float(steal) if steal is not None else None,
    )


def _format_mib(value: int | None) -> str:
    return f"{value / 1_048_576:.1f}" if value is not None else "unavailable"


def _format_count(value: int | None) -> str:
    return str(value) if value is not None else "unavailable"


def _format_cores(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "unavailable"


def _format_swap_delta(previous: int | None, current: int | None) -> str:
    if previous is None or current is None:
        return "unavailable"
    return f"{(current - previous) / 1_048_576:+.1f}"


def format_process_snapshot(snapshot: ProcessSnapshot) -> str:
    """Format an absolute process resource baseline without a comparison window."""
    return (
        f"process_rss_mib={snapshot.rss_bytes / 1_048_576:.1f} "
        f"process_swap_mib={_format_mib(snapshot.swap_bytes)} "
        f"host_memory_available_mib={_format_mib(snapshot.host_available_bytes)} "
        f"host_logical_cpus={_format_count(snapshot.host_logical_cpus)} "
        f"process_threads={snapshot.threads}"
    )


def format_process_resources(
    previous: ProcessSnapshot,
    current: ProcessSnapshot,
) -> str:
    """Format current RSS and process CPU since the prior snapshot."""
    values = process_resource_values(previous, current)
    return " ".join(
        (
            f"process_resource_window_s={values['process_resource_window_s']:.2f}",
            f"process_rss_mib={values['process_rss_mib']:.1f}",
            f"process_rss_delta_mib={values['process_rss_delta_mib']:+.1f}",
            f"process_swap_mib={_format_mib(current.swap_bytes)}",
            f"process_swap_delta_mib={_format_swap_delta(previous.swap_bytes, current.swap_bytes)}",
            f"host_memory_available_mib={_format_mib(current.host_available_bytes)}",
            f"host_logical_cpus={_format_count(current.host_logical_cpus)}",
            f"host_cpu_pct_avg={_format_cores(values['host_cpu_pct_avg'])}",
            f"host_busy_cores_equiv_avg={_format_cores(values['host_busy_cores_equiv_avg'])}",
            f"host_cpu_iowait_pct_avg={_format_cores(values['host_cpu_iowait_pct_avg'])}",
            f"host_cpu_steal_pct_avg={_format_cores(values['host_cpu_steal_pct_avg'])}",
            f"process_cpu_delta_s={values['process_cpu_delta_s']:.2f}",
            f"process_cpu_user_delta_s={values['process_cpu_user_delta_s']:.2f}",
            f"process_cpu_system_delta_s={values['process_cpu_system_delta_s']:.2f}",
            f"process_cpu_cores_avg={_format_cores(values['process_cpu_cores_avg'])}",
            f"process_threads={values['process_threads']}",
        )
    )


def process_resource_values(
    previous: ProcessSnapshot,
    current: ProcessSnapshot,
) -> dict[str, float | int | None]:
    """Return process-wide RSS, CPU, thread, and host-memory values."""
    wall_seconds = max(0.0, current.monotonic - previous.monotonic)
    cpu_user_seconds = max(0.0, current.cpu_user_seconds - previous.cpu_user_seconds)
    cpu_system_seconds = max(0.0, current.cpu_system_seconds - previous.cpu_system_seconds)
    cpu_seconds = cpu_user_seconds + cpu_system_seconds
    cpu_cores = cpu_seconds / wall_seconds if wall_seconds > 0 else 0.0
    rss_delta = current.rss_bytes - previous.rss_bytes
    return {
        "process_resource_window_s": wall_seconds,
        "process_rss_mib": current.rss_bytes / 1_048_576,
        "process_rss_delta_mib": rss_delta / 1_048_576,
        "process_swap_mib": current.swap_bytes / 1_048_576 if current.swap_bytes is not None else None,
        "process_swap_delta_mib": (
            (current.swap_bytes - previous.swap_bytes) / 1_048_576
            if current.swap_bytes is not None and previous.swap_bytes is not None else None
        ),
        "host_memory_available_mib": (
            current.host_available_bytes / 1_048_576
            if current.host_available_bytes is not None else None
        ),
        "host_cpu_pct_avg": _host_cpu_pct(previous, current),
        "host_busy_cores_equiv_avg": _host_busy_cores(previous, current),
        "host_cpu_iowait_pct_avg": _host_cpu_mode_pct(
            previous, current, "host_cpu_iowait_seconds",
        ),
        "host_cpu_steal_pct_avg": _host_cpu_mode_pct(
            previous, current, "host_cpu_steal_seconds",
        ),
        "process_cpu_delta_s": cpu_seconds,
        "process_cpu_user_delta_s": cpu_user_seconds,
        "process_cpu_system_delta_s": cpu_system_seconds,
        "process_cpu_cores_avg": cpu_cores if wall_seconds > 0 else None,
        "process_threads": current.threads,
    }


def _host_cpu_pct(previous: ProcessSnapshot, current: ProcessSnapshot) -> float | None:
    if (
        previous.host_cpu_total_seconds is None
        or current.host_cpu_total_seconds is None
        or previous.host_cpu_busy_seconds is None
        or current.host_cpu_busy_seconds is None
    ):
        return None
    total_delta = current.host_cpu_total_seconds - previous.host_cpu_total_seconds
    busy_delta = current.host_cpu_busy_seconds - previous.host_cpu_busy_seconds
    if total_delta <= 0:
        return None
    return min(100.0, max(0.0, busy_delta / total_delta * 100.0))


def _host_busy_cores(previous: ProcessSnapshot, current: ProcessSnapshot) -> float | None:
    host_cpu_pct = _host_cpu_pct(previous, current)
    if host_cpu_pct is None or current.host_logical_cpus is None:
        return None
    return host_cpu_pct * current.host_logical_cpus / 100.0


def _host_cpu_mode_pct(
    previous: ProcessSnapshot,
    current: ProcessSnapshot,
    field: str,
) -> float | None:
    previous_mode = getattr(previous, field)
    current_mode = getattr(current, field)
    previous_total = previous.host_cpu_total_seconds
    current_total = current.host_cpu_total_seconds
    if (
        previous_mode is None
        or current_mode is None
        or previous_total is None
        or current_total is None
    ):
        return None
    total_delta = current_total - previous_total
    if total_delta <= 0:
        return None
    mode_delta = current_mode - previous_mode
    return min(100.0, max(0.0, mode_delta / total_delta * 100.0))


class ProcessResourceSampler:
    """Periodically log process-wide resource samples while the server is ready."""

    def __init__(self, logger: logging.Logger, interval_seconds: float = 30.0) -> None:
        self._logger = logger
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        baseline = process_snapshot()
        if baseline is None:
            self._logger.debug("Process resource sampling unavailable")
            return
        self._logger.debug(
            "Process resources: phase=server_ready %s", format_process_snapshot(baseline),
        )
        self._thread = threading.Thread(
            target=self._sample,
            name="process-resource-sampler", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample(self) -> None:
        baseline = process_snapshot()
        if baseline is None:
            self._logger.debug("Process resource sampling unavailable")
            return
        previous = baseline
        peak_rss_bytes = baseline.rss_bytes
        peak_swap_bytes = baseline.swap_bytes
        while not self._stop_event.wait(self._interval_seconds):
            current = process_snapshot()
            if current is None:
                continue
            peak_rss_bytes = max(peak_rss_bytes, current.rss_bytes)
            if current.swap_bytes is not None:
                peak_swap_bytes = max(peak_swap_bytes or 0, current.swap_bytes)
            self._logger.debug(
                "Process resources: phase=sample uptime=%.1fs sample_window=%.1fs "
                "sampled_peak_process_rss_mib=%.1f sampled_peak_process_swap_mib=%s "
                "%s",
                current.monotonic - baseline.monotonic,
                current.monotonic - previous.monotonic,
                peak_rss_bytes / 1_048_576,
                _format_mib(peak_swap_bytes),
                format_process_resources(previous, current),
            )
            previous = current
