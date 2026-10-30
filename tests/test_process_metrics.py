import io
import logging
import time

from core.process_metrics import (
    ProcessResourceSampler,
    ProcessSnapshot,
    format_process_resources,
    process_resource_values,
)


def snapshot(
    *,
    monotonic: float,
    rss: int,
    swap: int | None,
    cpu_user: float,
    cpu_system: float,
    host_cpu_total: float = 100.0,
    host_cpu_busy: float = 60.0,
    host_cpu_iowait: float | None = 10.0,
    host_cpu_steal: float | None = 0.0,
    host_logical_cpus: int | None = 4,
) -> ProcessSnapshot:
    return ProcessSnapshot(
        monotonic=monotonic,
        rss_bytes=rss,
        swap_bytes=swap,
        cpu_user_seconds=cpu_user,
        cpu_system_seconds=cpu_system,
        threads=8,
        host_available_bytes=8 * 1_048_576,
        host_logical_cpus=host_logical_cpus,
        host_cpu_total_seconds=host_cpu_total,
        host_cpu_busy_seconds=host_cpu_busy,
        host_cpu_iowait_seconds=host_cpu_iowait,
        host_cpu_steal_seconds=host_cpu_steal,
    )


def test_process_resource_values_derive_average_cpu_cores_and_memory_deltas() -> None:
    previous = snapshot(
        monotonic=10.0,
        rss=100 * 1_048_576,
        swap=20 * 1_048_576,
        cpu_user=15.0,
        cpu_system=5.0,
    )
    current = snapshot(
        monotonic=12.0,
        rss=130 * 1_048_576,
        swap=24 * 1_048_576,
        cpu_user=20.0,
        cpu_system=8.0,
    )

    values = process_resource_values(previous, current)

    assert values["process_rss_mib"] == 130
    assert values["process_rss_delta_mib"] == 30
    assert values["process_swap_mib"] == 24
    assert values["process_swap_delta_mib"] == 4
    assert values["process_cpu_delta_s"] == 8
    assert values["process_resource_window_s"] == 2
    assert values["process_cpu_cores_avg"] == 4


def test_process_resource_format_marks_unsupported_swap_as_unavailable() -> None:
    previous = snapshot(
        monotonic=3.0,
        rss=200 * 1_048_576,
        swap=None,
        cpu_user=2.0,
        cpu_system=1.0,
    )
    current = snapshot(
        monotonic=4.0,
        rss=210 * 1_048_576,
        swap=None,
        cpu_user=2.5,
        cpu_system=1.5,
    )

    formatted = format_process_resources(previous, current)

    assert "process_swap_mib=unavailable" in formatted
    assert "process_swap_delta_mib=unavailable" in formatted
    assert "process_cpu_cores_avg=1.00" in formatted
    assert "process_resource_window_s=1.00" in formatted


def test_host_cpu_window_matches_the_same_process_snapshot_interval() -> None:
    previous = snapshot(
        monotonic=10.0,
        rss=100,
        swap=None,
        cpu_user=5.0,
        cpu_system=2.0,
        host_cpu_total=100.0,
        host_cpu_busy=40.0,
    )
    current = snapshot(
        monotonic=12.0,
        rss=100,
        swap=None,
        cpu_user=7.0,
        cpu_system=3.0,
        host_cpu_total=180.0,
        host_cpu_busy=80.0,
    )

    values = process_resource_values(previous, current)

    assert values["host_cpu_pct_avg"] == 50
    assert values["host_busy_cores_equiv_avg"] == 2
    formatted = format_process_resources(previous, current)
    assert "host_cpu_pct_avg=50.00" in formatted
    assert "host_busy_cores_equiv_avg=2.00" in formatted


def test_host_cpu_reports_iowait_and_vm_steal_separately() -> None:
    previous = snapshot(
        monotonic=10.0, rss=100, swap=None, cpu_user=5.0, cpu_system=2.0,
        host_cpu_total=100.0, host_cpu_busy=40.0,
        host_cpu_iowait=10.0, host_cpu_steal=1.0,
    )
    current = snapshot(
        monotonic=12.0, rss=100, swap=None, cpu_user=7.0, cpu_system=3.0,
        host_cpu_total=180.0, host_cpu_busy=80.0,
        host_cpu_iowait=14.0, host_cpu_steal=5.0,
    )

    values = process_resource_values(previous, current)

    assert values["host_cpu_iowait_pct_avg"] == 5
    assert values["host_cpu_steal_pct_avg"] == 5
    assert "host_cpu_steal_pct_avg=5.00" in format_process_resources(previous, current)


def test_missing_cpu_count_and_zero_window_stay_unavailable() -> None:
    host_previous = snapshot(
        monotonic=10.0, rss=100, swap=None, cpu_user=2.0, cpu_system=1.0,
        host_cpu_total=100.0, host_cpu_busy=50.0, host_logical_cpus=None,
    )
    host_current = snapshot(
        monotonic=12.0, rss=100, swap=None, cpu_user=3.0, cpu_system=1.0,
        host_cpu_total=180.0, host_cpu_busy=90.0, host_logical_cpus=None,
    )
    host_values = process_resource_values(host_previous, host_current)
    assert host_values["host_cpu_pct_avg"] == 50
    assert host_values["host_busy_cores_equiv_avg"] is None

    previous = snapshot(
        monotonic=4.0, rss=100, swap=None, cpu_user=2.0, cpu_system=1.0,
    )
    current = snapshot(
        monotonic=4.0, rss=100, swap=None, cpu_user=3.0, cpu_system=1.0,
    )
    assert process_resource_values(previous, current)["process_cpu_cores_avg"] is None
    assert "process_cpu_cores_avg=unavailable" in format_process_resources(previous, current)

    assert process_resource_values(previous, current)["host_cpu_pct_avg"] is None
    assert "host_cpu_pct_avg=unavailable" in format_process_resources(previous, current)


def test_unreported_host_cpu_modes_are_unavailable() -> None:
    previous = snapshot(
        monotonic=10.0, rss=100, swap=None, cpu_user=5.0, cpu_system=2.0,
        host_cpu_total=100.0, host_cpu_busy=60.0,
        host_cpu_iowait=None, host_cpu_steal=None,
    )
    current = snapshot(
        monotonic=12.0, rss=100, swap=None, cpu_user=7.0, cpu_system=3.0,
        host_cpu_total=180.0, host_cpu_busy=100.0,
        host_cpu_iowait=None, host_cpu_steal=None,
    )

    values = process_resource_values(previous, current)

    assert values["host_cpu_iowait_pct_avg"] is None
    assert values["host_cpu_steal_pct_avg"] is None


def test_debug_sampler_logs_host_cpu_alongside_process_cpu() -> None:
    output = io.StringIO()
    logger = logging.Logger("process-metrics-test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(output))
    sampler = ProcessResourceSampler(logger, interval_seconds=0.1)

    sampler.start()
    time.sleep(0.25)
    sampler.stop()

    assert "host_cpu_pct_avg=" in output.getvalue()
    assert "host_busy_cores_equiv_avg=" in output.getvalue()
