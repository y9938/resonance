import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.logging import prepare_log_file
from scripts import diagnostics


def test_export_logs_without_env_or_unrelated_files(tmp_path):
    base = tmp_path / "logs/server.log"
    selected = prepare_log_file(base)
    selected.write_text("session details\n")
    (base.parent / "unrelated.log").write_text("not ours")
    (tmp_path / ".env").write_text("API_KEY=private-test-value\n")
    output = diagnostics.export_diagnostics(tmp_path / "support.zip", selected)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("logs/" + selected.name) == b"session details\n"
        assert not any("unrelated" in name or ".env" in name for name in archive.namelist())
        details = archive.read("diagnostics.txt").decode()
        assert "private-test-value" not in details
        assert "logs/" + selected.name in details
        assert "diagnostics.json" not in archive.namelist()
        assert "summary.txt" not in archive.namelist()


def test_export_only_selected_session_and_reports_truncation(tmp_path, monkeypatch):
    base = tmp_path / "logs/server.log"
    selected = prepare_log_file(base)
    selected.write_bytes(b"first half-last half")
    prepare_log_file(base).write_text("newer session")
    monkeypatch.setattr(diagnostics, "MAX_FILE_BYTES", 9)
    output = diagnostics.export_diagnostics(tmp_path / "support.zip", selected)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("logs/" + selected.name) == b"last half"
        assert len([name for name in archive.namelist() if name.startswith("logs/")]) == 1
        assert "omitted" in archive.read("diagnostics.txt").decode()


def write_ips(path, body):
    now = datetime.now(timezone.utc)
    body = {"captureTime": now.isoformat(), "procLaunch": (now - timedelta(seconds=2)).isoformat(), **body}
    path.write_text(json.dumps({"bug_type": "309"}) + "\n" + json.dumps(body))


@pytest.mark.parametrize("legacy", [True, False])
def test_macos_export_collects_related_reports_and_records_missing_system_logs(tmp_path, monkeypatch, legacy):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    log = prepare_log_file(root / "logs/server.log")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (f"{started} | INFO | resonance.server | pid=1234 | starting\n" if legacy else
              f"# Python session: pid=1234; started={started}; revision=old-build\n")
    log.write_text(header)
    home = tmp_path / "home"
    reports = home / "Library/Logs/DiagnosticReports"
    reports.mkdir(parents=True)
    write_ips(reports / "Resonance.ips", {"pid": 1234, "bundleInfo": {"CFBundleIdentifier": "com.resonance.app"}})
    write_ips(reports / "Python.ips", {"pid": 1234, "procPath": str(Path(sys.executable).resolve()),
                                      "captureTime": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f %z")})
    write_ips(reports / "OtherPython.ips", {"pid": 5678, "procPath": str(Path(sys.executable).resolve())})
    write_ips(reports / "ReusedPid.ips", {"pid": 1234, "procPath": str(Path(sys.executable).resolve()),
                                         "captureTime": "2020-01-01T00:00:00+00:00"})
    write_ips(reports / "PreviousResonance.ips", {"pid": 5678, "procName": "Resonance"})
    write_ips(reports / "LaterReusedPid.ips", {"pid": 1234,
              "procLaunch": (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat()})
    write_ips(reports / "MissingTime.ips", {"pid": 1234, "captureTime": None})
    (reports / "Stackshot.ips").write_text('{"bug_type":"288"}\n')
    (reports / "Malformed.ips").write_text("not JSON")
    old = reports / "Resonance-old.ips"
    write_ips(old, {"procName": "Resonance"})
    os.utime(old, (time.time() - 8 * 86400, time.time() - 8 * 86400))
    commands = []

    def unavailable(command):
        commands.append(command)
        return "Unavailable: permission denied\n" if command[0] == "/usr/bin/log" else "version\n"

    monkeypatch.setattr(diagnostics.sys, "platform", "darwin")
    monkeypatch.setattr(diagnostics.platform, "platform", lambda: "test macOS")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(diagnostics, "command_output", unavailable)
    output = diagnostics.export_diagnostics(tmp_path / "support.zip", log)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "crashes/0/Resonance.ips" in names
        assert "crashes/0/Python.ips" in names
        assert not any("OtherPython" in name or "ReusedPid" in name or "PreviousResonance" in name or "-old" in name for name in names)
        assert "permission denied" in archive.read("system.log").decode()
        assert "Recent reports not checked" in archive.read("diagnostics.txt").decode()
        assert "crash or launch time is missing" in archive.read("diagnostics.txt").decode()
    query = next(command for command in commands if command[0] == "/usr/bin/log")
    predicate = query[query.index("--predicate") + 1]
    assert "processIdentifier IN {1234}" in predicate
    assert "logType == error OR logType == fault" in predicate
    assert 'subsystem == "com.apple.TCC"' in predicate
    assert "type == logEvent" in predicate
    assert "processImagePath" not in predicate
    assert "--last" not in query
    start = float(query[query.index("--start") + 1].removeprefix("@"))
    assert datetime.fromtimestamp(start, timezone.utc).strftime("%Y%m%dT%H%M%S.") in log.name
    assert start == int(start)


@pytest.mark.parametrize("name", ["Resonance", "Python"])
def test_text_crash_reports_match_actual_process_times_not_filename_or_mtime(tmp_path, name):
    now = datetime.now(timezone.utc)
    report = tmp_path / "renamed.crash"

    def write(launch, captured):
        report.write_text(f"Process: {name} [1234]\n"
                          f"Date/Time: {captured.strftime('%Y-%m-%d %H:%M:%S.%f %z')}\n"
                          f"Launch Time: {launch.strftime('%Y-%m-%d %H:%M:%S.%f %z')}\n")

    pids = {1234: now.timestamp()}
    start, end = now.timestamp() - 10, now.timestamp() + 20
    write(now - timedelta(seconds=2), now)
    assert diagnostics.related_crash_report(report, pids, start, end)
    assert not diagnostics.related_crash_report(report, {5678: now.timestamp()}, start, end)
    # A different process can reuse the PID while the export window still includes it.
    write(now + timedelta(seconds=3), now + timedelta(seconds=4))
    assert not diagnostics.related_crash_report(report, pids, start, end)
    write(now - timedelta(days=1), now - timedelta(hours=1))
    assert not diagnostics.related_crash_report(report, pids, start, end)


def test_crash_budget_keeps_complete_reports_and_explains_omitted_or_unchecked_reports(tmp_path, monkeypatch):
    log = prepare_log_file(tmp_path / "logs/server.log")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.write_text(f"# Python session: pid=1234; started={started}; revision=test\n")
    reports = tmp_path / "Library/Logs/DiagnosticReports"
    reports.mkdir(parents=True)
    first = reports / "Z-full.ips"
    write_ips(first, {"pid": 1234, "padding": "x" * 350})
    second = reports / "A-budget.ips"
    write_ips(second, {"pid": 1234, "padding": "x" * 350})
    oversized = reports / "Unverified.ips"
    write_ips(oversized, {"pid": 9999, "padding": "x" * 1000})
    monkeypatch.setattr(diagnostics.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(diagnostics, "command_output", lambda command: "test macOS\n")
    monkeypatch.setattr(diagnostics, "MAX_FILE_BYTES", 100)
    monkeypatch.setattr(diagnostics, "MAX_CRASH_BYTES", 800)
    output = diagnostics.export_diagnostics(tmp_path / "support.zip", log)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("crashes/0/Z-full.ips") == first.read_bytes()
        assert not any("A-budget" in name or "Unverified" in name for name in archive.namelist())
        notes = archive.read("diagnostics.txt").decode()
        assert "A-budget.ips: matching crash report omitted" in notes
        assert "Recent reports not checked: 1; larger than 800 bytes" in notes
        assert "Matching crash reports: 2; included: 1." in notes
        assert "Unverified.ips" not in notes


def test_failed_export_preserves_destination_and_removes_temporary_zip(tmp_path, monkeypatch):
    base = tmp_path / "logs/server.log"
    log = prepare_log_file(base)
    destination = tmp_path / "support.zip"
    destination.write_bytes(b"previous archive")

    original = zipfile.ZipFile.writestr

    def fail(archive, name, *args, **kwargs):
        if name.startswith("logs/"):
            raise OSError("disk full")
        return original(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", fail)
    with pytest.raises(OSError, match="disk full"):
        diagnostics.export_diagnostics(destination, log)
    assert destination.read_bytes() == b"previous archive"
    assert not list(tmp_path.glob(".resonance-diagnostics-*"))


def test_command_timeout_retains_partial_output(monkeypatch):
    def timeout(command, **kwargs):
        kwargs["stdout"].write(b"last useful event\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(diagnostics.subprocess, "run", timeout)
    result = diagnostics.command_output(["/usr/bin/log", "show"])
    assert result.startswith("Incomplete: command timed out")
    assert result.endswith("last useful event\n")


def test_launcher_helper_returns_unique_path_and_cli_exports_it(tmp_path):
    environment = {**os.environ, "RESONANCE_LOG_FILE": str(tmp_path / "logs/server.log"),
                   "RESONANCE_LOG_TO_FILE": "1"}
    command = [sys.executable, "-m", "scripts.diagnostics"]
    prepared = subprocess.run([*command, "--prepare-log"], cwd=diagnostics.ROOT,
                              env=environment, capture_output=True, text=True, check=True)
    log = Path(prepared.stdout.strip())
    assert log.is_file()
    log.write_text("helper and CLI agree\n")
    destination = tmp_path / "support with spaces.zip"
    subprocess.run([*command, "--log-file", str(log), "--output", str(destination)],
                   cwd=diagnostics.ROOT, env=environment, capture_output=True, text=True, check=True)
    with zipfile.ZipFile(destination) as archive:
        assert archive.read("logs/" + log.name) == b"helper and CLI agree\n"
    assert len(list(log.parent.iterdir())) == 1


def test_export_preserves_all_events_without_an_extracted_summary(tmp_path):
    log = prepare_log_file(tmp_path / "logs/server.log")
    events = "".join(f"12:00:00 INFO server: operation {i}\n" for i in range(30))
    content = events + "12:00:01 ERROR server: failed\nTraceback (most recent call last):\n  original traceback\n"
    log.write_text(content)
    output = diagnostics.export_diagnostics(tmp_path / "support.zip", log)
    with zipfile.ZipFile(output) as archive:
        assert archive.read("logs/" + log.name).decode() == content
        details = archive.read("diagnostics.txt").decode()
        assert "operation" not in details
        assert "logs/" + log.name in details
        assert set(archive.namelist()) == {"diagnostics.txt", "logs/" + log.name}


def test_export_keeps_session_identity_when_large_log_is_truncated(tmp_path, monkeypatch):
    log = prepare_log_file(tmp_path / "logs/server.log")
    header = b"# Python session: pid=1234; started=2026-10-08T12:00:00Z; revision=running-build\n"
    log.write_bytes(header + b"old output\n" * 1000 + b"final error\n")
    monkeypatch.setattr(diagnostics, "MAX_FILE_BYTES", 200)
    output = diagnostics.export_diagnostics(tmp_path / "support.zip", log)
    with zipfile.ZipFile(output) as archive:
        data = archive.read("logs/" + log.name)
        assert len(data) == 200
        assert data.startswith(header)
        assert data.endswith(b"final error\n")
        assert "omitted" in archive.read("diagnostics.txt").decode()
