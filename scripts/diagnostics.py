"""Local diagnostic export and shared log preparation for the macOS launcher."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from core.logging import (
    log_files,
    prepare_log_file,
    resolve_log_file,
)

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_CRASH_BYTES = 20 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 15


def command_output(command: list[str]) -> str:
    # System logs may be large; keep command output out of RAM until it is bounded.
    with tempfile.TemporaryFile() as output:
        status = ""
        try:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    timeout=COMMAND_TIMEOUT_SECONDS, check=False)
            if result.returncode:
                status = f"Exit code {result.returncode}\n"
        except subprocess.TimeoutExpired:
            # run() kills and waits for the child; its emitted output is still here.
            status = f"Incomplete: command timed out after {COMMAND_TIMEOUT_SECONDS}s; partial output retained.\n"
        except OSError as error:
            return f"Unavailable: {error}\n"
        size = output.tell()
        output.seek(max(0, size - MAX_FILE_BYTES))
        text = output.read(MAX_FILE_BYTES).decode("utf-8", errors="replace")
    if size > MAX_FILE_BYTES:
        status += "Incomplete: earlier output omitted; size limit reached.\n"
    return status + text


def report_time(value: str) -> float:
    # Apple's text reports use e.g. '2026-10-08 12:30:00.1234 +0300'.
    value = re.sub(r"\s*([+-]\d{2})(\d{2})$", r"\1:\2", value.strip().replace("Z", "+00:00"))
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.timestamp()


def related_crash_report(path: Path, pids: dict[int, float | None], start: float, end: float) -> bool:
    """Match recorded PIDs and their launch times, including shared Python interpreters."""
    if path.suffix == ".crash":
        with path.open(encoding="utf-8") as source:
            fields = dict(re.findall(r"^(Process|Date/Time|Launch Time):[ \t]*(.+)$", source.read(64 * 1024), re.MULTILINE))
        process = re.search(r"\[(\d+)\]$", fields.get("Process", "").strip())
        body = {"pid": int(process[1]) if process else None,
                "captureTime": fields.get("Date/Time"), "procLaunch": fields.get("Launch Time")}
    else:
        with path.open("rb") as source:
            metadata = source.readline(64 * 1024)
            header = json.loads(metadata)
            if not isinstance(header, dict) or str(header.get("bug_type")) != "309":
                return False
            report = source.read(max(0, MAX_CRASH_BYTES + 1 - len(metadata)))
        if len(metadata) + len(report) > MAX_CRASH_BYTES:
            raise ValueError(f"larger than {MAX_CRASH_BYTES} bytes; too large to inspect")
        body = json.loads(report)
        if not isinstance(body, dict):
            raise ValueError("crash report body is not an object")
    pid = body.get("pid")
    if not isinstance(pid, int) or pid not in pids:
        return False
    captured = body.get("captureTime")
    launched = body.get("procLaunch")
    if not isinstance(captured, str) or not isinstance(launched, str) or pids[pid] is None:
        raise ValueError("cannot match process: crash or launch time is missing")
    crash_time, launch_time = report_time(captured), report_time(launched)
    # Session headers record startup to whole seconds, after the process launched.
    # A later launch with the same PID belongs to a different process.
    return start <= crash_time <= end and launch_time <= min(crash_time, pids[pid] + 1)


def export_diagnostics(destination: Path, log_path: Path) -> Path:
    """Create one shareable ZIP, preserving crash reports in their original format."""
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    selected_log: str | None = None
    selected_start: float | None = None
    exported_at = time.time()
    system = (command_output(["/usr/bin/sw_vers"]).strip() + f"\nArchitecture: {platform.machine()}"
              if sys.platform == "darwin" else platform.platform())
    # Build beside the destination so an interrupted export leaves no partial ZIP.
    fd, temporary = tempfile.mkstemp(prefix=".resonance-diagnostics-", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            current_pids: dict[int, float | None] = {}
            path: Path | None = None
            try:
                files = log_files(log_path)
                # Prefer the selected running session even after a clock adjustment.
                path = log_path if log_path in files else next(iter(files), None)
            except OSError as error:
                notes.append(f"Logs unavailable: {error}")
            # Retention on disk is separate from export: send only the selected run.
            if path is not None:
                try:
                    with path.open("rb") as source:
                        size = os.fstat(source.fileno()).st_size
                        beginning = source.read(64 * 1024)
                        headers = b"".join(line for line in beginning.splitlines(keepends=True)
                                           if line.startswith((b"# Python session:", b"# Native session:")))
                        limit = MAX_FILE_BYTES
                        offset = 0
                        if size > limit:
                            headers = headers[:limit]
                            offset = size - (limit - len(headers))
                            source.seek(offset)
                            data = headers + source.read(limit - len(headers))
                        else:
                            source.seek(0)
                            data = source.read(limit)
                    identity = beginning + data
                    current_pids = {int(pid): None for pid in re.findall(rb"# (?:Python|Native) session: pid=(\d+);", identity)}
                    for pid, started in re.findall(rb"# (?:Python|Native) session: pid=(\d+); started=([^;\r\n]+)", identity):
                        try:
                            current_pids[int(pid)] = report_time(started.decode())
                        except ValueError as error:
                            notes.append(f"Process {int(pid)} start time unavailable: {error}.")
                    # Older logs identified processes on each line.
                    for started, pid in re.findall(rb"^([^|\r\n]+) \| [^|\r\n]+ \| resonance[\w.]* \| pid=(\d+) \|", identity, re.MULTILINE):
                        if int(pid) not in current_pids:
                            try:
                                current_pids[int(pid)] = report_time(started.decode().strip())
                            except ValueError:
                                current_pids[int(pid)] = None
                    stamp = re.search(r"-(\d{8}T\d{6}\.\d{6}Z)-", path.name)
                    if stamp:
                        selected_start = datetime.strptime(stamp[1], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
                    if offset:
                        notes.append(f"{path.name}: middle omitted; session headers and final output retained (tail starts at byte {offset}).")
                except OSError as error:
                    notes.append(f"{path.name}: {error}")
                else:
                    selected_log = "logs/" + path.name
                    archive.writestr(selected_log, data)

            if sys.platform == "darwin":
                if selected_start is not None and current_pids:
                    # Framework errors from our processes and our TCC permissions.
                    # Global state dumps can mention our bundle alongside other apps.
                    pids = ", ".join(str(pid) for pid in sorted(current_pids))
                    predicate = (
                        'type == logEvent AND ('
                        f'(processIdentifier IN {{{pids}}} AND (logType == error OR logType == fault))'
                        ' OR (subsystem == "com.apple.TCC" AND composedMessage CONTAINS "com.resonance.app"))'
                    )
                    system_log = command_output([
                        "/usr/bin/log", "show", "--start", f"@{int(selected_start)}",
                        "--end", f"@{int(exported_at) + 1}", "--style", "compact", "--info",
                        "--predicate", predicate,
                    ])
                    archive.writestr("system.log", system_log)
                    notes.append("System log: selected launch to export; process errors/faults and Resonance TCC permissions only.")
                    if system_log.startswith(("Unavailable:", "Exit code", "Incomplete:")):
                        notes.append("macOS system logs could not be fully collected; see system.log.")
                else:
                    notes.append("System logs and crash reports omitted: selected session start or process IDs unavailable.")
                directories = [Path.home() / "Library/Logs/DiagnosticReports",
                               Path("/Library/Logs/DiagnosticReports")]
                crash_remaining = MAX_CRASH_BYTES
                crash_count = 0
                matching_crashes = 0
                unchecked_reports: Counter[str] = Counter()
                for index, directory in enumerate(directories if selected_start is not None and current_pids else []):
                    try:
                        reports = sorted(directory.iterdir(), key=lambda path: path.name, reverse=True)
                    except FileNotFoundError:
                        continue
                    except OSError as error:
                        notes.append(f"Crash reports unavailable in {directory}: {error}")
                        continue
                    for report in reports:
                        try:
                            if report.is_symlink() or not report.is_file() or report.suffix not in {".ips", ".crash"}:
                                continue
                            stat = report.stat()
                            if stat.st_mtime < selected_start:
                                continue
                            if not related_crash_report(report, current_pids, selected_start, exported_at):
                                continue
                        except OSError as error:
                            unchecked_reports[f"unreadable: {error.strerror or type(error).__name__}"] += 1
                            continue
                        except ValueError as error:
                            reason = "invalid metadata" if isinstance(error, (json.JSONDecodeError, UnicodeDecodeError)) else str(error)
                            unchecked_reports[reason] += 1
                            continue
                        matching_crashes += 1
                        if stat.st_size > crash_remaining:
                            notes.append(f"{report.name}: matching crash report omitted ({stat.st_size} bytes; {crash_remaining} bytes of crash budget remaining).")
                            continue
                        archive.write(report, f"crashes/{index}/{report.name}")
                        crash_remaining -= stat.st_size
                        crash_count += 1
                if selected_start is not None and current_pids:
                    notes.append(f"Matching crash reports: {matching_crashes}; included: {crash_count}.")
                    for reason, count in unchecked_reports.items():
                        notes.append(f"Recent reports not checked: {count}; {reason}. Session ownership could not be verified.")
            if selected_log is None:
                notes.append("No application logs were available for this export.")
            details = [
                "Resonance diagnostics",
                f"Exported (UTC): {datetime.fromtimestamp(exported_at, timezone.utc).isoformat(timespec='seconds')}",
                f"Export environment: {system}",
                f"Exporter Python: {sys.version.split()[0]}",
                f"Selected / most recent log: {selected_log or 'unavailable'}",
                "Earlier application sessions are not included.", "",
                "Runtime versions, revisions and process IDs are in each log's session headers.",
                "Files included:",
                *(f"{entry.filename} ({entry.file_size} bytes)" for entry in archive.infolist()), "",
                "Collection notes:", *(notes or ["No collection errors reported."]), "",
                "Send this ZIP with what you clicked, what you expected, and what happened.",
            ]
            archive.writestr("diagnostics.txt", "\n".join(details) + "\n")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-log", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.prepare_log:
        print(prepare_log_file())
        return
    log_path = args.log_file or resolve_log_file(log_to_file=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%SZ")
    output = args.output or Path.cwd() / f"resonance-diagnostics-{stamp}.zip"
    print(export_diagnostics(output, log_path))


if __name__ == "__main__":
    main()
