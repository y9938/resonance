from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
DEFAULT_LOG_FORMAT: str = "%(asctime)s %(levelname)-7s %(module)s: %(message)s"
DEFAULT_DATE_FORMAT: str = "%H:%M:%S"
LOG_HISTORY_DAYS = 7
LOG_HISTORY_BYTES = 20 * 1024 * 1024
SESSION_SUFFIX = r"-\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}"
THIRD_PARTY_NOISY_LOGGERS: tuple[str, ...] = (
    "multipart",
    "httpcore",
    "httpx",
    "uvicorn.access",
)


def session_header() -> str:
    root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except (OSError, subprocess.TimeoutExpired):
        revision = "unavailable"
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    torch_version = getattr(sys.modules.get("torch"), "__version__", "not loaded")
    return (f"# Python session: pid={os.getpid()}; started={started}; "
            f"python={sys.version.split()[0]}; torch={torch_version}; "
            f"device={os.getenv('DEVICE', 'cpu')}; platform={platform.platform()}; revision={revision}")


def get_default_log_file() -> Path:
    """Returns standard platform-specific log file path."""
    if sys.platform == "darwin":
        base_dir = Path.home() / "Library" / "Logs" / "Resonance"
    elif sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        base_dir = Path(local_app_data) / "resonance" / "logs"
    else:
        cache_home = os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        base_dir = Path(cache_home) / "resonance" / "logs"
    return base_dir / "server.log"


def resolve_log_file(
    log_to_file: bool | None = None,
    log_file: str | Path | None = None,
) -> Path | None:
    """
    Resolves target log file path from arguments or environment variables.
    Precedence:
      1. Explicit log_file argument or RESONANCE_LOG_FILE env var -> custom Path.
      2. Explicit log_to_file=True or RESONANCE_LOG_TO_FILE=1/true -> default platform log Path.
      3. Otherwise -> None.
    """
    raw_file = (
        str(log_file)
        if log_file is not None
        else os.getenv("RESONANCE_LOG_FILE", "")
    ).strip()

    if raw_file:
        return Path(os.path.expanduser(raw_file)).resolve()

    should_log = (
        log_to_file
        if log_to_file is not None
        else os.getenv("RESONANCE_LOG_TO_FILE", "0").strip().lower() in {"1", "true", "yes"}
    )

    if should_log:
        return get_default_log_file()

    return None


def resolve_log_level(level_name: str | None = None) -> int:
    """
    Fail-fast resolution of log level from argument or environment variables.
    Precedence: explicit arg -> RESONANCE_LOG_LEVEL -> LOG_LEVEL -> INFO.

    Raises:
        ValueError: If resolved log level is not in VALID_LOG_LEVELS.
    """
    raw = (
        level_name
        or os.getenv("RESONANCE_LOG_LEVEL")
        or os.getenv("LOG_LEVEL")
        or "INFO"
    ).strip().upper()

    if raw not in VALID_LOG_LEVELS:
        valid_str = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ValueError(
            f"Invalid log level '{raw}'. Valid options: {valid_str}"
        )
    return getattr(logging, raw)


def log_files(path: Path) -> list[Path]:
    """Newest first; accept either a configured base path or a session file."""
    stem = re.sub(SESSION_SUFFIX + "$", "", path.stem)
    suffix = path.suffix or ".log"
    pattern = re.compile(re.escape(stem) + SESSION_SUFFIX + re.escape(suffix) + "$")
    legacy = {stem + suffix, *(f"{stem}{suffix}.{n}" for n in range(1, 5))}
    if stem == "server" and suffix == ".log":
        legacy.update({"backend.log", "backend.previous.log"})
    files = [
        entry for entry in path.parent.iterdir()
        if not entry.is_symlink() and entry.is_file()
        and (pattern.fullmatch(entry.name) or entry.name in legacy
             or (stem == "server" and entry.name.startswith("backend-") and entry.suffix == ".log"))
    ]
    return sorted(files, key=lambda entry: (entry.stat().st_mtime, entry.name), reverse=True)


def prepare_log_file(path: Path | None = None) -> Path:
    """Preserve the full active run; apply age/size limits to history at next start."""
    base = (path or resolve_log_file(log_to_file=True)).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    current = base.with_name(f"{base.stem}-{stamp}-{uuid.uuid4().hex}{base.suffix or '.log'}")
    current.touch(exist_ok=False)
    try:
        remaining = LOG_HISTORY_BYTES
        cutoff = time.time() - LOG_HISTORY_DAYS * 86400
        for previous in log_files(base):
            if previous == current:
                continue
            stat = previous.stat()
            if stat.st_mtime < cutoff or stat.st_size > remaining:
                previous.unlink()
            else:
                remaining -= stat.st_size
    except OSError as error:
        with current.open("a", encoding="utf-8") as output:
            output.write(f"Log history cleanup failed: {error}\n")
    return current


def setup_logging(
    level_name: str | None = None,
    stream: TextIO = sys.stderr,
    root_name: str = "resonance",
    log_to_file: bool | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Configure console output and an optional session file, with shared history retention.

    Args:
        level_name: Optional explicit level name (e.g. 'DEBUG', 'INFO').
        stream: Output stream (defaults to sys.stderr adhering to 12-factor / POSIX standards).
        root_name: Root logger name to configure.
        log_to_file: Optional flag to enable file logging.
        log_file: Optional custom log file path.

    Returns:
        Configured root logger instance.
    """
    level = resolve_log_level(level_name)
    logger = logging.getLogger(root_name)
    logger.setLevel(level)

    # Reconfiguring levels must not create another session file or leak handles.
    previous_handlers = logger.handlers[:]
    previous_files = [Path(handler.baseFilename) for handler in previous_handlers
                      if isinstance(handler, logging.FileHandler)]
    for old_handler in logger.handlers[:]:
        logger.removeHandler(old_handler)
        old_handler.close()
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    new_session_handlers = [] if previous_handlers else [handler]

    target_file = resolve_log_file(log_to_file=log_to_file, log_file=log_file)
    if target_file is not None:
        try:
            existing = next((path for path in previous_files
                             if path.parent == target_file.parent and path.exists()
                             and path in log_files(target_file)), None)
            session_file = existing or prepare_log_file(target_file)
            file_handler = logging.FileHandler(session_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            if existing is None:
                new_session_handlers.append(file_handler)
        except Exception as e:
            logger.warning(f"Failed to initialize file logger at '{target_file}': {e}")

    logger.propagate = False
    if new_session_handlers:
        header = session_header()
        for destination in new_session_handlers:
            # Session identity is metadata, independent of the selected verbosity.
            destination.stream.write(header + "\n")
            destination.flush()

    # Domain Invariant: Third-party transport loggers emit excessive payload tracing unless clamped to WARNING.
    external_level = logging.DEBUG if level == logging.DEBUG else logging.WARNING
    for noisy in THIRD_PARTY_NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(external_level)

    return logger
