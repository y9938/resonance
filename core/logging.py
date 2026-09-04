from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

VALID_LOG_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT: str = "%H:%M:%S"
THIRD_PARTY_NOISY_LOGGERS: tuple[str, ...] = (
    "multipart",
    "httpcore",
    "httpx",
    "uvicorn.access",
)


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


def setup_logging(
    level_name: str | None = None,
    stream: TextIO = sys.stderr,
    root_name: str = "resonance",
    log_to_file: bool | None = None,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Idempotently configures the root package logger, optional rotating file handler, and manages third-party transports.

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

    logger.handlers.clear()
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    target_file = resolve_log_file(log_to_file=log_to_file, log_file=log_file)
    if target_file is not None:
        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                target_file,
                maxBytes=5 * 1024 * 1024,
                backupCount=1,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to initialize file logger at '{target_file}': {e}")

    logger.propagate = False

    # Domain Invariant: Third-party transport loggers emit excessive payload tracing unless clamped to WARNING.
    external_level = logging.DEBUG if level == logging.DEBUG else logging.WARNING
    for noisy in THIRD_PARTY_NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(external_level)

    return logger
