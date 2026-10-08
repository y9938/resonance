import io
import logging
import os
import time
from pathlib import Path

import pytest

from core.logging import (
    log_files,
    prepare_log_file,
    resolve_log_file,
    resolve_log_level,
    setup_logging,
)


@pytest.fixture(autouse=True)
def isolate_file_logging(monkeypatch):
    monkeypatch.delenv("RESONANCE_LOG_TO_FILE", raising=False)
    monkeypatch.delenv("RESONANCE_LOG_FILE", raising=False)


def test_resolve_log_level_defaults(monkeypatch):
    monkeypatch.delenv("RESONANCE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert resolve_log_level() == logging.INFO


def test_resolve_log_level_explicit_arg():
    assert resolve_log_level("debug") == logging.DEBUG
    assert resolve_log_level("WARNING") == logging.WARNING


def test_resolve_log_level_env(monkeypatch):
    monkeypatch.setenv("RESONANCE_LOG_LEVEL", "ERROR")
    assert resolve_log_level() == logging.ERROR


def test_resolve_log_level_fallback_env(monkeypatch):
    monkeypatch.delenv("RESONANCE_LOG_LEVEL", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert resolve_log_level() == logging.WARNING


def test_resolve_log_level_invalid_fails_fast(monkeypatch):
    monkeypatch.setenv("RESONANCE_LOG_LEVEL", "INVALID_LEVEL")
    with pytest.raises(ValueError):
        resolve_log_level()


def test_setup_logging_idempotence_and_propagation():
    stream = io.StringIO()
    logger = setup_logging(level_name="DEBUG", stream=stream, root_name="resonance_test")

    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert logger.propagate is False

    setup_logging(level_name="DEBUG", stream=stream, root_name="resonance_test")
    assert len(logger.handlers) == 1

    child_logger = logging.getLogger("resonance_test.stt.stream_vad")
    child_logger.debug("Silero VAD test event")

    output = stream.getvalue()
    assert "DEBUG" in output
    assert "test_logging:" in output
    assert "resonance_test.stt.stream_vad" not in output
    assert "Silero VAD test event" in output


def test_third_party_transports_suppression():
    setup_logging(level_name="INFO", root_name="resonance_test")
    for noisy in ("multipart", "httpcore", "httpx", "uvicorn.access"):
        assert logging.getLogger(noisy).level == logging.WARNING

    setup_logging(level_name="DEBUG", root_name="resonance_test")
    for noisy in ("multipart", "httpcore", "httpx", "uvicorn.access"):
        assert logging.getLogger(noisy).level == logging.DEBUG


def test_resolve_log_file_defaults(monkeypatch):
    monkeypatch.delenv("RESONANCE_LOG_TO_FILE", raising=False)
    monkeypatch.delenv("RESONANCE_LOG_FILE", raising=False)
    assert resolve_log_file() is None


def test_resolve_log_file_enabled_by_env(monkeypatch):
    monkeypatch.setenv("RESONANCE_LOG_TO_FILE", "1")
    monkeypatch.delenv("RESONANCE_LOG_FILE", raising=False)
    resolved = resolve_log_file()
    assert resolved is not None


def test_resolve_log_file_custom_path(monkeypatch, tmp_path):
    custom = tmp_path / "custom.log"
    monkeypatch.setenv("RESONANCE_LOG_FILE", str(custom))
    resolved = resolve_log_file()
    assert resolved == custom.resolve()


def test_setup_logging_with_file_handler(tmp_path):
    log_file = tmp_path / "test_run.log"
    stream = io.StringIO()
    logger = setup_logging(
        level_name="INFO",
        stream=stream,
        root_name="resonance_file_test",
        log_file=log_file,
    )

    assert len(logger.handlers) == 2
    logger.info("Message for both stream and file")

    assert "Message for both stream and file" in stream.getvalue()

    for handler in logger.handlers:
        handler.flush()
    session = log_files(log_file)[0]
    assert session.name.startswith("test_run-")
    content = session.read_text(encoding="utf-8")
    assert "Message for both stream and file" in content

    old_handler = logger.handlers[-1]
    setup_logging(level_name="DEBUG", stream=stream, root_name="resonance_file_test", log_file=log_file)
    logger.info("Still the same session")
    assert log_files(log_file) == [session]
    assert old_handler.stream is None
    assert "Still the same session" in session.read_text()
    assert session.read_text().count("# Python session:") == 1


def test_session_history_uses_age_and_size_instead_of_start_count(tmp_path, monkeypatch):
    import core.logging as logs

    monkeypatch.setattr(logs, "LOG_HISTORY_BYTES", 100)
    base = tmp_path / "server.log"
    expired = prepare_log_file(base)
    expired.write_text("old failure")
    old_time = time.time() - 8 * 86400
    os.utime(expired, (old_time, old_time))
    first = prepare_log_file(base)
    assert not expired.exists()
    first.write_bytes(b"a" * 60)
    os.utime(first, (time.time() - 20, time.time() - 20))
    second = prepare_log_file(base)
    second.write_bytes(b"b" * 60)
    os.utime(second, (time.time() - 10, time.time() - 10))
    current = prepare_log_file(base)
    assert current.exists() and second.exists() and not first.exists()
    for _ in range(6):
        prepare_log_file(base)
    assert len(log_files(base)) == 8


def test_history_cleanup_preserves_unrelated_files_and_symlinks(tmp_path, monkeypatch):
    import core.logging as logs

    monkeypatch.setattr(logs, "LOG_HISTORY_BYTES", 0)
    base = tmp_path / "server.log"
    unrelated = tmp_path / "notes.log"
    unrelated.write_text("keep this")
    link = prepare_log_file(base)
    link.unlink()
    link.symlink_to(unrelated)
    current = prepare_log_file(base)
    assert unrelated.read_text() == "keep this"
    assert link.is_symlink() and current.exists()


def test_history_cleanup_failure_keeps_new_log_usable(tmp_path, monkeypatch):
    import core.logging as logs

    monkeypatch.setattr(logs, "LOG_HISTORY_BYTES", 0)
    base = tmp_path / "server.log"
    old = prepare_log_file(base)
    old.write_text("failure")
    unlink = Path.unlink

    def cannot_remove_old(path, **kwargs):
        if path == old:
            raise PermissionError("old log is read-only")
        return unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", cannot_remove_old)
    current = prepare_log_file(base)
    assert current.exists() and old.exists()
    assert "Log history cleanup failed" in current.read_text()


def test_session_metadata_is_kept_at_error_level_and_not_repeated_per_message(tmp_path):
    stream = io.StringIO()
    base = tmp_path / "server.log"
    logger = setup_logging(level_name="ERROR", stream=stream, root_name="resonance.header_test", log_file=base)
    logger.info("filtered progress")
    logger.error("actual failure")
    content = log_files(base)[0].read_text()
    header, message = content.splitlines()
    assert header.startswith(f"# Python session: pid={os.getpid()}; started=")
    assert "revision=" in header and "python=" in header and "platform=" in header
    assert "filtered progress" not in content
    assert "pid=" not in message and "resonance.header_test" not in message
    assert "ERROR" in message and "actual failure" in message
