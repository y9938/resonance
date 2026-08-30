import io
import logging

import pytest

from core.logging import (
    VALID_LOG_LEVELS,
    get_default_log_file,
    resolve_log_file,
    resolve_log_level,
    setup_logging,
)


def test_valid_log_levels_constant():
    assert "DEBUG" in VALID_LOG_LEVELS
    assert "INFO" in VALID_LOG_LEVELS
    assert "WARNING" in VALID_LOG_LEVELS
    assert "ERROR" in VALID_LOG_LEVELS
    assert "CRITICAL" in VALID_LOG_LEVELS


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
    with pytest.raises(ValueError, match="Invalid log level 'INVALID_LEVEL'"):
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
    assert "resonance_test.stt.stream_vad" in output
    assert "Silero VAD test event" in output


def test_third_party_transports_suppression():
    setup_logging(level_name="INFO", root_name="resonance_test")
    for noisy in ("multipart", "httpcore", "httpx", "uvicorn.access"):
        assert logging.getLogger(noisy).level == logging.WARNING

    setup_logging(level_name="DEBUG", root_name="resonance_test")
    for noisy in ("multipart", "httpcore", "httpx", "uvicorn.access"):
        assert logging.getLogger(noisy).level == logging.DEBUG


def test_get_default_log_file():
    default_path = get_default_log_file()
    assert default_path.name == "server.log"


def test_resolve_log_file_defaults(monkeypatch):
    monkeypatch.delenv("RESONANCE_LOG_TO_FILE", raising=False)
    monkeypatch.delenv("RESONANCE_LOG_FILE", raising=False)
    assert resolve_log_file() is None


def test_resolve_log_file_enabled_by_env(monkeypatch):
    monkeypatch.setenv("RESONANCE_LOG_TO_FILE", "1")
    monkeypatch.delenv("RESONANCE_LOG_FILE", raising=False)
    resolved = resolve_log_file()
    assert resolved is not None
    assert resolved.name == "server.log"


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
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Message for both stream and file" in content
