import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from unittest.mock import ANY, MagicMock, patch

import pytest

from stt.models.gigaam import GigaAMAdapter, load_gigaam
from stt.models.granite import GraniteAdapter
from stt.models.manager import ModelManager
from stt.models.whisper import WhisperAdapter


@pytest.mark.parametrize(
    ("cache_dir", "expected"),
    [
        (None, None),
        ("", None),
        ("/models/gigaam", "/models/gigaam"),
        ("~/models/gigaam", str(Path.home() / "models/gigaam")),
        ("models/gigaam", "models/gigaam"),
    ],
)
def test_gigaam_cache_override(monkeypatch, cache_dir, expected):
    import gigaam

    if cache_dir is None:
        monkeypatch.delenv("GIGAAM_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("GIGAAM_CACHE_DIR", cache_dir)
    upstream_loader = MagicMock()
    monkeypatch.setattr(gigaam, "load_model", upstream_loader)

    load_gigaam("cpu")

    assert upstream_loader.call_args.kwargs["download_root"] == expected


def test_model_manager_get_stt_model():
    mock_gigaam = MagicMock(spec=GigaAMAdapter)
    mock_whisper = MagicMock(spec=WhisperAdapter)
    mock_granite = MagicMock(spec=GraniteAdapter)

    mgr = ModelManager(
        gigaam_loader=lambda: mock_gigaam,
        whisper_loader=lambda: mock_whisper,
        granite_loader=lambda: mock_granite,
    )

    assert mgr.get_stt_model("gigaam") is mock_gigaam
    assert mgr.get_stt_model("whisper") is mock_whisper
    assert mgr.get_stt_model("granite") is mock_granite
    assert mgr.get_stt_model("  WHISPER  ") is mock_whisper


def test_model_manager_unknown_model_raises():
    mgr = ModelManager()
    with pytest.raises(ValueError):
        mgr.get_stt_model("non_existent")


def test_model_manager_reports_wait_for_another_load():
    load_started = Event()
    finish_load = Event()
    second_waiting = Event()
    model = object()

    def loader():
        load_started.set()
        assert finish_load.wait(timeout=2)
        return model

    manager = ModelManager(gigaam_loader=loader)

    class TrackingLock:
        def __init__(self):
            self._lock = Lock()
            self._attempts = 0

        def __enter__(self):
            self._attempts += 1
            if self._attempts == 1:
                assert self._lock.acquire(blocking=False)
            elif not self._lock.acquire(blocking=False):
                second_waiting.set()
                self._lock.acquire()
            return self

        def __exit__(self, *_exc):
            self._lock.release()

    manager._lock = TrackingLock()
    with patch("stt.models.manager.log.isEnabledFor", return_value=True), patch(
        "stt.models.manager.log.debug",
    ) as debug_log:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(manager.stt_gigaam)
            assert load_started.wait(timeout=1)
            second = executor.submit(manager.stt_gigaam)
            assert second_waiting.wait(timeout=1)
            time.sleep(0.02)
            finish_load.set()
            assert first.result(timeout=2) is model
            assert second.result(timeout=2) is model

    debug_log.assert_any_call(
        "Model manager wait: model_id=%s lock_wait=%.2fs cache_hit=%s",
        "gigaam", ANY, True,
    )
