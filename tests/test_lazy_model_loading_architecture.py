"""Regression tests for lazy-only model loading architecture."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "Dockerfile"
SERVER = ROOT / "server.py"


def test_dockerfile_does_not_predownload_models() -> None:
    source = DOCKERFILE.read_text()

    assert "Pre-download models" not in source
    assert "torch.hub.load('snakers4/silero-models'" not in source
    assert "gigaam.load_model('v3_e2e_ctc')" not in source
    assert "TTSModel.load_model()" not in source
    assert "/root/.cache/gigaam" not in source
    assert "/root/.cache/huggingface" not in source
    assert "/root/.cache/torch" not in source


def test_server_does_not_preload_models_on_startup() -> None:
    source = SERVER.read_text()

    assert "async def _preload_models" not in source
    assert "Pre-loading models in background" not in source
    assert "Models pre-loaded" not in source
    assert "preload_task =" not in source


def test_project_uses_kokoro_and_not_pocket_tts() -> None:
    source = (ROOT / "pyproject.toml").read_text()

    assert '"kokoro' in source
    assert "pocket-tts" not in source
