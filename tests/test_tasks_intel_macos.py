"""Intel tasks preserve the installed native stack instead of resolving ARM wheels."""

import json

import pytest

from scripts import tasks


@pytest.fixture
def intel_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    monkeypatch.setattr(tasks.sys, "platform", "darwin")
    monkeypatch.setattr(tasks.platform, "machine", lambda: "x86_64")
    prefix = tmp_path / ".deps/macos-intel"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").touch()
    return tmp_path, prefix


def test_serve_intel_uses_native_python_and_ffmpeg_without_homebrew(intel_repo, monkeypatch):
    root, prefix = intel_repo
    (root / "dist/web").mkdir(parents=True)
    (root / "dist/web/index.html").touch()
    launched = []
    monkeypatch.setattr(tasks, "execute", lambda command, env: launched.append((command, env)))
    monkeypatch.setattr(tasks.shutil, "which", lambda *args, **kwargs: pytest.fail("Homebrew must not replace conda libraries"))
    original = {"PATH": "/usr/bin", "RESONANCE_FRONTEND_DEV": "1"}

    tasks.serve_local(original)

    command, env = launched[0]
    assert command[:3] == [str(prefix / "bin/python"), "-m", "uvicorn"]
    assert env["PATH"].split(":")[0] == str(prefix / "bin")
    assert "DYLD_LIBRARY_PATH" not in env
    assert "RESONANCE_FRONTEND_DEV" not in env
    assert original == {"PATH": "/usr/bin", "RESONANCE_FRONTEND_DEV": "1"}


def test_dev_deps_preserves_native_libraries_when_lock_already_installed(intel_repo, monkeypatch):
    root, prefix = intel_repo
    profile = root / "tools/macos-intel"
    profile.mkdir(parents=True)
    package = {"url": "https://conda.example/osx-64/native.conda", "md5": "tested-build"}
    (profile / "conda-osx-64.lock").write_text(f"@EXPLICIT\n{package['url']}#{package['md5']}\n")
    (prefix / "conda-meta").mkdir()
    (prefix / "conda-meta/native.json").write_text(json.dumps(package))
    launched = []
    monkeypatch.setattr(tasks, "execute", lambda command, env: launched.append(command))
    monkeypatch.setattr(tasks, "frontend", lambda *args: None)
    monkeypatch.setattr(tasks.shutil, "which", lambda *args, **kwargs: pytest.fail("Native stack is already installed"))

    tasks.dev_deps({"PATH": "/usr/bin"})

    assert len(launched) == 1
    assert launched[0][:3] == ["uv", "pip", "install"]
