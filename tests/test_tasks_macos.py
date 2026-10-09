"""Installing one macOS bundle preserves a working app when compilation fails."""

import plistlib
import subprocess
from pathlib import Path

import pytest

from scripts import tasks


def test_macos_backend_receives_ffmpeg_path_without_changing_parent_env(tmp_path, monkeypatch):
    (tmp_path / "dist/web").mkdir(parents=True)
    (tmp_path / "dist/web/index.html").touch()
    prefix = tmp_path / "homebrew with spaces/ffmpeg"
    (prefix / "lib").mkdir(parents=True)
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    monkeypatch.setattr(tasks.sys, "platform", "darwin")
    monkeypatch.setattr(tasks.shutil, "which", lambda *args, **kwargs: "/test/brew")
    launched = []

    def run(command, **kwargs):
        if command[0] == "/test/brew":
            return subprocess.CompletedProcess(command, 0, str(prefix) + "\n", "")
        launched.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tasks.subprocess, "run", run)
    original = {"DYLD_LIBRARY_PATH": "/custom/lib", "RESONANCE_FRONTEND_DEV": "1"}
    tasks.serve_local(original)

    _, child_env = launched[0]
    assert child_env["DYLD_LIBRARY_PATH"] == f"/custom/lib:{prefix}/lib"
    assert "RESONANCE_FRONTEND_DEV" not in child_env
    assert original == {"DYLD_LIBRARY_PATH": "/custom/lib", "RESONANCE_FRONTEND_DEV": "1"}


def test_arm_dependencies_and_backend_share_ffmpeg(tmp_path, monkeypatch):
    prefix = tmp_path / "homebrew with spaces/ffmpeg"
    (prefix / "lib").mkdir(parents=True)
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(tasks, "ROOT", tmp_path)
    monkeypatch.setattr(tasks.sys, "platform", "darwin")
    monkeypatch.setattr(tasks.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(tasks.shutil, "which", lambda name, **kwargs: f"/test/{name}")
    monkeypatch.setattr(tasks, "frontend", lambda *args: None)
    installed = []
    monkeypatch.setattr(tasks, "execute", lambda command, env: installed.append((command, env)))

    def run(command, **kwargs):
        output = str(prefix) if command[0] == "/test/brew" else "18.1.0"
        return subprocess.CompletedProcess(command, 0, output + "\n", "")

    monkeypatch.setattr(tasks.subprocess, "run", run)
    original = {"PATH": "/usr/bin", "PKG_CONFIG_PATH": "/custom/pkgconfig"}
    tasks.dev_deps(original)
    backend_env = tasks.backend_environment(original)

    command, build_env = installed[-1]
    assert "--no-binary" in command and "--reinstall-package" in command
    assert "--no-cache" in command and command[-1] == "av==18.1.0"
    assert build_env["PKG_CONFIG_PATH"].split(":")[0] == f"{backend_env['DYLD_LIBRARY_PATH']}/pkgconfig"
    assert original == {"PATH": "/usr/bin", "PKG_CONFIG_PATH": "/custom/pkgconfig"}


@pytest.fixture
def macos_build(tmp_path, monkeypatch):
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(tasks, "ROOT", repo)
    monkeypatch.setattr(tasks.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(tasks, "frontend", lambda *args: None)
    (repo / "build").mkdir()
    (repo / "build/AppIcon.icns").write_bytes(b"icon")
    (repo / "build/StatusBarIcon.png").write_bytes(b"status")
    (repo / "src/macos").mkdir(parents=True)
    metadata = {"CFBundleIdentifier": "com.resonance.app"}
    (repo / "src/macos/Info.plist").write_bytes(plistlib.dumps(metadata))

    def execute(command, env):
        if command[0] == "swiftc":
            Path(command[-1]).write_bytes(b"compiled launcher")
        elif command[0] == "codesign":
            app = Path(command[-1])
            assert (app / "Contents/Resources/RepositoryPath.txt").read_text() == str(repo)
            assert tasks.is_resonance_app(app)

    monkeypatch.setattr(tasks, "execute", execute)
    destination = home / "Applications/Resonance.app"
    destination.parent.mkdir()
    return repo, destination


def bundle(path, identifier="com.resonance.app"):
    (path / "Contents").mkdir(parents=True)
    (path / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": identifier}))
    (path / "old-version").write_text("old app")


@pytest.mark.parametrize("legacy_link", [False, True])
def test_macos_installs_one_real_bundle(macos_build, legacy_link):
    repo, destination = macos_build
    legacy = repo / "build/Resonance.app"
    bundle(legacy)
    if legacy_link:
        destination.symlink_to(legacy, target_is_directory=True)
    else:
        bundle(destination)
    tasks.build_macos({})
    assert destination.is_dir() and not destination.is_symlink()
    assert not legacy.exists()
    assert not (destination / "old-version").exists()
    assert (destination / "Contents/MacOS/Resonance").read_bytes() == b"compiled launcher"
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["swiftc", "codesign"])
def test_macos_failed_build_preserves_installed_app(macos_build, monkeypatch, failure):
    _, destination = macos_build
    bundle(destination)
    execute = tasks.execute

    def fail(command, env):
        if command[0] == failure:
            raise subprocess.CalledProcessError(1, command)
        execute(command, env)

    monkeypatch.setattr(tasks, "execute", fail)
    with pytest.raises(subprocess.CalledProcessError):
        tasks.build_macos({})
    assert (destination / "old-version").read_text() == "old app"
    assert list(destination.parent.iterdir()) == [destination]


def test_macos_install_failure_rolls_back(macos_build, monkeypatch):
    _, destination = macos_build
    bundle(destination)
    rename = Path.rename

    def fail_new_bundle(path, target):
        if path.name == "Resonance.app" and path.parent.name.startswith(".resonance-build-"):
            raise OSError("installation failed")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_new_bundle)
    with pytest.raises(OSError, match="installation failed"):
        tasks.build_macos({})
    assert (destination / "old-version").read_text() == "old app"
    assert list(destination.parent.iterdir()) == [destination]


def test_macos_does_not_replace_unrelated_app(macos_build):
    _, destination = macos_build
    bundle(destination, identifier="other.application")
    with pytest.raises(FileExistsError):
        tasks.build_macos({})
    assert (destination / "old-version").is_file()
