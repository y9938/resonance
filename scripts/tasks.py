#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv>=1.0.0"]
# ///

"""Run development tasks independently of the project environment."""

import argparse
import csv
import gzip
import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONTAINER_CACHE_ROOT = "/home/resonance/.cache"


def dev_deps(env: dict[str, str]) -> None:
    """Install project and development dependencies into <repo>/.venv."""
    venv = ROOT / ".venv"
    if not venv.exists():
        execute(["uv", "venv", str(venv)], env)

    backend = torch_backend(env)
    cmd = [
        "uv", "pip", "install",
        "--python", str(venv),
        "-r", "pyproject.toml", "--group", "dev",
    ]
    if backend:
        cmd.append(f"--torch-backend={backend}")
    execute(cmd, env)


def dev(env: dict[str, str]) -> None:
    """Run the development server in the project's environment."""
    port = env.get("RESONANCE_PORT", "8000")
    cmd = [
        "uv", "run", "--no-sync", "python", "-m", "uvicorn",
        "server:create_local_app", "--factory", "--reload",
        "--host", "127.0.0.1", "--no-proxy-headers",
    ]
    for path in ("server.py", "core", "stt", "tts", "public"):
        cmd.extend(["--reload-dir", path])
    cmd.extend(["--port", port])
    execute(cmd, env)


def run_tests(env: dict[str, str], args: list[str]) -> None:
    port = env.get("RESONANCE_PORT", "8000")
    cmd = [
        "uv", "run", "--no-sync", "python", "-m", "pytest", "tests/",
        "--base-url", f"http://localhost:{port}",
        *args,
    ]
    execute(cmd, env)


def check(env: dict[str, str], args: list[str]) -> None:
    cmd = [
        "uv", "run", "--no-sync", "python", "-m", "ruff", "check",
        "server.py", "core/", "stt/", "tts/", "tests/", "scripts/",
        *args,
    ]
    execute(cmd, env)


def build_image(env: dict[str, str], args: list[str]) -> None:
    cmd = [
        "docker", "build",
        "--build-arg", f"PYTORCH_BACKEND={build_backend(env)}",
        "-t", container_image(env),
        *args, ".",
    ]
    execute(cmd, env)


def run_container(env: dict[str, str], args: list[str]) -> None:
    """Create host cache directories and run Docker with options before image."""
    if sys.platform == "linux" and os.geteuid() == 0:
        raise PermissionError("Run Docker as a non-root user")
    # Host directories must exist before Docker starts.
    host_caches = cache_paths(env)
    for path in host_caches.values():
        path.mkdir(parents=True, exist_ok=True)
    mounts, container_env = docker_caches(env, host_caches)

    cuda = env.get("DEVICE", "cpu") == "cuda"
    port = env.get("RESONANCE_PORT", "8000")
    cmd = ["docker", "run", "-it", "--rm", "--name", "resonance"]
    if sys.platform == "linux":
        cmd.extend(["--user", f"{os.getuid()}:{os.getgid()}"])
    if cuda:
        cmd.extend(["--gpus", "all"])
    if (ROOT / ".env").exists():
        cmd.extend(["--env-file", str(ROOT / ".env")])

    for source, target in mounts:
        cmd.extend(["--mount", bind_mount(source, target)])

    # Explicit container paths override any host paths loaded from .env.
    for variable, value in container_env.items():
        cmd.extend(["-e", f"{variable}={value}"])
    cmd.extend([
        "-e", f"DEVICE={'cuda' if cuda else 'cpu'}",
        "-e", f"RESONANCE_PORT={port}",
        "-p", f"{port}:{port}",
        *args, container_image(env),  # User arguments are Docker options, before image.
    ])
    execute(cmd, env)


def save_image(env: dict[str, str]) -> None:
    """Replace the archive only after Docker has successfully exported the image."""
    prefix = "gpu_" if env.get("DEVICE", "cpu") == "cuda" else ""
    archive = ROOT / f"resonance_{prefix}{env.get('IMAGE_TAG', 'latest')}.tar.gz"
    with tempfile.NamedTemporaryFile(
        dir=ROOT, prefix=f".{archive.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        cmd = ["docker", "save", container_image(env)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=ROOT, env=env) as proc:
            with (
                proc.stdout,
                gzip.open(temporary_path, "wb", compresslevel=6) as output,
            ):
                shutil.copyfileobj(proc.stdout, output)
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        temporary_path.replace(archive)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Saved to {archive.name}")


def icons(env: dict[str, str]) -> None:
    """Generate favicon and Apple touch icons with resvg and ImageMagick."""
    for tool, message in (
        ("resvg", "resvg not found."),
        ("magick", "magick (ImageMagick) not found."),
    ):
        if not shutil.which(tool, path=env.get("PATH")):
            raise FileNotFoundError(message)

    (ROOT / "build/favicons").mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48):
        execute(
            [
                "resvg", "-w", str(size), "-h", str(size), "public/icon.svg",
                f"build/favicons/{size}.png",
            ],
            env,
        )
    execute(
        [
            "magick", "-background", "none",
            "build/favicons/16.png", "build/favicons/32.png", "build/favicons/48.png",
            "public/favicon.ico",
        ],
        env,
    )
    execute(
        ["resvg", "-w", "180", "-h", "180", "public/icon.svg", "public/apple-touch-icon.png"],
        env,
    )
    print("Web icons generated with pixel-perfect resvg vector rendering.")


def build_macos(env: dict[str, str]) -> None:
    """Build and sign the macOS menu bar app, retaining its existing icon script."""
    if sys.platform != "darwin":
        raise OSError("build-macos is only available on macOS")

    execute(["bash", "scripts/build-icns.sh"], env)
    app = ROOT / "build/Resonance.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents/MacOS"
    resources = app / "Contents/Resources"
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "build/AppIcon.icns", resources)
    status_icons = sorted((ROOT / "build").glob("StatusBarIcon*.png"))
    if not status_icons:
        raise FileNotFoundError("build/StatusBarIcon*.png")
    for icon in status_icons:
        shutil.copy(icon, resources)
    execute(
        [
            "swiftc", "-O", "src/swift/main.swift", "src/swift/CaptureEngine.swift",
            "-o", str(macos / "Resonance"),
        ],
        env,
    )
    shutil.copy(ROOT / "src/swift/Info.plist", app / "Contents/Info.plist")
    # Keep the existing designated requirement stable across recompiles for TCC.
    execute(
        [
            "codesign", "--force", "--deep", "--sign", "-",
            "--requirements", '=designated => identifier "com.resonance.app"',
            str(app),
        ],
        env,
    )
    print("Built and signed: build/Resonance.app")
    applications = Path.home() / "Applications"
    applications.mkdir(parents=True, exist_ok=True)
    link = applications / "Resonance.app"
    link.unlink(missing_ok=True)
    link.symlink_to(app, target_is_directory=True)
    link.touch()  # Refresh the macOS icon cache, following the symlink.
    print("Added to Launchpad")


# Configuration and execution helpers

def torch_backend(env: dict[str, str]) -> str:
    """Choose the backend using the project's CPU/CUDA/MPS defaults."""
    device = env.get("DEVICE", "cpu")
    return env.get("PYTORCH_BACKEND", "cpu" if device == "cpu" else "")


def container_image(env: dict[str, str]) -> str:
    tag = env.get("IMAGE_TAG", "latest")
    prefix = "gpu-" if env.get("DEVICE", "cpu") == "cuda" else ""
    return f"resonance:{prefix}{tag}"


def build_backend(env: dict[str, str]) -> str:
    """Docker uses CPU unless CUDA was explicitly selected."""
    return torch_backend(env) if env.get("DEVICE", "cpu") == "cuda" else "cpu"


def cache_paths(env: dict[str, str]) -> dict[str, Path]:
    """Resolve model cache sources using upstream and Resonance precedence."""
    home_cache = Path.home() / ".cache"
    cache_root = Path(env.get("XDG_CACHE_HOME", str(home_cache)))

    hf_value = env.get("HF_HOME", str(cache_root / "huggingface"))
    hf_home = Path(os.path.expandvars(os.path.expanduser(hf_value)))
    hub_value = env.get(
        "HF_HUB_CACHE", env.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    )
    hf_hub = Path(os.path.expandvars(os.path.expanduser(hub_value)))

    torch_home = Path(os.path.expanduser(env.get("TORCH_HOME", str(cache_root / "torch"))))
    # GigaAM's default does not use XDG_CACHE_HOME.
    gigaam = Path(os.path.expanduser(env.get("GIGAAM_CACHE_DIR") or str(home_cache / "gigaam")))
    resonance = Path(env.get("RESONANCE_CACHE_DIR") or cache_root / "resonance")
    # Sherpa uses RESONANCE_CACHE_DIR itself, without appending /sherpa.
    sherpa = Path(
        env.get("SHERPA_HOME")
        or env.get("RESONANCE_CACHE_DIR")
        or cache_root / "resonance" / "sherpa"
    )

    host_paths = {
        "hf_hub": hf_hub,
        "torch": torch_home,
        "whisper": cache_root / "whisper",
        "gigaam": gigaam,
        "resonance": resonance,
        "sherpa": sherpa,
    }
    return {name: (ROOT / path).resolve() for name, path in host_paths.items()}


def docker_caches(
    env: dict[str, str], host_paths: dict[str, Path]
) -> tuple[list[tuple[Path, str]], dict[str, str]]:
    """Pair host caches with Linux mounts and matching container settings."""
    hf_hub = f"{CONTAINER_CACHE_ROOT}/huggingface/hub"
    torch = f"{CONTAINER_CACHE_ROOT}/torch"
    gigaam = f"{CONTAINER_CACHE_ROOT}/gigaam"
    resonance = f"{CONTAINER_CACHE_ROOT}/resonance"
    mounts = [
        (host_paths["hf_hub"], hf_hub),
        (host_paths["torch"], torch),
        (host_paths["whisper"], f"{CONTAINER_CACHE_ROOT}/whisper"),
        (host_paths["gigaam"], gigaam),
        (host_paths["resonance"], resonance),
    ]
    container_env = {
        "HOME": "/home/resonance",
        "XDG_CACHE_HOME": CONTAINER_CACHE_ROOT,
        # Persist only Hub repositories; tokens, Xet and assets stay in the container.
        "HF_HOME": "/tmp/resonance-huggingface",
        "HF_HUB_CACHE": hf_hub,
        "HUGGINGFACE_HUB_CACHE": hf_hub,
        "TORCH_HOME": torch,
        "GIGAAM_CACHE_DIR": gigaam,
        # Empty values preserve the runtime's unset-or-empty fallback semantics.
        "RESONANCE_CACHE_DIR": "",
        "SHERPA_HOME": "",
    }
    if env.get("RESONANCE_CACHE_DIR"):
        container_env["RESONANCE_CACHE_DIR"] = resonance
    # Without an explicit override, Sherpa lives within the Resonance mount.
    if env.get("SHERPA_HOME"):
        sherpa = f"{CONTAINER_CACHE_ROOT}/sherpa"
        mounts.append((host_paths["sherpa"], sherpa))
        container_env["SHERPA_HOME"] = sherpa
    return mounts, container_env


def bind_mount(source: Path, target: str) -> str:
    """Encode Docker's CSV mount syntax, including commas or quotes in paths."""
    output = io.StringIO()
    csv.writer(output, lineterminator="").writerow(
        ["type=bind", f"source={source}", f"target={target}"]
    )
    return output.getvalue()


def task_environment() -> dict[str, str]:
    """Load repository configuration and prepare the environment for child uv."""
    load_dotenv(ROOT / ".env", override=False)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    # Neither the PEP 723 runner nor an activated external environment owns
    # the project's virtual environment.
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    return env


def execute(cmd: list[str], env: dict[str, str]) -> None:
    """Run a command from the repository and stop on failure."""
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    # Command name: (handler, accepts arguments for the underlying tool).
    tasks = {
        "dev-deps": (dev_deps, False),
        "dev": (dev, False),
        "test": (run_tests, True),
        "check": (check, True),
        "build": (build_image, True),
        "run": (run_container, True),
        "save": (save_image, False),
        "icons": (icons, False),
        "build-macos": (build_macos, False),
    }
    parser = argparse.ArgumentParser(
        prog="r", usage="%(prog)s [-h] [task] [args ...]", description=__doc__
    )
    parser.add_argument(
        "task", choices=tasks, metavar="task", nargs="?",
        help=", ".join(tasks),
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER, help="Arguments passed to the selected task"
    )
    args = parser.parse_args()
    if args.task is None:
        parser.print_help()
        return 0

    handler, accepts_args = tasks[args.task]
    if args.args and not accepts_args:
        parser.error(f"{args.task} does not accept arguments")

    try:
        env = task_environment()
        if accepts_args:
            handler(env, args.args)
        else:
            handler(env)
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except (OSError, ValueError) as exc:
        print(f"tasks: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
