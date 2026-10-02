#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["python-dotenv>=1.0.0"]
# ///

"""Bootstrap development tasks independently of the project environment."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


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


def torch_backend(env: dict[str, str]) -> str:
    """Choose the backend using the project's CPU/CUDA/MPS defaults."""
    device = env.get("DEVICE", "cpu")
    return env.get("PYTORCH_BACKEND", "cpu" if device == "cpu" else "")


def dev_deps(env: dict[str, str]) -> None:
    """Install project and development dependencies into <repo>/.venv."""
    venv = ROOT / ".venv"
    if not venv.exists():
        subprocess.run(["uv", "venv", str(venv)], cwd=ROOT, env=env, check=True)

    backend = torch_backend(env)
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv),
        "-r",
        "pyproject.toml",
        "--group",
        "dev",
    ]
    if backend:
        cmd.append(f"--torch-backend={backend}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["dev-deps"], nargs="?")
    args = parser.parse_args()
    if args.task is None:
        parser.print_help()
        return 0

    try:
        dev_deps(task_environment())
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except OSError as exc:
        print(f"tasks: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
