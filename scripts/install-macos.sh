#!/bin/zsh
set -eu

REPO_DIR="${${(%):-%x}:A:h:h}"
cd "$REPO_DIR"

command -v brew >/dev/null 2>&1 || {
  if [ "$(uname -m)" = "x86_64" ]; then
    print -u2 "Homebrew is required; see tools/macos-intel/README.md#homebrew-on-intel"
  else
    print -u2 "Homebrew is required: https://brew.sh/"
  fi
  exit 1
}

if [ "$(uname -m)" = "x86_64" ]; then
  brew install uv micromamba
else
  brew install uv node ffmpeg pkg-config resvg imagemagick
fi

if [ ! -f .env ]; then
  if [ "$(uname -m)" = "arm64" ]; then
    cat << 'EOF' > .env
DEVICE=mps
PYTORCH_ENABLE_MPS_FALLBACK=1
RESONANCE_LOG_TO_FILE=1
EOF
  else
    cat << 'EOF' > .env
DEVICE=cpu
RESONANCE_LOG_TO_FILE=1
EOF
  fi
fi

if [ "$(uname -m)" != "x86_64" ]; then
  uv venv --python 3.12
fi

uv run --script scripts/tasks.py dev-deps
