#!/bin/zsh
set -eu

REPO_DIR="${${(%):-%x}:A:h:h}"
cd "$REPO_DIR"

brew install just uv ffmpeg resvg imagemagick

if [ ! -f .env ]; then
  cp .env.example .env

  if [ "$(uname -m)" = "arm64" ]; then
    sed -i '' 's/^DEVICE=cpu$/DEVICE=mps/' .env
    sed -i '' '/^DEVICE=/a\
PYTORCH_ENABLE_MPS_FALLBACK=1
' .env
  else
    sed -i '' 's/^PYTORCH_BACKEND=$/PYTORCH_BACKEND=cpu/' .env
  fi

  sed -i '' 's/^RESONANCE_LOG_TO_FILE=0$/RESONANCE_LOG_TO_FILE=1/' .env
fi

uv venv --python 3.12

just dev-deps
