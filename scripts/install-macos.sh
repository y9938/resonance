#!/bin/zsh
set -eu

REPO_DIR="${${(%):-%x}:A:h:h}"
cd "$REPO_DIR"

brew install just uv ffmpeg resvg imagemagick

if [ ! -f .env ]; then
  if [ "$(uname -m)" = "arm64" ]; then
    cat << 'EOF' > .env
DEVICE=mps
PYTORCH_ENABLE_MPS_FALLBACK=1
RESONANCE_LOG_TO_FILE=1
EOF
  else
    cat << 'EOF' > .env
RESONANCE_LOG_TO_FILE=1
EOF
  fi
fi

uv venv --python 3.12

just dev-deps
