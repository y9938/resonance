#!/bin/zsh
set -eu

REPO_DIR="${0:A:h:h:h}"
cd "$REPO_DIR"

brew install just uv ffmpeg resvg imagemagick

[ -f .env ] || cp .env.example .env

if [ "$(uname -m)" = "arm64" ]; then
  sed -i '' 's/^DEVICE=cpu$/DEVICE=mps/' .env

  if ! grep -q '^PYTORCH_ENABLE_MPS_FALLBACK=' .env; then
    sed -i '' '/^DEVICE=/a\
PYTORCH_ENABLE_MPS_FALLBACK=1
' .env
  fi
else
  sed -i '' 's/^PYTORCH_BACKEND=$/PYTORCH_BACKEND=cpu/' .env
fi

uv venv --python 3.12

just dev-deps
