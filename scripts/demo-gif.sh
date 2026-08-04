#!/usr/bin/env bash
set -euo pipefail

# Record demo video and convert to GIF

SCRIPT_DIR="$(cd -P "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

echo "=== Recording demo video ==="
.venv/bin/pytest tests/e2e/demo_recording.py --video=on

echo "=== Converting to GIF ==="
VIDEO_FILE=$(find "test-results" -name "video.webm" -type f | head -1)

ffmpeg -i "$VIDEO_FILE" \
    -vf "fps=1,scale=1280:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 demo.gif -y

echo "Created: demo.gif"

# Optional: create MP4 if --mp4 flag is passed
if [[ "${1:-}" == "--mp4" ]]; then
    echo "=== Creating MP4 ==="
    ffmpeg -i "$VIDEO_FILE" -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k demo.mp4 -y
    echo "Created: demo.mp4"
fi
