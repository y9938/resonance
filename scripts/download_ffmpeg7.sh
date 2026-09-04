#!/usr/bin/env bash
set -euo pipefail

# Idempotent downloader for FFmpeg 7.1 shared libraries (TorchCodec runtime requirement)
DEPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.deps/ffmpeg7"

if [ -f "$DEPS_DIR/lib/libavcodec.so.61" ]; then
    echo "FFmpeg 7.1 shared libraries already installed in $DEPS_DIR"
    exit 0
fi

echo "Installing FFmpeg 7.1 shared libraries into $DEPS_DIR..."
mkdir -p "$DEPS_DIR"

URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-lgpl-shared-7.1.tar.xz"
EXPECTED_SHA256="f5f0ad52c6ee28a222eb10838c231469a10ad325f84063d3bc0aadf08164b3ec"

TMP_ARCHIVE="$(mktemp --suffix=.tar.xz)"
trap 'rm -f "$TMP_ARCHIVE"' EXIT

echo "Downloading $URL..."
curl -sSL "$URL" -o "$TMP_ARCHIVE"

echo "Verifying SHA-256 checksum..."
ACTUAL_SHA256="$(sha256sum "$TMP_ARCHIVE" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: Checksum mismatch for downloaded archive!" >&2
    echo "Expected: $EXPECTED_SHA256" >&2
    echo "Got:      $ACTUAL_SHA256" >&2
    exit 1
fi

echo "Extracting archive into $DEPS_DIR..."
tar -xJ -f "$TMP_ARCHIVE" -C "$DEPS_DIR" --strip-components=1

echo "Verification: libavutil.so.59 present: $(ls -l "$DEPS_DIR/lib/libavutil.so.59")"
