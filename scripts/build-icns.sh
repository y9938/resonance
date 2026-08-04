#!/usr/bin/env bash
set -euo pipefail

SRC="src/assets/AppIcon.svg"
OUT="build/AppIcon.icns"
STATUS_SRC="src/assets/StatusBarIcon.svg"
STATUS_OUT="build/StatusBarIcon.png"

# Build AppIcon.icns
if [[ ! -f "$OUT" || "$SRC" -nt "$OUT" ]]; then
    ICONSET="build/AppIcon.iconset"
    mkdir -p "$ICONSET"

    resvg -w 1024 -h 1024 "$SRC" "$ICONSET/icon_512x512@2x.png"

    for sz in 16 32 128 256 512; do
        sips -z "$sz" "$sz" "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
        sips -z "$((sz * 2))" "$((sz * 2))" "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
    done

    iconutil -c icns "$ICONSET" -o "$OUT"
    echo "Successfully built: $OUT"
fi

# Build StatusBarIcon.png
if [[ ! -f "$STATUS_OUT" || "$STATUS_SRC" -nt "$STATUS_OUT" ]]; then
    resvg -w 36 -h 36 "$STATUS_SRC" build/StatusBarIcon@2x.png
    sips -z 18 18 build/StatusBarIcon@2x.png --out "$STATUS_OUT" >/dev/null
    echo "Successfully built status bar template icons."
fi
