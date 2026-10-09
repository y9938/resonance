#!/bin/sh
REPO_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = x86_64 ]; then
    PATH="$REPO_DIR/.deps/macos-intel/bin:$PATH"
    export PATH
fi
exec uv run --script "$REPO_DIR/scripts/tasks.py" "$@"
