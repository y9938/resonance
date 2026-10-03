#!/bin/sh
exec uv run --script "$(dirname "$0")/scripts/tasks.py" "$@"
