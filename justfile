set dotenv-load
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

IMAGE_NAME := "resonance"
IMAGE_TAG  := env("IMAGE_TAG", "latest")
# Image name: resonance:latest or resonance:gpu-latest
IMAGE := if DEVICE == "cuda" { IMAGE_NAME + ":gpu-" + IMAGE_TAG } else { IMAGE_NAME + ":" + IMAGE_TAG }
# Archive name
TAR_FILE := if DEVICE == "cuda" { IMAGE_NAME + "_gpu_" + IMAGE_TAG + ".tar.gz" } else { IMAGE_NAME + "_" + IMAGE_TAG + ".tar.gz" }

DEVICE := env("DEVICE", "cpu")
PYTORCH_BACKEND := env("PYTORCH_BACKEND")
GPU_FLAGS := if DEVICE == "cuda" { "--gpus all" } else { "" }

# Show available recipes
default:
    @{{just_executable()}} --list

# Install dev dependencies
dev-deps:
    @command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
    @[[ -d .venv ]] || uv venv
    @if [[ -n "{{PYTORCH_BACKEND}}" ]]; then \
        uv pip install -r pyproject.toml --group dev --torch-backend={{PYTORCH_BACKEND}}; \
    else \
        uv pip install -r pyproject.toml --group dev; \
    fi
    @echo "Dev env ready. Start server with: just dev"

# Run server locally
dev:
    @command -v ffmpeg >/dev/null 2>&1 || { echo "ffmpeg not found."; exit 1; }
    .venv/bin/uvicorn server:app --reload

# Run pytest
test:
    .venv/bin/pytest tests/

# Run ruff
check:
    .venv/bin/ruff check server.py tests/

# Build image
build *ARGS:
    @if [[ -n "{{PYTORCH_BACKEND}}" ]]; then \
        docker build --build-arg PYTORCH_BACKEND={{PYTORCH_BACKEND}} -t {{IMAGE}} {{ARGS}} .; \
    else \
        docker build -t {{IMAGE}} {{ARGS}} .; \
    fi

# Run container
run:
    docker run -it --rm --name resonance {{GPU_FLAGS}} --env-file .env -p 8000:8000 {{IMAGE}}

# Save image to compressed archive
save:
    docker save {{IMAGE}} | gzip > {{TAR_FILE}}
    @echo "Saved to {{TAR_FILE}}"
