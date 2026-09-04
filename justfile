set dotenv-load
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set windows-shell := ["powershell.exe", "-c"]

export PYTHONUTF8 := "1"

IMAGE_NAME := "resonance"
IMAGE_TAG  := env("IMAGE_TAG", "latest")
# Image name: resonance:latest or resonance:gpu-latest
IMAGE := if DEVICE == "cuda" { IMAGE_NAME + ":gpu-" + IMAGE_TAG } else { IMAGE_NAME + ":" + IMAGE_TAG }
# Archive name
TAR_FILE := if DEVICE == "cuda" { IMAGE_NAME + "_gpu_" + IMAGE_TAG + ".tar.gz" } else { IMAGE_NAME + "_" + IMAGE_TAG + ".tar.gz" }

DEVICE := env("DEVICE", "cpu")
PYTORCH_BACKEND := env("PYTORCH_BACKEND", "")
GPU_FLAGS := if DEVICE == "cuda" { "--gpus all" } else { "" }

RESONANCE_PORT := env("RESONANCE_PORT", "8000")

TORCH_BACKEND_ARG := if PYTORCH_BACKEND != "" { "--torch-backend=" + PYTORCH_BACKEND } else { "" }
BUILD_ARG := if PYTORCH_BACKEND != "" { "--build-arg PYTORCH_BACKEND=" + PYTORCH_BACKEND } else { "" }

# Show available recipes
default:
    @{{just_executable()}} --list

# Install dev dependencies
dev-deps:
    {{ if path_exists(".venv") == "true" { "" } else { "uv venv" } }}
    uv pip install -r pyproject.toml --group dev {{TORCH_BACKEND_ARG}}

# Run server locally
dev:
    uv run uvicorn server:app --reload --reload-dir server.py --reload-dir core --reload-dir stt --reload-dir tts --reload-dir public --port {{RESONANCE_PORT}}

# Run pytest
test *ARGS:
    uv run pytest tests/ --base-url "http://localhost:{{RESONANCE_PORT}}" {{ARGS}}

# Run ruff
check *ARGS:
    uv run ruff check server.py core/ stt/ tts/ tests/ {{ARGS}}

# Build image
build *ARGS:
    docker build {{BUILD_ARG}} -t {{IMAGE}} {{ARGS}} .

HF_CACHE_DIR        := env("HF_HOME", env("HOME", env("USERPROFILE", "~")) + "/.cache/huggingface")
TORCH_CACHE_DIR     := env("TORCH_HOME", env("HOME", env("USERPROFILE", "~")) + "/.cache/torch")
GIGAAM_CACHE_DIR    := env("GIGAAM_CACHE_DIR", env("HOME", env("USERPROFILE", "~")) + "/.cache/gigaam")
SHERPA_CACHE_DIR    := env("SHERPA_HOME", env("HOME", env("USERPROFILE", "~")) + "/.cache/sherpa")
RESONANCE_CACHE_DIR := env("RESONANCE_CACHE_DIR", env("HOME", env("USERPROFILE", "~")) + "/.cache/resonance")

# Run container
run *ARGS:
    @mkdir -p "{{HF_CACHE_DIR}}" "{{TORCH_CACHE_DIR}}" "{{GIGAAM_CACHE_DIR}}" "{{SHERPA_CACHE_DIR}}" "{{RESONANCE_CACHE_DIR}}"
    docker run -it --rm --name resonance {{GPU_FLAGS}} \
        --env-file .env \
        -v "{{HF_CACHE_DIR}}":/home/resonance/.cache/huggingface \
        -v "{{TORCH_CACHE_DIR}}":/home/resonance/.cache/torch \
        -v "{{GIGAAM_CACHE_DIR}}":/home/resonance/.cache/gigaam \
        -v "{{SHERPA_CACHE_DIR}}":/home/resonance/.cache/sherpa \
        -v "{{RESONANCE_CACHE_DIR}}":/home/resonance/.cache/resonance \
        -e RESONANCE_PORT={{RESONANCE_PORT}} \
        -p {{RESONANCE_PORT}}:{{RESONANCE_PORT}} \
        {{ARGS}} \
        {{IMAGE}}

# Save image to compressed archive
save:
    docker save {{IMAGE}} | gzip > {{TAR_FILE}}
    @echo "Saved to {{TAR_FILE}}"

# Generate favicon.ico and apple-touch-icon.png
icons:
    @command -v resvg >/dev/null 2>&1 || { echo "resvg not found."; exit 1; }
    @command -v magick >/dev/null 2>&1 || { echo "magick (ImageMagick) not found."; exit 1; }
    @mkdir -p build/favicons
    resvg -w 16 -h 16 public/icon.svg build/favicons/16.png
    resvg -w 32 -h 32 public/icon.svg build/favicons/32.png
    resvg -w 48 -h 48 public/icon.svg build/favicons/48.png
    magick -background none build/favicons/16.png build/favicons/32.png build/favicons/48.png public/favicon.ico
    resvg -w 180 -h 180 public/icon.svg public/apple-touch-icon.png
    @echo "Web icons generated with pixel-perfect resvg vector rendering."

# Build macOS menu bar app
build-macos:
    bash scripts/build-icns.sh
    rm -rf build/Resonance.app
    mkdir -p build/Resonance.app/Contents/{MacOS,Resources}
    cp build/AppIcon.icns build/Resonance.app/Contents/Resources/
    cp build/StatusBarIcon*.png build/Resonance.app/Contents/Resources/
    swiftc -O src/swift/main.swift src/swift/CaptureEngine.swift -o build/Resonance.app/Contents/MacOS/Resonance
    cp src/swift/Info.plist build/Resonance.app/Contents/Info.plist
    # Pin the Designated Requirement to the Bundle ID instead of the binary's cdhash.
    # Without this, macOS TCC re-prompts for Screen Recording on every recompile because
    # ad-hoc signing embeds a cdhash-based DR that changes with each build.
    codesign --force --deep --sign "-" \
        --requirements '=designated => identifier "com.resonance.app"' \
        build/Resonance.app
    @echo "Built and signed: build/Resonance.app"
    mkdir -p ~/Applications
    ln -sfn "$PWD/build/Resonance.app" ~/Applications/Resonance.app
    touch ~/Applications/Resonance.app # Force macOS Icon Cache refresh
    @echo "Added to Launchpad"
