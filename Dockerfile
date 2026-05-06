FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ARG PYTORCH_BACKEND=

WORKDIR /app

# Install git for pip dependencies from git repositories
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_TORCH_BACKEND=${PYTORCH_BACKEND} \
    UV_LINK_MODE=copy

COPY pyproject.toml .

RUN uv pip install --system -r pyproject.toml

# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r resonance && useradd -r -m -g resonance -s /bin/false resonance

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=resonance:resonance server.py .
COPY --chown=resonance:resonance stt/ ./stt/
COPY --chown=resonance:resonance tts/ ./tts/
COPY --chown=resonance:resonance public/ ./public/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

USER resonance

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "-g", "--"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--no-access-log"]
