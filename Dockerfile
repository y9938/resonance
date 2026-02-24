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

# Pre-download models (speeds up startup)
RUN python -c "import torch; torch.hub.load('snakers4/silero-models', model='silero_tts', language='ru', speaker='v5_cis_base', trust_repo=True)"
RUN python -c "import gigaam; gigaam.load_model('v3_e2e_ctc')"

# -----------------------------------------------------------------------------
# Runtime stage
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r resonance && useradd -r -g resonance -s /bin/false resonance

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --chown=resonance:resonance --from=builder /root/.cache/gigaam /home/resonance/.cache/gigaam
COPY --chown=resonance:resonance --from=builder /root/.cache/torch /home/resonance/.cache/torch

COPY --chown=resonance:resonance server.py .
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
