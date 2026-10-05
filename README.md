# Resonance

Unified Speech-to-Text (STT) and Text-to-Speech (TTS) API Server.

## Features

- **STT**: GigaAM-v3 (RU), Whisper Turbo (EN), and IBM Granite (with speaker diarization) (EN)
- **TTS**: Russian Silero v5 voices and English Kokoro voices
- **i18n**: Interface available in English, Russian, Chinese

## Demo

![Demo](https://raw.githubusercontent.com/y9938/assets/main/resonance/demo.gif)

## Local

### Deps

- [**FFmpeg**](https://ffmpeg.org) — audio decoding and streaming backend
- [**Node.js 22.12+**](https://nodejs.org/) and npm — frontend development and build
- [**uv**](https://docs.astral.sh/uv/) — fast Python package and project manager

#### macOS

```bash
./scripts/install-macos.sh
```

This creates `.env` if missing, configures the device and installs development dependencies.

#### Linux

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

FFmpeg: system shared libs (v4–v7) or run `./scripts/download_ffmpeg7.sh`

#### Windows

```powershell
powershell -ExecutionPolicy ByPass -File scripts/download_ffmpeg7.ps1
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
winget install Microsoft.VCRedist.2015+.x64 -e
```

### Run

Use `./r` on Linux/macOS or `.\r` in Windows PowerShell.

```bash
./r dev-deps
npm run build
./r serve-local
```

Run `dev-deps` again when dependencies change, and `npm run build` after frontend changes.
For development with automatic reload, use `./r dev`.

Open http://localhost:8000, or the port set by `RESONANCE_PORT`.
Models are loaded on first use, so initial STT/TTS requests may take longer.

See `./r -h` for tasks and [frontend notes](src/web/README.md) for development details.

On **macOS** you can build `Resonance.app` menu bar app via:

```bash
./r build-macos
```

The app is installed in `~/Applications/Resonance.app`.
Rebuild the app if you move the repository folder.

If you want to run from terminal with live logs:

```bash
~/Applications/Resonance.app/Contents/MacOS/Resonance
```

## Docker

```bash
./r build
./r run
```

Frontend assets are built into the image; Node is used only during the build.
Local-file STT and System Audio require the local launcher and are unavailable in Docker.

Model caches are shared with the host. On Linux, run as a non-root user.
Enforcing SELinux may require [bind-mount label configuration](https://docs.docker.com/engine/storage/bind-mounts/#configure-the-selinux-label).

## Configuration

`.env` in the repository is optional; existing environment variables take
precedence. Out of the box Resonance runs with CPU defaults.

- **CUDA**: set `DEVICE=cuda`, leave `PYTORCH_BACKEND=` for PyPI default or set a specific backend like `cu126`
- **CPU**: by default lightweight CPU-only wheels
- **macOS**: configured via setup script
- **Port**: set `RESONANCE_PORT` (default `8000`)
- **CORS**: set `RESONANCE_CORS_ORIGINS` only for custom origins; default follows `RESONANCE_PORT`

See [.env.example](.env.example) for available overrides.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | Public configuration including TTS `language -> voice` catalog |
| `/api/models` | GET | List backend/model status plus TTS catalog |
| `/api/context/tail` | GET | Get recent recognized conversation context for active session |
| `/api/jobs` | GET | List current session jobs (compact DTO); query `limit` (default 60), `offset`; JSON includes `has_more`, `next_offset` |
| `/api/jobs/stt` | POST | Start STT job, returns `job_id` |
| `/api/jobs/stt/local` | POST | Local launcher only: JSON `{"path":"/absolute/media.wav"}`, same STT query options and job lifecycle |
| `/api/jobs/live/start` | POST | Start session-scoped microphone live STT, returns `job_id` |
| `/api/jobs/live/{job_id}/chunk` | POST | Append one ordered audio chunk to a live microphone job |
| `/api/jobs/live/{job_id}/stop` | POST | Flush buffered microphone speech and complete a live job |
| `/api/jobs/tts` | POST | Start TTS job with `text`, `language`, `voice_id`; returns `job_id` |
| `/api/jobs/{job_id}` | GET | Get job status/result (session-scoped) |
| `/api/jobs/{job_id}/events` | GET | Stream job events (SSE, session-scoped) |
| `/api/jobs/{job_id}/cancel` | POST | Cancel active job (session-scoped) |
| `/api/stream/download` | GET | Download TTS audio |
| `/api/system-audio/start` | POST | Local launcher only: start host-wide system-audio capture, optionally including the microphone |
| `/api/system-audio/stop` | POST | Local launcher only: stop capture, flush recognized speech, and complete the same job |
