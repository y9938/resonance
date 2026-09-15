# Resonance

Unified Speech-to-Text (STT) and Text-to-Speech (TTS) API Server.

## Features

- **STT**: GigaAM-v3 (RU), Distil-Whisper-v3 (EN), and IBM Granite (with speaker diarization) (EN)
- **TTS**: Russian Silero v5 voices and English Kokoro voices
- **i18n**: Interface available in English, Russian, Chinese

## Demo

![Demo](https://raw.githubusercontent.com/y9938/assets/main/resonance/demo.gif)

## Configuration

`.env` is optional. Out of the box Resonance runs with CPU defaults.

Create `.env` only if you want to override defaults:

```bash
touch .env
```

- **CUDA**: set `DEVICE=cuda`, leave `PYTORCH_BACKEND=` for PyPI default or set a specific backend like `cu128`
- **CPU**: by default lightweight CPU-only wheels
- **macOS**: use the setup script below
- **Port**: set `RESONANCE_PORT` (default `8000`)
- **CORS**: set `RESONANCE_CORS_ORIGINS` only for custom origins; default follows `RESONANCE_PORT`

See `.env.example` for all available overrides and reference values.

## Docker

```bash
just build
just run
```

## Local

### Deps

- [**FFmpeg**](https://ffmpeg.org/download.html) — audio decoding and streaming backend
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/) — fast Python package and project manager
- [**just**](https://github.com/casey/just#installation) — command runner

install it on your platform

#### macOS

```bash
./scripts/install-macos.sh
```

This creates `.env` if missing, configures the device, installs dev dependencies

#### Linux

manually choose your methods for all deps, but:
System FFmpeg (v4–v7 shared libs) or run `./scripts/download_ffmpeg7.sh`

#### Windows

```powershell
powershell -ExecutionPolicy ByPass -File scripts/download_ffmpeg7.ps1
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv tool install rust-just
winget install Microsoft.VCRedist.2015+.x64 -e
```

### Run

```bash
just dev-deps
just dev
```

On **macOS** you can build `Resonance.app` menu bar app via:

```bash
just build-macos
```

and if you want to run from terminal with live logs:

```bash
./build/Resonance.app/Contents/MacOS/Resonance
```

## Note

Open `http://localhost:${RESONANCE_PORT}` (default: http://localhost:8000)

Models are loaded lazily on first real STT/TTS use. Startup does not pre-download or pre-load model weights, so the first request to a specific backend may take noticeably longer.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | Public configuration including TTS `language -> voice` catalog |
| `/api/models` | GET | List backend/model status plus TTS catalog |
| `/api/context/tail` | GET | Get recent recognized conversation context for active session |
| `/api/jobs` | GET | List current session jobs (compact DTO); query `limit` (default 60), `offset`; JSON includes `has_more`, `next_offset` |
| `/api/jobs/stt` | POST | Start STT job, returns `job_id` |
| `/api/jobs/live/start` | POST | Start session-scoped microphone live STT, returns `job_id` |
| `/api/jobs/live/{job_id}/chunk` | POST | Append one ordered audio chunk to a live microphone job |
| `/api/jobs/live/{job_id}/stop` | POST | Flush buffered microphone speech and complete a live job |
| `/api/jobs/tts` | POST | Start TTS job with `text`, `language`, `voice_id`; returns `job_id` |
| `/api/jobs/{job_id}` | GET | Get job status/result (session-scoped) |
| `/api/jobs/{job_id}/events` | GET | Stream job events (SSE, session-scoped) |
| `/api/jobs/{job_id}/cancel` | POST | Cancel active job (session-scoped) |
| `/api/stream/download` | GET | Download TTS audio |
| `/api/system-audio/start` | POST | Start host-wide system-audio capture, optionally including the microphone |
| `/api/system-audio/stop` | POST | Stop capture, flush recognized speech, and complete the same job |

## IPC

Local socket for desktop integration:
- POSIX: `$XDG_RUNTIME_DIR/resonance.sock` (fallback `~/.cache/resonance/ipc.sock`)
- Windows: `\\.\pipe\resonance-ipc`
- Custom path: `RESONANCE_IPC_PATH` in `.env`
