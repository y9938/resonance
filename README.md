# Resonance

Unified Speech-to-Text (STT) and Text-to-Speech (TTS) API Server.

## Features

- **STT**: GigaAM-v3 (RU), Distil-Whisper-v3 (EN), and IBM Granite (with speaker diarization) (EN)
- **TTS**: Russian Silero v5 voices and English Kokoro voices
- **i18n**: Interface available in English, Russian, Chinese

## Demo

![Demo](https://raw.githubusercontent.com/y9938/assets/main/resonance/demo.gif)

## Configuration

Create `.env` before running:

```bash
cp .env.example .env
```

See `.env.example` for available options.

Before installing dependencies, configure `.env`:
- **CUDA**: set `DEVICE=cuda`, leave `PYTORCH_BACKEND=` for PyPI default or set a specific backend like `cu128`
- **CPU**: set `DEVICE=cpu` and `PYTORCH_BACKEND=cpu`
- **macOS**: use the setup script below
- **Port**: set `RESONANCE_PORT` (default `8000`)
- **CORS**: set `RESONANCE_CORS_ORIGINS` only for custom origins; default follows `RESONANCE_PORT`

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

manually choose your method

#### Windows

```powershell
winget install "FFmpeg (Essentials Build)"
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
| `/api/jobs` | GET | List current session jobs (compact DTO); query `limit` (default 60), `offset`; JSON includes `has_more`, `next_offset` |
| `/api/jobs/stt` | POST | Start STT job, returns `job_id` |
| `/api/jobs/tts` | POST | Start TTS job with `text`, `language`, `voice_id`; returns `job_id` |
| `/api/jobs/{job_id}` | GET | Get job status/result (session-scoped) |
| `/api/jobs/{job_id}/events` | GET | Stream job events (SSE, session-scoped) |
| `/api/jobs/{job_id}/cancel` | POST | Cancel active job (session-scoped) |
| `/api/stream/download` | GET | Download TTS audio |
| `/api/system-audio/start` | POST | Start internal system & microphone audio capture |
| `/api/system-audio/stop` | POST | Stop capture and launch interleaved dual-stream STT job |

## F5 Recovery Model

- Frontend stores only active job IDs in `localStorage`:
  - `resonance_stt_active_job_id`
  - `resonance_tts_active_job_id`
- Drawer jobs list is loaded from `GET /api/jobs` (paginated with `offset` / `has_more`) and contains only jobs for the current browser session.
- Backend assigns `resonance_session_id` cookie and enforces ownership on `status/events/cancel` endpoints.
- After page reload, UI restores state via `GET /api/jobs/{job_id}` and continues progress via `/events`.
- Job data is in-memory on server (`JobRegistry`), so after server restart unknown `job_id` is cleared on client and UI resets to neutral state.
