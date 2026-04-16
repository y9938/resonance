# Resonance

Unified Speech-to-Text (STT) and Text-to-Speech (TTS) API Server.

## Features

- **STT**: GigaAM-v3 model for Russian speech recognition
- **TTS**: Silero v5_ru with 27 voices
- **i18n**: Interface available in English, Russian, Chinese

## Demo

![Demo](https://raw.githubusercontent.com/y9938/assets/main/resonance/demo.gif)

## Quick Start

```bash
cp .env.example .env
just build
just run
```

Open http://localhost:8000

**GPU:** set `DEVICE=cuda` in `.env` before building.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET | Public configuration |
| `/api/models` | GET | List available models |
| `/api/jobs` | GET | List current session jobs (compact DTO); query `limit` (default 60), `offset`; JSON includes `has_more`, `next_offset` |
| `/api/jobs/stt` | POST | Start STT job, returns `job_id` |
| `/api/jobs/tts` | POST | Start TTS job, returns `job_id` |
| `/api/jobs/{job_id}` | GET | Get job status/result (session-scoped) |
| `/api/jobs/{job_id}/events` | GET | Stream job events (SSE, session-scoped) |
| `/api/jobs/{job_id}/cancel` | POST | Cancel active job (session-scoped) |
| `/api/stream/download` | GET | Download TTS audio |

## F5 Recovery Model

- Frontend stores only active job IDs in `localStorage`:
  - `resonance_stt_active_job_id`
  - `resonance_tts_active_job_id`
- Drawer jobs list is loaded from `GET /api/jobs` (paginated with `offset` / `has_more`) and contains only jobs for the current browser session.
- Backend assigns `resonance_session_id` cookie and enforces ownership on `status/events/cancel` endpoints.
- After page reload, UI restores state via `GET /api/jobs/{job_id}` and continues progress via `/events`.
- Job data is in-memory on server (`JobRegistry`), so after server restart unknown `job_id` is cleared on client and UI resets to neutral state.

## Configuration

See `.env.example` for available options.
