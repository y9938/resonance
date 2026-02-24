# Resonance

Unified Speech-to-Text (STT) and Text-to-Speech (TTS) API Server.

## Features

- **STT**: GigaAM-v3 model for Russian speech recognition
- **TTS**: Silero v5_ru with 5 voices (aidar, baya, kseniya, xenia, eugene)
- **Long Text Support**: Automatic chunking, no artificial input limits (configurable)
- **Streaming**: Real-time progress via Server-Sent Events (SSE)

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
| `/api/stream/stt` | POST | Stream STT progress (SSE) |
| `/api/stream/tts` | POST | Stream TTS progress (SSE) |
| `/api/stream/download` | GET | Download TTS audio |

## Configuration

See `.env.example` for available options.
