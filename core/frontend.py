"""Static frontend delivery; development modules never proxy API requests."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


class FrontendFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if path.startswith("assets/")
            else "no-cache"
        )
        return response


def mount_frontend(app: FastAPI, root: Path) -> None:
    frontend = root / "src/web"
    development = os.environ.get("RESONANCE_FRONTEND_DEV") == "1"
    assets = frontend / "public" if development else root / "dist/web"
    if development:
        origin = os.environ.get("RESONANCE_VITE_ORIGIN", "http://127.0.0.1:5173")

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def development_index():
            html = (frontend / "index.html").read_text(encoding="utf-8")
            marker = '<script type="module" src="/main.ts"></script>'
            if marker not in html:
                raise RuntimeError("Frontend entry marker is missing from src/web/index.html")
            html = html.replace(
                marker,
                f'<script type="module" src="{origin}/@vite/client"></script>'
                f'<script type="module" src="{origin}/main.ts"></script>',
            )
            return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    if assets.exists():
        app.mount("/", FrontendFiles(directory=assets, html=True), name="static")
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def missing_frontend():
            return HTMLResponse(
                "Frontend build is missing. Run npm run build to build it.",
                status_code=503,
            )
