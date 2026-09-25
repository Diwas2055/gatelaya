"""FastAPI application: dashboard API plus the static frontend mount."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from .db import init_db
from .dependencies import require_token
from .routers import calibration_api, config_api, decisions, review, stats
from .settings import get_settings

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create the dashboard tables on boot."""
    await init_db()
    yield


def _ensure_static_dir() -> None:
    """Create the frontend asset directory (owned by the frontend) when missing."""
    if not STATIC_DIR.is_dir():
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        (STATIC_DIR / ".gitkeep").touch()


def create_app() -> FastAPI:
    """Build the app: bearer auth on every /api route except the health probe."""
    app = FastAPI(title="GateLaya Dashboard API", lifespan=lifespan)

    @app.get("/api/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe — never requires the bearer token."""
        return {"status": "ok"}

    auth = [Depends(require_token)]
    for router in (
        stats.router,
        decisions.router,
        review.router,
        config_api.router,
        calibration_api.router,
    ):
        app.include_router(router, dependencies=auth)

    _ensure_static_dir()
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app()


def run() -> None:
    """Serve the dashboard (`gatelaya-dashboard` entry point)."""
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
