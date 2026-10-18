"""FastAPI application factory and lifespan-managed model initialization."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .model_loader import ModelLoadError, load_artifacts
from .routes import router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifacts = None
        try:
            app.state.artifacts = load_artifacts(settings.model_dir)
            logger.info("TrustGuard inference artifacts loaded")
        except ModelLoadError:
            # Keep health and readiness endpoints available even if deployment is incomplete.
            logger.exception("TrustGuard inference artifacts are unavailable")
        yield
        app.state.artifacts = None

    app = FastAPI(
        title="TrustGuard AI API",
        version="1.0.0",
        description="Two-stage job-posting fraud classification and human-review recommendation API.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    return app


app = create_app()
