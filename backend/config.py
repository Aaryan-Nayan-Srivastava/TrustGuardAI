"""Application configuration sourced from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,"
    "http://localhost:5173,http://127.0.0.1:5173"
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Model artifacts are always local, project-owned files."""

    project_root: Path
    model_dir: Path
    cors_origins: list[str]
    log_level: str

    @classmethod
    def from_environment(cls) -> "Settings":
        project_root = Path(__file__).resolve().parents[1]
        configured_model_dir = os.getenv("TRUSTGUARD_MODEL_DIR")
        model_dir = (
            Path(configured_model_dir).expanduser().resolve()
            if configured_model_dir
            else project_root / "models"
        )
        origins = [
            origin.strip()
            for origin in os.getenv("CORS_ALLOW_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
            if origin.strip()
        ]
        return cls(
            project_root=project_root,
            model_dir=model_dir,
            cors_origins=origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
