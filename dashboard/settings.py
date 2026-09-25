"""Dashboard settings loaded from GATELAYA_* environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    """Runtime settings for the dashboard API server."""

    model_config = ConfigDict(extra="forbid")

    database_url: str = "sqlite+aiosqlite:///./gatelaya_dashboard.db"
    config_path: str = "./gatelaya.yaml"
    calibration_path: str = "./calibration.json"
    token: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the environment (empty or unset means default)."""

        def env(name: str, default: object) -> object:
            return os.environ.get(name) or default

        return cls(
            database_url=env("GATELAYA_DATABASE_URL", cls.model_fields["database_url"].default),
            config_path=env("GATELAYA_CONFIG_PATH", cls.model_fields["config_path"].default),
            calibration_path=env(
                "GATELAYA_CALIBRATION_PATH", cls.model_fields["calibration_path"].default
            ),
            token=os.environ.get("GATELAYA_DASHBOARD_TOKEN") or None,
            host=env("GATELAYA_DASHBOARD_HOST", cls.model_fields["host"].default),
            port=env("GATELAYA_DASHBOARD_PORT", cls.model_fields["port"].default),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton (tests call get_settings.cache_clear())."""
    return Settings.from_env()
