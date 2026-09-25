"""Read/write the GateLaya YAML config — the proxy hot-reloads the file on change."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from gatelaya.config import GateLayaConfig
from gatelaya.errors import GuardrailConfigurationError
from gatelaya.hotreload import env_flag

from ..dependencies import SettingsDep

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigView(BaseModel):
    """Current config file contents plus its path on disk."""

    config: dict[str, Any]
    path: str
    exists: bool


class ConfigPatch(BaseModel):
    """Partial update; omitted fields keep their current values."""

    model_config = ConfigDict(extra="forbid")

    thresholds: dict[str, float] | None = None
    actions: dict[str, str] | None = None
    enabled_checks: list[str] | None = None
    fail_open: bool | None = None


class WriteAck(BaseModel):
    """Write acknowledgement — hot reload picks the file up (~1s) unless disabled."""

    ok: bool
    path: str
    restart_required: bool
    hot_reload: bool


def load_config(path: Path) -> tuple[GateLayaConfig, bool]:
    """Load the config file, or built-in defaults when it does not exist."""
    if not path.exists():
        return GateLayaConfig(), False
    try:
        return GateLayaConfig.from_yaml(path), True
    except GuardrailConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("", response_model=ConfigView)
async def get_config(settings: SettingsDep) -> ConfigView:
    """Return the config file (or defaults) without touching disk."""
    path = Path(settings.config_path)
    cfg, exists = load_config(path)
    return ConfigView(config=cfg.model_dump(mode="json"), path=str(path), exists=exists)


@router.put("", response_model=WriteAck)
async def put_config(patch: ConfigPatch, settings: SettingsDep) -> WriteAck:
    """Apply a patch, validate it as GateLayaConfig, and write the YAML file."""
    path = Path(settings.config_path)
    base, _ = load_config(path)
    merged: dict[str, Any] = base.model_dump(mode="json")
    if patch.thresholds is not None:
        merged["thresholds"] = {**merged.get("thresholds", {}), **patch.thresholds}
    if patch.actions is not None:
        merged["actions"] = {**merged.get("actions", {}), **patch.actions}
    if patch.enabled_checks is not None:
        merged["enabled_checks"] = patch.enabled_checks
    if patch.fail_open is not None:
        merged["fail_open"] = patch.fail_open
    try:
        cfg = GateLayaConfig(**merged)
    except PydanticValidationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid config patch: {exc}") from exc
    cfg.to_yaml(path)
    hot_reload = env_flag("GATELAYA_HOT_RELOAD", True)
    return WriteAck(
        ok=True, path=str(path), restart_required=not hot_reload, hot_reload=hot_reload
    )
