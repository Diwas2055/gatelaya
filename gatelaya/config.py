"""Pydantic configuration model for GateLaya."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from .errors import GuardrailConfigurationError

Action = Literal["allow", "mask", "block", "flag"]
CheckName = Literal["pii", "injection", "toxicity", "secret_leak"]
ModeName = Literal["pre_call", "post_call"]

ALL_CHECKS: tuple[str, ...] = ("pii", "injection", "toxicity", "secret_leak")

DEFAULT_THRESHOLDS: dict[str, float] = {
    "pii": 0.85,
    "injection": 0.90,
    "toxicity": 0.90,
    "secret_leak": 0.85,
}

DEFAULT_ACTIONS: dict[str, Action] = {
    "pii": "mask",
    "injection": "block",
    "toxicity": "block",
    "secret_leak": "block",
}

DEFAULT_ENGLISH_CHECKPOINT = "convaiinnovations/laya"
DEFAULT_MULTILINGUAL_CHECKPOINT = "convaiinnovations/laya-multilingual"


class GateLayaConfig(BaseModel):
    """Thresholds, actions, routing, and runtime flags for the guardrail."""

    model_config = ConfigDict(extra="forbid")

    thresholds: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    actions: dict[str, Action] = Field(default_factory=lambda: dict(DEFAULT_ACTIONS))
    mode: list[ModeName] = Field(default_factory=lambda: ["pre_call", "post_call"])
    english_checkpoint: str = DEFAULT_ENGLISH_CHECKPOINT
    multilingual_checkpoint: str = DEFAULT_MULTILINGUAL_CHECKPOINT
    enabled_checks: list[CheckName] = Field(
        default_factory=lambda: ["pii", "injection", "toxicity", "secret_leak"]
    )
    calibration_path: Path | None = None
    audit_enabled: bool = True
    fail_open: bool = True

    @field_validator("thresholds")
    @classmethod
    def _merge_and_validate_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        merged = {**DEFAULT_THRESHOLDS, **value}
        for check, threshold in merged.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold for {check!r} must be in [0, 1], got {threshold}")
        return merged

    @field_validator("actions")
    @classmethod
    def _merge_actions(cls, value: dict[str, Action]) -> dict[str, Action]:
        return {**DEFAULT_ACTIONS, **value}

    @field_validator("enabled_checks")
    @classmethod
    def _validate_checks(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(ALL_CHECKS)
        if unknown:
            raise ValueError(f"unknown checks: {sorted(unknown)}; allowed: {list(ALL_CHECKS)}")
        return value

    def threshold(self, check: str) -> float:
        """Return the confidence threshold for a check."""
        try:
            return self.thresholds[check]
        except KeyError:
            raise GuardrailConfigurationError(f"missing threshold for check {check!r}") from None

    def action(self, check: str) -> Action:
        """Return the configured action for a check."""
        try:
            return self.actions[check]
        except KeyError:
            raise GuardrailConfigurationError(f"missing action for check {check!r}") from None

    @classmethod
    def from_yaml(cls, path: str | Path) -> GateLayaConfig:
        """Load configuration from a YAML file."""
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GuardrailConfigurationError(f"config file not found: {p}") from exc
        except yaml.YAMLError as exc:
            raise GuardrailConfigurationError(f"invalid YAML in {p}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise GuardrailConfigurationError(f"config root in {p} must be a mapping")
        try:
            return cls(**raw)
        except PydanticValidationError as exc:
            raise GuardrailConfigurationError(f"invalid config in {p}: {exc}") from exc

    def to_yaml(self, path: str | Path) -> None:
        """Write this configuration to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
