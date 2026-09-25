"""Multi-tenant per-key policies: match a LiteLLM key identity, override config."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from .config import ALL_CHECKS, Action, GateLayaConfig
from .errors import GuardrailConfigurationError

logger = logging.getLogger("gatelaya")

_HEX = re.compile(r"^[0-9a-f]{12,64}$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")

_MATCH_PRIORITY = ("key_alias", "user_id", "team_id", "api_key_hash")


def sha256_hex(value: str) -> str:
    """Hex SHA-256 of a plaintext API key (never store or log the plaintext)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_api_key(value: str | None) -> str | None:
    """Treat an already-hashed key as-is; hash anything else. None stays None."""
    if value is None:
        return None
    if _HEX64.fullmatch(value):
        return value.lower()
    return sha256_hex(value)


class PolicyMatch(BaseModel):
    """Identity fields; every specified field must match (AND), None = unspecified."""

    model_config = ConfigDict(extra="forbid")

    key_alias: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    #: Full sha256 hex or a >=12-char hex prefix; plaintext keys are rejected.
    api_key_hash: str | None = None

    @field_validator("api_key_hash")
    @classmethod
    def _valid_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if not _HEX.fullmatch(lowered):
            raise ValueError("api_key_hash must be 12-64 hex chars (sha256, not plaintext)")
        return lowered

    @model_validator(mode="after")
    def _at_least_one_field(self) -> PolicyMatch:
        if not self.specified():
            raise ValueError("match must specify at least one identity field")
        return self

    def specified(self) -> dict[str, str]:
        """Specified fields as {name: value} (None values dropped)."""
        return {
            name: getattr(self, name)
            for name in _MATCH_PRIORITY
            if getattr(self, name) is not None
        }


class RoutingOverrides(BaseModel):
    """Per-tenant router overrides (applied only when the router is active)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    tiers: dict[str, str] | None = None
    default_model: str | None = None
    sensitive_model: str | None = None
    sensitive_threshold: float | None = None
    confidence_threshold: float | None = None


class PolicyOverrides(BaseModel):
    """Config overrides for a matched key: dicts merge, scalars/lists replace."""

    model_config = ConfigDict(extra="forbid")

    thresholds: dict[str, float] | None = None
    actions: dict[str, Action] | None = None
    enabled_checks: list[str] | None = None
    fail_open: bool | None = None
    routing: RoutingOverrides | None = None

    @field_validator("thresholds")
    @classmethod
    def _validate_thresholds(cls, value: dict[str, float]) -> dict[str, float]:
        for check, threshold in value.items():
            if check not in ALL_CHECKS:
                raise ValueError(f"unknown check: {check!r}; allowed: {list(ALL_CHECKS)}")
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold for {check!r} must be in [0, 1], got {threshold}")
        return value

    @field_validator("enabled_checks")
    @classmethod
    def _validate_checks(cls, value: list[str]) -> list[str]:
        unknown = set(value) - set(ALL_CHECKS)
        if unknown:
            raise ValueError(f"unknown checks: {sorted(unknown)}; allowed: {list(ALL_CHECKS)}")
        return value

    def empty(self) -> bool:
        """True when no override field is specified."""
        return all(v is None for v in self.model_dump(exclude_none=True).values())


class TenantPolicy(BaseModel):
    """One per-key policy: identity match plus the overrides it activates."""

    model_config = ConfigDict(extra="forbid")

    match: PolicyMatch
    overrides: PolicyOverrides = Field(default_factory=PolicyOverrides)


class PolicySet(BaseModel):
    """Ordered policy list; first full match wins at request time."""

    model_config = ConfigDict(extra="forbid")

    policies: list[TenantPolicy] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicySet:
        """Load a policy set from a YAML file (missing/invalid = config error)."""
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GuardrailConfigurationError(f"policy file not found: {p}") from exc
        except yaml.YAMLError as exc:
            raise GuardrailConfigurationError(f"invalid YAML in {p}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise GuardrailConfigurationError(f"policy root in {p} must be a mapping")
        try:
            return cls(**raw)
        except PydanticValidationError as exc:
            raise GuardrailConfigurationError(f"invalid policy file in {p}: {exc}") from exc

    def resolve(self, user_api_key_dict: Any) -> tuple[TenantPolicy | None, str | None]:
        """First policy matching this key's identity; (None, None) = global config."""
        identities: dict[str, str | None] = {
            "key_alias": getattr(user_api_key_dict, "key_alias", None),
            "user_id": getattr(user_api_key_dict, "user_id", None),
            "team_id": getattr(user_api_key_dict, "team_id", None),
            "api_key_hash": normalize_api_key(getattr(user_api_key_dict, "api_key", None)),
        }
        for policy in self.policies:
            specified = policy.match.specified()
            if all(_field_matches(identities.get(name), expected) for name, expected in specified.items()):
                return policy, _policy_label(identities, specified)
        return None, None


def _field_matches(actual: str | None, expected: str) -> bool:
    """Compare one identity field; hash fields match on prefix, others exactly."""
    if actual is None:
        return False
    if len(expected) >= 12 and _HEX.fullmatch(expected) and _HEX64.fullmatch(actual):
        return actual.startswith(expected)
    return actual == expected


def _policy_label(identities: dict[str, str | None], specified: dict[str, str]) -> str:
    """Short non-plaintext audit label, e.g. 'key_alias=team-a'."""
    for name in _MATCH_PRIORITY:
        if name in specified:
            actual = identities.get(name) or ""
            value = actual[:16] if name == "api_key_hash" else actual
            return f"{name}={value}"
    return "policy"  # unreachable: PolicyMatch requires >=1 field


def reload_policies(
    path: Path | None, current: PolicySet | None
) -> tuple[PolicySet | None, bool]:
    """(set, ok): unset path -> (None, True); load failure keeps `current`."""
    if path is None:
        return None, True
    try:
        return PolicySet.from_yaml(path), True
    except Exception as exc:
        logger.warning(
            "GateLaya: could not load policies from %s, keeping previous: %s", path, exc
        )
        return current, False


def apply_overrides(config: GateLayaConfig, overrides: PolicyOverrides) -> GateLayaConfig:
    """Return a new config with this policy's overrides applied (dicts merge)."""
    if overrides is None or overrides.empty():
        return config
    update: dict[str, Any] = {}
    if overrides.thresholds:
        update["thresholds"] = {**config.thresholds, **overrides.thresholds}
    if overrides.actions:
        update["actions"] = {**config.actions, **overrides.actions}
    if overrides.enabled_checks is not None:
        update["enabled_checks"] = list(overrides.enabled_checks)
    if overrides.fail_open is not None:
        update["fail_open"] = overrides.fail_open
    # model_copy skips revalidation; every field was validated by PolicyOverrides.
    return config.model_copy(update=update) if update else config
