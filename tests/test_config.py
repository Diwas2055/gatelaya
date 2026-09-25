"""Tests for GateLayaConfig: defaults, validation, YAML roundtrip, env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gatelaya.audit import InMemoryAuditSink
from gatelaya.config import (
    DEFAULT_ACTIONS,
    DEFAULT_THRESHOLDS,
    GateLayaConfig,
)
from gatelaya.errors import GuardrailConfigurationError
from gatelaya.guardrail import GateLayaGuardrail, _env_overrides

from .conftest import FakeAgent

# ------------------------------------------------------------------ defaults


def test_default_thresholds_and_actions() -> None:
    """Arrange/act: build defaults. Assert: documented thresholds/actions."""
    # Arrange
    cfg = GateLayaConfig()
    # Act
    # Assert
    assert cfg.thresholds == DEFAULT_THRESHOLDS
    assert cfg.thresholds["pii"] == 0.85
    assert cfg.thresholds["injection"] == 0.90
    assert cfg.thresholds["toxicity"] == 0.90
    assert cfg.thresholds["secret_leak"] == 0.85
    assert cfg.actions == DEFAULT_ACTIONS
    assert cfg.actions["pii"] == "mask"
    assert cfg.actions["injection"] == "block"


def test_default_runtime_flags() -> None:
    """Arrange/act: defaults. Assert: fail-open, audit on, both modes, all checks."""
    cfg = GateLayaConfig()
    assert cfg.fail_open is True
    assert cfg.audit_enabled is True
    assert cfg.mode == ["pre_call", "post_call"]
    assert set(cfg.enabled_checks) == {"pii", "injection", "toxicity", "secret_leak"}
    assert cfg.calibration_path is None
    assert cfg.english_checkpoint == "convaiinnovations/laya"
    assert cfg.multilingual_checkpoint == "convaiinnovations/laya-multilingual"


def test_threshold_and_action_accessors() -> None:
    """Arrange/act: lookups. Assert: values returned for known checks."""
    cfg = GateLayaConfig()
    assert cfg.threshold("pii") == 0.85
    assert cfg.action("injection") == "block"


def test_threshold_missing_check_raises_config_error() -> None:
    """Arrange/act: unknown check lookup. Assert: GuardrailConfigurationError."""
    cfg = GateLayaConfig()
    with pytest.raises(GuardrailConfigurationError, match="missing threshold"):
        cfg.threshold("unknown")


def test_action_missing_check_raises_config_error() -> None:
    """Arrange/act: unknown check lookup. Assert: GuardrailConfigurationError."""
    cfg = GateLayaConfig()
    with pytest.raises(GuardrailConfigurationError, match="missing action"):
        cfg.action("unknown")


# --------------------------------------------------------------- validation


@pytest.mark.parametrize("bad_threshold", [-0.01, 1.5])
def test_out_of_range_threshold_rejected(bad_threshold: float) -> None:
    """Arrange/act: threshold outside [0,1]. Assert: pydantic ValidationError."""
    with pytest.raises(ValidationError):
        GateLayaConfig(thresholds={"pii": bad_threshold})


def test_invalid_action_value_rejected() -> None:
    """Arrange/act: unknown action literal. Assert: pydantic ValidationError."""
    with pytest.raises(ValidationError):
        GateLayaConfig(actions={"pii": "destroy"})  # type: ignore[dict-item]


def test_unknown_enabled_check_rejected() -> None:
    """Arrange/act: unknown check name. Assert: ValidationError listing the
    allowed CheckName literals (Literal validation fires before the custom
    'unknown checks' field validator)."""
    with pytest.raises(ValidationError) as excinfo:
        GateLayaConfig(enabled_checks=["pii", "hallucination"])  # type: ignore[list-item]
    message = str(excinfo.value)
    assert "hallucination" in message
    assert "secret_leak" in message


def test_extra_field_forbidden() -> None:
    """Arrange/act: unexpected field. Assert: extra='forbid' raises."""
    with pytest.raises(ValidationError):
        GateLayaConfig(bogus_field=1)  # type: ignore[call-arg]


def test_partial_thresholds_merge_with_defaults() -> None:
    """Arrange/act: partial thresholds dict. Assert: defaults merged in."""
    cfg = GateLayaConfig(thresholds={"pii": 0.5})
    assert cfg.thresholds["pii"] == 0.5
    assert cfg.thresholds["injection"] == 0.90


# ---------------------------------------------------------------- YAML I/O


def test_yaml_roundtrip(tmp_path: Path) -> None:
    """Arrange: config with overrides. Act: to_yaml -> from_yaml. Assert: equal."""
    original = GateLayaConfig(
        thresholds={"pii": 0.7, "injection": 0.8},
        actions={"pii": "flag"},
        enabled_checks=["pii", "injection"],
        fail_open=False,
        audit_enabled=False,
        mode=["pre_call"],
    )
    path = tmp_path / "gatelaya.yaml"
    # Act
    original.to_yaml(path)
    loaded = GateLayaConfig.from_yaml(path)
    # Assert
    assert loaded.thresholds == original.thresholds
    assert loaded.actions == original.actions
    assert loaded.enabled_checks == original.enabled_checks
    assert loaded.fail_open is False
    assert loaded.audit_enabled is False
    assert loaded.mode == ["pre_call"]
    assert loaded.model_dump(mode="json") == original.model_dump(mode="json")


def test_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    """Arrange/act: nonexistent path. Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="config file not found"):
        GateLayaConfig.from_yaml(tmp_path / "nope.yaml")


def test_from_yaml_invalid_action_raises(tmp_path: Path) -> None:
    """Arrange: YAML with bad action value. Act: from_yaml. Assert: config error."""
    path = tmp_path / "bad.yaml"
    path.write_text("actions:\n  pii: destroy\n", encoding="utf-8")
    # Act
    with pytest.raises(GuardrailConfigurationError, match="invalid config"):
        GateLayaConfig.from_yaml(path)


def test_from_yaml_non_mapping_root_raises(tmp_path: Path) -> None:
    """Arrange: YAML list root. Act: from_yaml. Assert: config error."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(GuardrailConfigurationError, match="must be a mapping"):
        GateLayaConfig.from_yaml(path)


def test_from_yaml_empty_file_returns_defaults(tmp_path: Path) -> None:
    """Arrange: empty YAML. Act: from_yaml. Assert: default config."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    cfg = GateLayaConfig.from_yaml(path)
    assert cfg.thresholds == DEFAULT_THRESHOLDS


def test_from_yaml_invalid_yaml_raises(tmp_path: Path) -> None:
    """Arrange: malformed YAML. Act: from_yaml. Assert: config error."""
    path = tmp_path / "broken.yaml"
    path.write_text("key: [unterminated\n", encoding="utf-8")
    with pytest.raises(GuardrailConfigurationError, match="invalid YAML"):
        GateLayaConfig.from_yaml(path)


# ------------------------------------------------------------- env overrides


def test_env_overrides_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: GATELAYA_* env vars. Act: _env_overrides. Assert: values applied."""
    monkeypatch.setenv("GATELAYA_THRESHOLD_PII", "0.5")
    monkeypatch.setenv("GATELAYA_ACTION_INJECTION", "flag")
    monkeypatch.setenv("GATELAYA_FAIL_OPEN", "false")
    monkeypatch.setenv("GATELAYA_ENABLED_CHECKS", "pii, injection")
    monkeypatch.setenv("GATELAYA_AUDIT_ENABLED", "0")
    monkeypatch.setenv("GATELAYA_CALIBRATION_PATH", "")
    # Act
    cfg = _env_overrides(GateLayaConfig())
    # Assert
    assert cfg.thresholds["pii"] == 0.5
    assert cfg.actions["injection"] == "flag"
    assert cfg.fail_open is False
    assert cfg.enabled_checks == ["pii", "injection"]
    assert cfg.audit_enabled is False
    assert cfg.calibration_path is None
    # defaults untouched
    assert cfg.thresholds["toxicity"] == 0.90


def test_guardrail_applies_env_overrides_when_config_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: env vars + no explicit config. Act: construct guardrail.
    Assert: deployment-path overrides landed on guardrail.config."""
    monkeypatch.delenv("GATELAYA_CONFIG_PATH", raising=False)
    monkeypatch.delenv("GATELAYA_DATABASE_URL", raising=False)
    monkeypatch.delenv("GATELAYA_CALIBRATION_PATH", raising=False)
    monkeypatch.setenv("GATELAYA_THRESHOLD_SECRET_LEAK", "0.4")
    monkeypatch.setenv("GATELAYA_ACTION_PII", "block")
    monkeypatch.setenv("GATELAYA_FAIL_OPEN", "yes")
    # Act
    guard = GateLayaGuardrail(
        config=None, agent=FakeAgent(), audit=InMemoryAuditSink()
    )
    # Assert
    assert guard.config.threshold("secret_leak") == 0.4
    assert guard.config.action("pii") == "block"
    assert guard.config.fail_open is True


def test_explicit_config_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: env var set but config passed explicitly.
    Act: construct guardrail. Assert: env NOT applied (overrides are
    config=None path only, per `_env_overrides` docstring)."""
    monkeypatch.setenv("GATELAYA_THRESHOLD_PII", "0.1")
    explicit = GateLayaConfig()
    # Act
    guard = GateLayaGuardrail(config=explicit, agent=FakeAgent(), audit=InMemoryAuditSink())
    # Assert
    assert guard.config.thresholds["pii"] == 0.85
