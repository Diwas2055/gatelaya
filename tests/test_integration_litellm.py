"""Integration checks: LiteLLM boot path for the custom guardrail pack.

Verifies the package resolves the way litellm's proxy does it
(`custom_guardrail.gatelaya.guardrail.GateLayaGuardrail`), that the guardrail
can be constructed with only litellm's kwargs, and that proxy_config.yaml
parses with the expected guardrails section.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from gatelaya.guardrail import GateLayaGuardrail

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROXY_CONFIG = PROJECT_ROOT / "proxy_config.yaml"

try:
    from litellm.integrations.custom_guardrail import CustomGuardrail as LitellmCustomGuardrail
    from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
    from litellm.types.guardrails import LitellmParams

    LITELLM_IMPORTABLE = True
except ImportError:  # pragma: no cover
    LITELLM_IMPORTABLE = False


# ------------------------------------------------------------------ imports


def test_custom_guardrail_module_resolves() -> None:
    """Arrange/act: import the dotted path litellm uses in proxy_config.yaml.
    Assert: module resolves and exposes GateLayaGuardrail."""
    module = importlib.import_module("custom_guardrail.gatelaya.guardrail")
    assert module.GateLayaGuardrail is GateLayaGuardrail


@pytest.mark.skipif(not LITELLM_IMPORTABLE, reason="litellm not importable")
def test_guardrail_subclasses_litellm_custom_guardrail() -> None:
    """Arrange/act: class hierarchy. Assert: real litellm base class used."""
    assert issubclass(GateLayaGuardrail, LitellmCustomGuardrail)


def test_instantiation_as_litellm_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: no config/agent/audit — exactly what litellm passes at boot.
    Act: construct with guardrail_name/event_hook/default_on.
    Assert: no exception, expected attributes, defaults wired, laya not loaded."""
    monkeypatch.delenv("GATELAYA_CONFIG_PATH", raising=False)
    monkeypatch.delenv("GATELAYA_DATABASE_URL", raising=False)
    monkeypatch.delenv("GATELAYA_CALIBRATION_PATH", raising=False)
    # Act
    guard = GateLayaGuardrail(
        guardrail_name="gatelaya",
        event_hook=["pre_call", "post_call"],
        default_on=True,
    )
    # Assert
    assert guard.guardrail_name == "gatelaya"
    assert guard.event_hook == ["pre_call", "post_call"]
    assert guard.default_on is True
    assert guard.config.enabled_checks == ["pii", "injection", "toxicity", "secret_leak"]
    assert type(guard.agent).__name__ == "LayaRouterAgent"
    assert "laya" not in sys.modules


# ------------------------------------------------------------ proxy_config.yaml


def test_proxy_config_yaml_parses() -> None:
    """Arrange/act: safe_load proxy_config.yaml. Assert: mapping with guardrails."""
    raw = yaml.safe_load(PROXY_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert "guardrails" in raw
    assert "model_list" in raw


def test_proxy_config_guardrails_section_shape() -> None:
    """Arrange/act: load guardrails section. Assert: gatelaya entry shape matches
    what litellm expects (guardrail_name + litellm_params)."""
    raw = yaml.safe_load(PROXY_CONFIG.read_text(encoding="utf-8"))
    guardrails = raw["guardrails"]
    assert isinstance(guardrails, list) and guardrails
    entry = next(g for g in guardrails if g.get("guardrail_name") == "gatelaya")
    params = entry["litellm_params"]
    assert params["guardrail"] == "custom_guardrail.gatelaya.guardrail.GateLayaGuardrail"
    assert params["mode"] == ["pre_call", "post_call"]
    assert params["default_on"] is True


# ------------------------------------------- litellm handler initialization


@pytest.mark.skipif(not LITELLM_IMPORTABLE, reason="litellm not importable")
def test_initialize_through_litellm_handler() -> None:
    """Arrange: proxy_config-shaped guardrail dict + LitellmParams.
    Act: InMemoryGuardrailHandler.initialize_custom_guardrail.
    Assert: returns a live GateLayaGuardrail with litellm's kwargs applied."""
    import litellm

    handler = InMemoryGuardrailHandler()
    guardrail_entry: dict[str, Any] = {
        "guardrail_name": "gatelaya",
        "litellm_params": {
            "guardrail": "custom_guardrail.gatelaya.guardrail.GateLayaGuardrail",
            "mode": ["pre_call", "post_call"],
            "default_on": True,
        },
    }
    params = LitellmParams(**guardrail_entry["litellm_params"])
    # Act
    instance = handler.initialize_custom_guardrail(
        guardrail_entry,
        "custom_guardrail.gatelaya.guardrail.GateLayaGuardrail",
        params,
        config_file_path=str(PROXY_CONFIG),
    )
    try:
        # Assert
        assert isinstance(instance, GateLayaGuardrail)
        assert instance.guardrail_name == "gatelaya"
        assert instance.event_hook == ["pre_call", "post_call"]
        assert instance.default_on is True
    finally:
        # keep litellm's global callback registry clean for other tests
        litellm.logging_callback_manager.remove_callback_from_all_lists(instance)
