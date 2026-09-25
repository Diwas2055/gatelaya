"""Tests for the single-checkpoint fine-tune override (model_path wiring)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gatelaya.agent import LayaRouterAgent
from gatelaya.config import GateLayaConfig
from gatelaya.guardrail import GateLayaGuardrail, _env_overrides

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval.py"


# ------------------------------------------------------------------ config


def test_config_round_trips_model_path(tmp_path: Path) -> None:
    """model_path survives model_dump/YAML and stays optional."""
    cfg = GateLayaConfig(model_path=tmp_path / "ft")
    data = cfg.model_dump()
    again = GateLayaConfig(**data)
    assert again.model_path == tmp_path / "ft"
    assert GateLayaConfig().model_path is None


def test_yaml_round_trip_model_path(tmp_path: Path) -> None:
    path = tmp_path / "gatelaya.yaml"
    GateLayaConfig(model_path=tmp_path / "ft").to_yaml(path)
    loaded = GateLayaConfig.from_yaml(path)
    assert loaded.model_path == tmp_path / "ft"


def test_env_override_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATELAYA_MODEL_PATH", "/models/gatelaya-ft")
    cfg = _env_overrides(GateLayaConfig())
    assert cfg.model_path == Path("/models/gatelaya-ft")
    monkeypatch.setenv("GATELAYA_MODEL_PATH", "")
    assert _env_overrides(GateLayaConfig()).model_path is None


def test_guardrail_router_gets_model_path() -> None:
    guard = GateLayaGuardrail(config=GateLayaConfig(model_path="/models/ft"))
    assert isinstance(guard.agent, LayaRouterAgent)
    assert guard.agent.model_path == "/models/ft"


# ------------------------------------------------------------------- agent


class _FakeAgent:
    """Stand-in for a laya Agent."""

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint

    def predict(self, state: dict, questions: dict) -> dict:
        return {"answers": {"model": self.checkpoint}}


class _FakeLaya:
    """Stand-in for the laya module: records load() calls."""

    def __init__(self) -> None:
        self.loaded: list[str] = []

    def load(self, checkpoint: str) -> _FakeAgent:
        self.loaded.append(checkpoint)
        return _FakeAgent(checkpoint)


@pytest.fixture
def fake_laya(monkeypatch: pytest.MonkeyPatch) -> _FakeLaya:
    fake = _FakeLaya()
    monkeypatch.setattr(LayaRouterAgent, "_laya", staticmethod(lambda: fake))
    return fake


def test_router_uses_single_model_for_both_buckets(fake_laya: _FakeLaya) -> None:
    router = LayaRouterAgent(model_path="/models/ft")
    en = router.predict({"text": "plain ascii"}, {})
    non_en = router.predict({"text": "नेपाली पाठ"}, {})
    assert en == {"answers": {"model": "/models/ft"}}
    assert non_en == {"answers": {"model": "/models/ft"}}
    # one shared instance for both buckets
    assert fake_laya.loaded == ["/models/ft"]
    assert router.route({"text": "plain ascii"}) == "english"
    assert router.route({"text": "नेपाली"}) == "multilingual"


def test_router_without_model_path_routes_normally(fake_laya: _FakeLaya) -> None:
    router = LayaRouterAgent()
    router.predict({"text": "ascii only"}, {})
    router.predict({"text": "non ascii हिंदी"}, {})
    assert fake_laya.loaded == [
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
    ]


# --------------------------------------------------------------- eval CLI


def test_eval_cli_parses_model_flag() -> None:
    spec = importlib.util.spec_from_file_location("gatelaya_eval_model_flag", EVAL_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--model", "/models/ft", "--agent", "laya"])
    assert args.model == "/models/ft"
    assert module.parse_args([]).model is None
