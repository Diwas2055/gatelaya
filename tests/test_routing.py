"""Phase 2 tests: RoutingPolicy config, GateLayaRouter pre-call model routing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from gatelaya.audit import DecisionRecord, InMemoryAuditSink
from gatelaya.calibration import TemperatureMap
from gatelaya.errors import GuardrailConfigurationError
from gatelaya.questions import COMPLEXITY_OPTIONS, TASK_OPTIONS, build_routing_questions
from gatelaya.routing import GateLayaRouter, RoutingPolicy

from .conftest import FakeAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTING_CONFIG = PROJECT_ROOT / "proxy_config_routing.yaml"

ROUTING_ENV_VARS = (
    "GATELAYA_ROUTE_ENABLED",
    "GATELAYA_ROUTE_DEFAULT",
    "GATELAYA_ROUTE_SENSITIVE",
    "GATELAYA_ROUTE_SENSITIVE_THRESHOLD",
    "GATELAYA_ROUTE_CONFIDENCE_THRESHOLD",
    "GATELAYA_ROUTE_TIERS",
    "GATELAYA_ROUTING_PATH",
    "GATELAYA_CALIBRATION_PATH",
    "GATELAYA_DATABASE_URL",
)


def routing_answers(
    *,
    task: str = "generate",
    complexity: str = "hard",
    complexity_confidence: float = 0.95,
    sensitive: float = 0.01,
) -> dict[str, Any]:
    """Laya-shaped answers for the three routing questions."""
    return {
        "task": {"choice": task, "confidence": 0.9},
        "complexity": {"choice": complexity, "confidence": complexity_confidence},
        "sensitive": {"noul": sensitive},
    }


def make_router(
    policy: RoutingPolicy | None = None,
    answers: dict[str, Any] | None = None,
    *,
    agent: FakeAgent | None = None,
    audit: InMemoryAuditSink | None = None,
    **kwargs: Any,
) -> tuple[GateLayaRouter, FakeAgent, InMemoryAuditSink]:
    """Factory: ``router, agent, sink = make_router(...)`` with a fake agent."""
    real_agent = agent if agent is not None else FakeAgent(
        answers if answers is not None else routing_answers()
    )
    sink = audit if audit is not None else InMemoryAuditSink()
    router = GateLayaRouter(policy=policy, agent=real_agent, audit=sink, **kwargs)
    return router, real_agent, sink


def make_data(content: str = "Summarize the quarterly report.", model: str = "gpt-4o-mini") -> dict:
    """Sample LiteLLM pre-call payload with a model and one user message."""
    return {"model": model, "messages": [{"role": "user", "content": content}]}


async def run_route(router: GateLayaRouter, data: dict) -> dict | None:
    """Invoke the router pre-call hook with neutral LiteLLM context."""
    return await router.async_pre_call_hook(None, None, data, "acompletion")


# ---------------------------------------------------------------- policy config


def test_policy_defaults() -> None:
    """Arrange/act: default policy. Assert: enabled, empty tiers, documented thresholds."""
    policy = RoutingPolicy()
    assert policy.enabled is True
    assert policy.tiers == {}
    assert policy.default_model is None
    assert policy.sensitive_model is None
    assert policy.sensitive_threshold == 0.85
    assert policy.confidence_threshold == 0.70


def test_policy_extra_field_forbidden() -> None:
    """Arrange/act: unexpected field. Assert: extra='forbid' raises."""
    with pytest.raises(ValidationError):
        RoutingPolicy(bogus_field=1)  # type: ignore[call-arg]


def test_policy_rejects_unknown_tier_key() -> None:
    """Arrange/act: tier key outside {trivial, moderate, hard}. Assert: ValidationError."""
    with pytest.raises(ValidationError, match="unknown tier keys"):
        RoutingPolicy(tiers={"medium": "mid-tier"})


def test_policy_rejects_tier_key_case_mismatch() -> None:
    """Arrange/act: capitalized tier key. Assert: rejected (keys are exact)."""
    with pytest.raises(ValidationError, match="unknown tier keys"):
        RoutingPolicy(tiers={"Hard": "frontier"})


@pytest.mark.parametrize("field", ["sensitive_threshold", "confidence_threshold"])
@pytest.mark.parametrize("bad_value", [-0.01, 1.5])
def test_policy_rejects_out_of_range_threshold(field: str, bad_value: float) -> None:
    """Arrange/act: threshold outside [0,1]. Assert: pydantic ValidationError."""
    with pytest.raises(ValidationError):
        RoutingPolicy(**{field: bad_value})


def test_policy_yaml_roundtrip(tmp_path: Path) -> None:
    """Arrange: policy with overrides. Act: to_yaml -> from_yaml. Assert: equal."""
    original = RoutingPolicy(
        enabled=False,
        tiers={"trivial": "cheap-mini", "hard": "frontier"},
        default_model="mid-tier",
        sensitive_model="secure-model",
        sensitive_threshold=0.9,
        confidence_threshold=0.6,
    )
    path = tmp_path / "routing.yaml"
    # Act
    original.to_yaml(path)
    loaded = RoutingPolicy.from_yaml(path)
    # Assert
    assert loaded.model_dump(mode="json") == original.model_dump(mode="json")


def test_policy_from_yaml_missing_file_raises(tmp_path: Path) -> None:
    """Arrange/act: nonexistent path. Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="config file not found"):
        RoutingPolicy.from_yaml(tmp_path / "nope.yaml")


def test_policy_from_yaml_bad_tier_key_wrapped(tmp_path: Path) -> None:
    """Arrange: YAML with unknown tier key. Act: from_yaml. Assert: config error."""
    path = tmp_path / "bad.yaml"
    path.write_text("tiers:\n  medium: mid-tier\n", encoding="utf-8")
    with pytest.raises(GuardrailConfigurationError, match="invalid config"):
        RoutingPolicy.from_yaml(path)


def test_policy_from_yaml_non_mapping_root_raises(tmp_path: Path) -> None:
    """Arrange: YAML list root. Act: from_yaml. Assert: config error."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(GuardrailConfigurationError, match="must be a mapping"):
        RoutingPolicy.from_yaml(path)


# -------------------------------------------------------------- env overrides


def test_router_env_overrides_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: GATELAYA_ROUTE_* env vars, no explicit policy. Act: construct router.
    Assert: every override landed on router.policy."""
    for name in ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GATELAYA_ROUTE_ENABLED", "false")
    monkeypatch.setenv("GATELAYA_ROUTE_DEFAULT", "mid-tier")
    monkeypatch.setenv("GATELAYA_ROUTE_SENSITIVE", "frontier")
    monkeypatch.setenv("GATELAYA_ROUTE_SENSITIVE_THRESHOLD", "0.5")
    monkeypatch.setenv("GATELAYA_ROUTE_CONFIDENCE_THRESHOLD", "0.9")
    monkeypatch.setenv("GATELAYA_ROUTE_TIERS", '{"trivial": "cheap-mini"}')
    # Act
    router, _, _ = make_router(policy=None)
    # Assert
    policy = router.policy
    assert policy.enabled is False
    assert policy.default_model == "mid-tier"
    assert policy.sensitive_model == "frontier"
    assert policy.sensitive_threshold == 0.5
    assert policy.confidence_threshold == 0.9
    assert policy.tiers == {"trivial": "cheap-mini"}


def test_explicit_policy_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: env set but policy passed explicitly. Act: construct.
    Assert: env NOT applied (overrides are policy=None path only)."""
    monkeypatch.setenv("GATELAYA_ROUTE_DEFAULT", "env-model")
    router, _, _ = make_router(policy=RoutingPolicy(default_model="explicit-model"))
    assert router.policy.default_model == "explicit-model"


def test_route_default_env_empty_string_means_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: GATELAYA_ROUTE_DEFAULT=''. Act: construct without policy.
    Assert: empty string clears the model (keeps original at decision time)."""
    for name in ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GATELAYA_ROUTE_DEFAULT", "")
    router, _, _ = make_router(policy=None)
    assert router.policy.default_model is None


def test_route_tiers_env_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: GATELAYA_ROUTE_TIERS that is not JSON. Act: construct.
    Assert: GuardrailConfigurationError naming the variable."""
    monkeypatch.setenv("GATELAYA_ROUTE_TIERS", "not-json{")
    with pytest.raises(GuardrailConfigurationError, match="GATELAYA_ROUTE_TIERS"):
        GateLayaRouter(agent=FakeAgent(), audit=InMemoryAuditSink())


def test_route_tiers_env_non_object_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: GATELAYA_ROUTE_TIERS JSON array. Act: construct.
    Assert: GuardrailConfigurationError."""
    monkeypatch.setenv("GATELAYA_ROUTE_TIERS", '["trivial"]')
    with pytest.raises(GuardrailConfigurationError, match="GATELAYA_ROUTE_TIERS"):
        GateLayaRouter(agent=FakeAgent(), audit=InMemoryAuditSink())


def test_routing_path_env_yaml_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Arrange: GATELAYA_ROUTING_PATH pointing at a policy YAML. Act: construct
    without a policy. Assert: policy loaded from the file."""
    for name in ROUTING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "routing.yaml"
    RoutingPolicy(tiers={"hard": "frontier"}, default_model="mid-tier").to_yaml(path)
    monkeypatch.setenv("GATELAYA_ROUTING_PATH", str(path))
    # Act
    router, _, _ = make_router(policy=None)
    # Assert
    assert router.policy.tiers == {"hard": "frontier"}
    assert router.policy.default_model == "mid-tier"


def test_env_threshold_out_of_range_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: GATELAYA_ROUTE_CONFIDENCE_THRESHOLD=1.5. Act: construct.
    Assert: pydantic ValidationError (env values revalidated)."""
    monkeypatch.setenv("GATELAYA_ROUTE_CONFIDENCE_THRESHOLD", "1.5")
    with pytest.raises(ValidationError):
        GateLayaRouter(agent=FakeAgent(), audit=InMemoryAuditSink())


# ------------------------------------------------------------- questions builder


def test_routing_questions_shape() -> None:
    """Arrange/act: build_routing_questions(). Assert: three questions, right types
    and option counts (task=7, complexity=3)."""
    questions = build_routing_questions()
    assert set(questions) == {"task", "complexity", "sensitive"}
    assert questions["task"]["type"] == "choice"
    assert questions["complexity"]["type"] == "choice"
    assert questions["sensitive"]["type"] == "noul"
    assert list(questions["task"]["criteria"]) == list(TASK_OPTIONS)
    assert list(questions["complexity"]["criteria"]) == list(COMPLEXITY_OPTIONS)
    assert len(questions["task"]["criteria"]) == 7
    assert len(questions["complexity"]["criteria"]) == 3


# ----------------------------------------------------------------- tier routing


async def test_tier_rewrite_hard_to_frontier() -> None:
    """Arrange: complexity=hard, tiers maps hard->frontier. Act: hook.
    Assert: data['model'] rewritten to frontier, reason 'tier'."""
    policy = RoutingPolicy(tiers={"hard": "frontier"})
    router, agent, _ = make_router(policy, routing_answers(complexity="hard"))
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "frontier"
    assert data["model"] == "frontier"
    assert data["gatelaya_routing"]["reason"] == "tier"
    assert agent.call_count == 1
    assert set(agent.last_questions) == {"task", "complexity", "sensitive"}


async def test_tier_rewrite_trivial_to_cheap() -> None:
    """Arrange: complexity=trivial, tiers maps trivial->cheap-mini. Act: hook.
    Assert: model rewritten to cheap-mini."""
    policy = RoutingPolicy(tiers={"trivial": "cheap-mini", "hard": "frontier"})
    router, _, _ = make_router(policy, routing_answers(complexity="trivial"))
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "cheap-mini"
    assert data["gatelaya_routing"]["complexity"] == "trivial"


async def test_no_tier_match_keeps_original() -> None:
    """Arrange: complexity=moderate with no moderate entry. Act: hook.
    Assert: original model kept, reason 'no-tier-match'."""
    policy = RoutingPolicy(tiers={"hard": "frontier"})
    router, _, sink = make_router(policy, routing_answers(complexity="moderate"))
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "gpt-4o-mini"
    assert data["gatelaya_routing"]["reason"] == "no-tier-match"
    assert sink.records[0].action_taken == "gpt-4o-mini"


async def test_empty_tiers_keep_original() -> None:
    """Arrange: default policy (no tiers). Act: hook. Assert: original model kept."""
    router, _, _ = make_router(RoutingPolicy())
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "gpt-4o-mini"
    assert data["gatelaya_routing"]["reason"] == "no-tier-match"


# ------------------------------------------------------------ sensitive override


async def test_sensitive_override_wins_over_tier() -> None:
    """Arrange: sensitive_p=0.95 >= 0.85 with sensitive_model set (tier also matches).
    Act: hook. Assert: sensitive model chosen, reason 'sensitive'."""
    policy = RoutingPolicy(tiers={"hard": "frontier"}, sensitive_model="secure-model")
    router, _, sink = make_router(policy, routing_answers(complexity="hard", sensitive=0.95))
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "secure-model"
    decision = data["gatelaya_routing"]
    assert decision["reason"] == "sensitive"
    assert decision["sensitive"] is True
    assert decision["sensitive_p"] == pytest.approx(0.95, abs=0.01)
    rec = sink.records[0]
    assert rec.check == "routing"
    assert rec.detail["reason"] == "sensitive"


async def test_sensitive_without_sensitive_model_falls_through_to_tier() -> None:
    """Arrange: sensitive request but sensitive_model=None. Act: hook.
    Assert: tier logic applies (frontier), not low-confidence or sensitive."""
    policy = RoutingPolicy(tiers={"hard": "frontier"})
    router, _, _ = make_router(policy, routing_answers(complexity="hard", sensitive=0.95))
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "frontier"
    assert data["gatelaya_routing"]["reason"] == "tier"
    assert data["gatelaya_routing"]["sensitive"] is True


async def test_calibration_softens_sensitive_decision() -> None:
    """Arrange: raw sensitive p=0.86 >= 0.85, but noul temperature 2.0 pulls the
    calibrated prob below threshold. Act: calibrated run vs uncalibrated control.
    Assert: control takes the sensitive model, calibrated run falls through to tier."""
    answers = routing_answers(complexity="hard", sensitive=0.86)
    policy = RoutingPolicy(tiers={"hard": "frontier"}, sensitive_model="secure-model")
    cal_router, _, _ = make_router(policy, answers, temperatures=TemperatureMap(noul={"default": 2.0}))
    ctrl_router, _, _ = make_router(policy, answers)
    cal_data, ctrl_data = make_data(), make_data()
    # Act
    cal_result = await run_route(cal_router, cal_data)
    ctrl_result = await run_route(ctrl_router, ctrl_data)
    # Assert
    assert ctrl_result is not None and ctrl_result["model"] == "secure-model"
    assert cal_result is not None and cal_result["model"] == "frontier"
    assert cal_data["gatelaya_routing"]["sensitive"] is False


# ------------------------------------------------------------- low confidence


async def test_low_confidence_uses_default_model() -> None:
    """Arrange: complexity confidence 0.40 < 0.70, default_model set. Act: hook.
    Assert: default model chosen, reason 'low-confidence'."""
    policy = RoutingPolicy(default_model="mid-tier", tiers={"moderate": "frontier"})
    router, _, sink = make_router(
        policy, routing_answers(complexity="moderate", complexity_confidence=0.40)
    )
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "mid-tier"
    decision = data["gatelaya_routing"]
    assert decision["reason"] == "low-confidence"
    assert decision["confidence"] == pytest.approx(0.40, abs=0.01)
    rec = sink.records[0]
    assert rec.probs["raw"] == pytest.approx(0.40, abs=0.01)
    assert rec.probs["calibrated"] == pytest.approx(0.40, abs=0.01)


async def test_low_confidence_default_none_keeps_original() -> None:
    """Arrange: low confidence, default_model=None. Act: hook.
    Assert: original model kept, reason 'low-confidence'."""
    router, _, _ = make_router(
        RoutingPolicy(), routing_answers(complexity="moderate", complexity_confidence=0.30)
    )
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "gpt-4o-mini"
    assert data["gatelaya_routing"]["reason"] == "low-confidence"


async def test_choice_temperature_softens_confidence() -> None:
    """Arrange: raw confidence 0.75 >= 0.70 but choice temperature 3.0 pulls the
    calibrated prob below threshold. Act: calibrated run vs control.
    Assert: control routes by tier, calibrated run falls back to default_model."""
    answers = routing_answers(complexity="hard", complexity_confidence=0.75)
    policy = RoutingPolicy(tiers={"hard": "frontier"}, default_model="mid-tier")
    cal_router, _, _ = make_router(
        policy, answers, temperatures=TemperatureMap(choice={"3": 3.0})
    )
    ctrl_router, _, _ = make_router(policy, answers)
    cal_data, ctrl_data = make_data(), make_data()
    # Act
    cal_result = await run_route(cal_router, cal_data)
    ctrl_result = await run_route(ctrl_router, ctrl_data)
    # Assert
    assert ctrl_result is not None and ctrl_result["model"] == "frontier"
    assert cal_result is not None and cal_result["model"] == "mid-tier"
    assert cal_data["gatelaya_routing"]["reason"] == "low-confidence"


# ------------------------------------------------------------------- disabled


async def test_disabled_policy_skips_agent_and_leaves_data_untouched() -> None:
    """Arrange: enabled=False. Act: hook. Assert: None returned, agent never
    called, data unchanged (no model rewrite, no stash)."""
    router, agent, sink = make_router(RoutingPolicy(enabled=False))
    data = make_data()
    snapshot = dict(data)
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is None
    assert agent.call_count == 0
    assert data == snapshot
    assert "gatelaya_routing" not in data
    assert sink.records == []


# -------------------------------------------------------------- failure modes


async def test_agent_error_keeps_original_model_and_audits() -> None:
    """Arrange: agent raises. Act: hook. Assert: no exception, original model kept,
    routing_error audit record written."""
    sink = InMemoryAuditSink()
    router, agent, _ = make_router(
        RoutingPolicy(tiers={"hard": "frontier"}),
        agent=FakeAgent(raise_error=RuntimeError("model exploded")),
        audit=sink,
    )
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert data["model"] == "gpt-4o-mini"
    assert data["gatelaya_routing"]["reason"] == "agent-error"
    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec.check == "routing_error"
    assert rec.mode == "route"
    assert rec.action_taken == "gpt-4o-mini"
    assert rec.blocked is False
    assert rec.probs == {}
    assert agent.call_count == 1


async def test_malformed_answer_keeps_original_model_and_audits() -> None:
    """Arrange: complexity answer not in options. Act: hook. Assert: treated like
    an agent error — original model, routing_error record, no exception."""
    sink = InMemoryAuditSink()
    router, _, _ = make_router(
        RoutingPolicy(tiers={"hard": "frontier"}),
        answers={"complexity": "not-an-option"},
        audit=sink,
    )
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert data["model"] == "gpt-4o-mini"
    assert sink.records[0].check == "routing_error"


async def test_missing_complexity_answer_falls_to_low_confidence() -> None:
    """Arrange: answers without 'complexity'. Act: hook. Assert: original model
    kept via the low-confidence path (confidence 0.0)."""
    router, _, _ = make_router(
        RoutingPolicy(default_model="mid-tier"),
        answers={"sensitive": {"noul": 0.01}, "task": {"choice": "summarize", "confidence": 0.9}},
    )
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "mid-tier"
    assert data["gatelaya_routing"]["reason"] == "low-confidence"
    assert data["gatelaya_routing"]["complexity"] == "unknown"


async def test_audit_sink_failure_never_breaks_routing() -> None:
    """Arrange: sink that raises on record. Act: tier run. Assert: model still
    rewritten, hook returns data (audit errors are swallowed by design)."""

    class BrokenSink:
        async def record(self, rec: DecisionRecord) -> None:
            raise RuntimeError("sink down")

    router, _, _ = make_router(
        RoutingPolicy(tiers={"hard": "frontier"}), audit=BrokenSink()  # type: ignore[arg-type]
    )
    data = make_data()
    # Act
    result = await run_route(router, data)
    # Assert
    assert result is not None
    assert result["model"] == "frontier"


async def test_empty_scan_text_short_circuits_without_agent_call() -> None:
    """Arrange: payload without usable text. Act: hook. Assert: None, agent untouched."""
    router, agent, _ = make_router(RoutingPolicy(tiers={"hard": "frontier"}))
    # Act
    assert await run_route(router, {}) is None
    assert await run_route(router, {"model": "m", "messages": []}) is None
    assert await run_route(router, {"model": "m", "messages": [{"role": "user", "content": "  "}]}) is None
    # Assert
    assert agent.call_count == 0


# ------------------------------------------------------------ stash + audit shape


async def test_routing_decision_stash_fields_present() -> None:
    """Arrange: tier run. Act: hook. Assert: data['gatelaya_routing'] carries every
    RoutingDecision field."""
    policy = RoutingPolicy(tiers={"hard": "frontier"})
    router, _, _ = make_router(policy, routing_answers(task="summarize", complexity="hard"))
    data = make_data()
    # Act
    await run_route(router, data)
    # Assert
    stash = data["gatelaya_routing"]
    assert set(stash) == {
        "original_model",
        "chosen_model",
        "task",
        "complexity",
        "sensitive",
        "sensitive_p",
        "confidence",
        "reason",
        "latency_ms",
    }
    assert stash["original_model"] == "gpt-4o-mini"
    assert stash["chosen_model"] == "frontier"
    assert stash["task"] == "summarize"
    assert stash["latency_ms"] >= 0


async def test_audit_record_written_with_routing_check_and_route_mode() -> None:
    """Arrange: tier run. Act: hook. Assert: one DecisionRecord with check='routing',
    mode='route', blocked/masked False, detail carrying task/complexity/reason."""
    sink = InMemoryAuditSink()
    policy = RoutingPolicy(tiers={"hard": "frontier"})
    router, _, _ = make_router(policy, routing_answers(task="reason", complexity="hard"), audit=sink)
    data = make_data()
    # Act
    await run_route(router, data)
    # Assert
    assert len(sink.records) == 1
    rec = sink.records[0]
    assert rec.check == "routing"
    assert rec.mode == "route"
    assert rec.action_taken == "frontier"
    assert rec.blocked is False
    assert rec.masked is False
    assert rec.language == "english"
    assert len(rec.input_sha256) == 64
    assert set(rec.probs) == {"raw", "calibrated"}
    assert rec.detail["task"] == "reason"
    assert rec.detail["complexity"] == "hard"
    assert rec.detail["sensitive"] == "false"
    assert rec.detail["reason"] == "tier"


async def test_scan_text_is_system_plus_last_user_message() -> None:
    """Arrange: system + user messages. Act: hook. Assert: agent scanned the joined
    text (same rule as the Phase 1 guardrail)."""
    router, agent, _ = make_router(RoutingPolicy(tiers={"hard": "frontier"}))
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "System part."},
            {"role": "user", "content": "User part."},
        ],
    }
    # Act
    await run_route(router, data)
    # Assert
    assert agent.last_state["text"] == "System part.\nUser part."


# ------------------------------------------------- litellm integration surface


def test_router_module_resolves_like_litellm() -> None:
    """Arrange/act: import the dotted path litellm uses in proxy_config_routing.yaml.
    Assert: module resolves and exposes GateLayaRouter."""
    module = importlib.import_module("custom_guardrail.gatelaya.router")
    assert module.GateLayaRouter is GateLayaRouter


def test_instantiation_as_litellm_would(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: no policy/agent/audit — exactly what litellm passes at boot.
    Act: construct with guardrail_name/event_hook/default_on. Assert: expected
    attributes, defaults wired, laya not loaded."""
    monkeypatch.delenv("GATELAYA_ROUTING_PATH", raising=False)
    monkeypatch.delenv("GATELAYA_DATABASE_URL", raising=False)
    monkeypatch.delenv("GATELAYA_CALIBRATION_PATH", raising=False)
    # Act
    router = GateLayaRouter(
        guardrail_name="gatelaya-router",
        event_hook=["pre_call"],
        default_on=True,
    )
    # Assert
    assert router.guardrail_name == "gatelaya-router"
    assert router.event_hook == ["pre_call"]
    assert router.default_on is True
    assert router.policy.enabled is True
    assert type(router.agent).__name__ == "LayaRouterAgent"
    assert "laya" not in sys.modules


def test_proxy_config_routing_yaml_parses() -> None:
    """Arrange/act: safe_load proxy_config_routing.yaml. Assert: model_list with the
    three tier models and a guardrails list with both entries."""
    raw = yaml.safe_load(ROUTING_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert {m["model_name"] for m in raw["model_list"]} == {"cheap-mini", "mid-tier", "frontier"}
    guardrails = raw["guardrails"]
    assert {g["guardrail_name"] for g in guardrails} == {"gatelaya", "gatelaya-router"}


def test_proxy_config_routing_guardrail_entries_shape() -> None:
    """Arrange/act: load guardrails section. Assert: gatelaya keeps pre/post, the
    router entry is pre_call only and points at the router shim."""
    raw = yaml.safe_load(ROUTING_CONFIG.read_text(encoding="utf-8"))
    guardrails = {g["guardrail_name"]: g["litellm_params"] for g in raw["guardrails"]}
    assert guardrails["gatelaya"]["guardrail"] == "custom_guardrail.gatelaya.guardrail.GateLayaGuardrail"
    assert guardrails["gatelaya"]["mode"] == ["pre_call", "post_call"]
    assert guardrails["gatelaya-router"]["guardrail"] == "custom_guardrail.gatelaya.router.GateLayaRouter"
    assert guardrails["gatelaya-router"]["mode"] == ["pre_call"]
    assert guardrails["gatelaya-router"]["default_on"] is True
