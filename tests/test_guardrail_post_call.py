"""Tests for GateLayaGuardrail.async_post_call_success_hook (response scanning)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from gatelaya.audit import InMemoryAuditSink
from gatelaya.config import GateLayaConfig
from gatelaya.guardrail import GateLayaGuardrail

from .conftest import FakeAgent, clean_answers

try:  # real litellm response object
    from litellm import ModelResponse

    LITELLM_IMPORTABLE = True
except ImportError:  # pragma: no cover
    ModelResponse = None  # type: ignore[assignment,misc]
    LITELLM_IMPORTABLE = False


async def run_post(guard: GateLayaGuardrail, response: Any, data: dict | None = None) -> Any:
    """Invoke the post-call hook with neutral LiteLLM context."""
    return await guard.async_post_call_success_hook(
        data if data is not None else {}, None, response
    )


def text_response(text: str) -> dict:
    """Dict-shaped chat completion containing `text` as assistant content."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
    }


# ------------------------------------------------------------------ blocking


async def test_secret_leak_above_threshold_raises_400() -> None:
    """Arrange: secret_leak p=0.99 >= 0.85, action=block. Act: hook.
    Assert: HTTPException 400 naming the secret leak."""
    agent = FakeAgent(clean_answers(secret_leak={"noul": 0.99}))
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=InMemoryAuditSink())
    response = text_response("Your API key is sk-live-1234567890abcdef")
    # Act
    with pytest.raises(HTTPException) as excinfo:
        await run_post(guard, response)
    # Assert
    assert excinfo.value.status_code == 400
    assert "secret leak" in excinfo.value.detail
    assert "in response" in excinfo.value.detail


async def test_block_stashes_meta_and_audits_post_call() -> None:
    """Arrange: blocking run. Act: hook. Assert: HTTPException raised, decision
    meta stashed on data, audit records mode='post_call'."""
    sink = InMemoryAuditSink()
    agent = FakeAgent(clean_answers(secret_leak={"noul": 0.99}))
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=sink)
    data: dict = {}
    response = text_response("password: hunter2secret")
    # Act
    with pytest.raises(HTTPException):
        await run_post(guard, response, data)
    # Assert
    assert "gatelaya" in data
    blocked = next(d for d in data["gatelaya"]["decisions"] if d["check"] == "secret_leak")
    assert blocked["blocked"] is True
    assert {r.check for r in sink.records} == {"pii", "injection", "toxicity", "secret_leak"}
    for rec in sink.records:
        assert rec.mode == "post_call"
        assert rec.latency_ms >= 0
        assert len(rec.input_sha256) == 64


# ---------------------------------------------------------------------- clean


async def test_clean_response_passes(sample_response: dict) -> None:
    """Arrange: all-clean answers. Act: hook. Assert: returns None, no meta."""
    guard = GateLayaGuardrail(
        config=GateLayaConfig(), agent=FakeAgent(), audit=InMemoryAuditSink()
    )
    data: dict = {}
    # Act
    result = await run_post(guard, sample_response, data)
    # Assert
    assert result is None
    assert "gatelaya" not in data


async def test_dict_response_shape_handled() -> None:
    """Arrange: raw dict response (choices/message/content). Act: hook.
    Assert: text extracted, agent called exactly once."""
    agent = FakeAgent()
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=InMemoryAuditSink())
    # Act
    result = await run_post(guard, text_response("plain answer"))
    # Assert
    assert result is None
    assert agent.call_count == 1


async def test_empty_response_text_skips_agent() -> None:
    """Arrange: response with no/blank content. Act: hook. Assert: None and
    agent never called."""
    agent = FakeAgent()
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=InMemoryAuditSink())
    # Act
    assert await run_post(guard, {"choices": []}) is None
    assert await run_post(guard, text_response("   ")) is None
    # Assert
    assert agent.call_count == 0


@pytest.mark.skipif(not LITELLM_IMPORTABLE, reason="litellm not importable")
async def test_real_model_response_handled() -> None:
    """Arrange: real litellm.ModelResponse. Act: hook (clean).
    Assert: text extracted and hook returns None."""
    agent = FakeAgent()
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=InMemoryAuditSink())
    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "A perfectly fine answer."}}]
    )
    # Act
    result = await run_post(guard, response)
    # Assert
    assert result is None
    assert agent.call_count == 1


@pytest.mark.skipif(not LITELLM_IMPORTABLE, reason="litellm not importable")
async def test_real_model_response_block_raises_400() -> None:
    """Arrange: real ModelResponse leaking a secret. Act: hook.
    Assert: HTTPException 400."""
    agent = FakeAgent(clean_answers(secret_leak={"noul": 0.98}))
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=InMemoryAuditSink())
    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "token sk-secret-xyz"}}]
    )
    # Act
    with pytest.raises(HTTPException) as excinfo:
        await run_post(guard, response)
    # Assert
    assert excinfo.value.status_code == 400


# ---------------------------------------------------------------- flag paths


async def test_flag_path_attaches_metadata_without_raising() -> None:
    """Arrange: secret_leak action=flag above threshold. Act: hook. Assert: no
    exception, decision meta stashed on data, audit records flag."""
    sink = InMemoryAuditSink()
    cfg = GateLayaConfig(actions={"secret_leak": "flag"})
    agent = FakeAgent(clean_answers(secret_leak={"noul": 0.99}))
    guard = GateLayaGuardrail(config=cfg, agent=agent, audit=sink)
    data: dict = {}
    # Act
    result = await run_post(guard, text_response("sk-flagged-key"), data)
    # Assert
    assert result is None
    decision = next(d for d in data["gatelaya"]["decisions"] if d["check"] == "secret_leak")
    assert decision["action"] == "flag"
    leak = next(r for r in sink.records if r.check == "secret_leak")
    assert leak.action_taken == "flag"


@pytest.mark.skipif(not LITELLM_IMPORTABLE, reason="litellm not importable")
async def test_flag_path_attaches_meta_to_model_response() -> None:
    """Arrange: flag action + real ModelResponse. Act: hook. Assert: no raise,
    `_hidden_params['gatelaya']` attached best-effort."""
    cfg = GateLayaConfig(actions={"secret_leak": "flag"})
    agent = FakeAgent(clean_answers(secret_leak={"noul": 0.99}))
    guard = GateLayaGuardrail(config=cfg, agent=agent, audit=InMemoryAuditSink())
    response = ModelResponse(
        choices=[{"message": {"role": "assistant", "content": "sk-attach-me"}}]
    )
    # Act
    result = await run_post(guard, response)
    # Assert
    assert result is None
    hidden = getattr(response, "_hidden_params", None)
    assert isinstance(hidden, dict)
    assert "gatelaya" in hidden


async def test_mask_action_on_response_degrades_to_flag() -> None:
    """Arrange: secret_leak action='mask' (documented v1: response masks have no
    redaction patterns). Act: hook. Assert: no raise, decision recorded as flag."""
    sink = InMemoryAuditSink()
    cfg = GateLayaConfig(actions={"secret_leak": "mask"})
    agent = FakeAgent(clean_answers(secret_leak={"noul": 0.99}))
    guard = GateLayaGuardrail(config=cfg, agent=agent, audit=sink)
    data: dict = {}
    # Act
    result = await run_post(guard, text_response("sk-masked"), data)
    # Assert
    assert result is None
    leak = next(r for r in sink.records if r.check == "secret_leak")
    assert leak.action_taken == "flag"
    decision = next(d for d in data["gatelaya"]["decisions"] if d["check"] == "secret_leak")
    assert decision["action"] == "flag"


# --------------------------------------------------------- failure modes


async def test_agent_error_fail_open_returns_none_and_audits() -> None:
    """Arrange: agent raises, fail_open=True. Act: hook. Assert: None returned,
    agent_error audit record with mode post_call."""
    sink = InMemoryAuditSink()
    agent = FakeAgent(raise_error=RuntimeError("model exploded"))
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=sink)
    # Act
    result = await run_post(guard, text_response("hello"))
    # Assert
    assert result is None
    assert len(sink.records) == 1
    assert sink.records[0].check == "agent_error"
    assert sink.records[0].action_taken == "allow"
    assert sink.records[0].mode == "post_call"


async def test_agent_error_fail_closed_raises_400() -> None:
    """Arrange: agent raises, fail_open=False. Act: hook. Assert: HTTPException
    400 with fail-closed detail."""
    cfg = GateLayaConfig(fail_open=False)
    agent = FakeAgent(raise_error=RuntimeError("model exploded"))
    guard = GateLayaGuardrail(config=cfg, agent=agent, audit=InMemoryAuditSink())
    # Act
    with pytest.raises(HTTPException) as excinfo:
        await run_post(guard, text_response("hello"))
    # Assert
    assert excinfo.value.status_code == 400
    assert "fail-closed" in excinfo.value.detail


# ----------------------------------------------------- check restriction


async def test_enabled_checks_restriction_honored_on_post() -> None:
    """Arrange: enabled_checks=['toxicity'] only. Act: hook with toxic answer.
    Assert: only the toxicity question sent; secret_leak question absent."""
    cfg = GateLayaConfig(enabled_checks=["toxicity"])
    agent = FakeAgent(clean_answers(toxicity={"noul": 0.01}))
    guard = GateLayaGuardrail(config=cfg, agent=agent, audit=InMemoryAuditSink())
    # Act
    result = await run_post(guard, text_response("fine"))
    # Assert
    assert result is None
    assert set(agent.last_questions) == {"toxicity"}


async def test_disabled_post_checks_skip_agent() -> None:
    """Arrange: enabled_checks=[] (no checks at all). Act: hook. Assert: agent
    never called, None returned — post-call scans every enabled check."""
    cfg = GateLayaConfig(enabled_checks=[])
    agent = FakeAgent()
    guard = GateLayaGuardrail(config=cfg, agent=agent, audit=InMemoryAuditSink())
    # Act
    result = await run_post(guard, text_response("anything"))
    # Assert
    assert result is None
    assert agent.call_count == 0
