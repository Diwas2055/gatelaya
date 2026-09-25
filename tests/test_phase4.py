"""Phase 4: post-call output safety, streaming scanning, hot reload, tenant policies."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi import HTTPException
from pydantic import ValidationError

from gatelaya.audit import InMemoryAuditSink
from gatelaya.config import GateLayaConfig
from gatelaya.errors import GuardrailConfigurationError
from gatelaya.guardrail import REDACTION_TOKEN, GateLayaGuardrail
from gatelaya.hotreload import FileWatcher, build_watcher
from gatelaya.policies import PolicyMatch, PolicySet
from gatelaya.routing import GateLayaRouter, RoutingPolicy

from .conftest import FakeAgent, FakeStreamResponse, clean_answers

#: Tests that need hot reload set this to a tiny interval, then sleep past it.
FAST_POLL = "GATELAYA_RELOAD_INTERVAL"

ENV_VARS = (
    "GATELAYA_CONFIG_PATH", "GATELAYA_CALIBRATION_PATH", "GATELAYA_POLICY_PATH",
    "GATELAYA_ROUTING_PATH", "GATELAYA_HOT_RELOAD", "GATELAYA_RELOAD_INTERVAL",
    "GATELAYA_THRESHOLD_PII", "GATELAYA_THRESHOLD_INJECTION", "GATELAYA_THRESHOLD_TOXICITY",
    "GATELAYA_THRESHOLD_SECRET_LEAK", "GATELAYA_ACTION_PII", "GATELAYA_ACTION_INJECTION",
    "GATELAYA_ACTION_TOXICITY", "GATELAYA_ACTION_SECRET_LEAK", "GATELAYA_FAIL_OPEN",
    "GATELAYA_ENABLED_CHECKS", "GATELAYA_AUDIT_ENABLED", "GATELAYA_MODEL_PATH",
    "GATELAYA_ROUTE_ENABLED", "GATELAYA_ROUTE_DEFAULT", "GATELAYA_ROUTE_SENSITIVE",
    "GATELAYA_ROUTE_SENSITIVE_THRESHOLD", "GATELAYA_ROUTE_CONFIDENCE_THRESHOLD",
    "GATELAYA_ROUTE_TIERS", "GATELAYA_DATABASE_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from GATELAYA_* leakage across the suite."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------- helpers


def pre_data(text: str = "Hello, how are you?") -> dict:
    """Minimal pre-call payload with one user message."""
    return {"model": "m", "messages": [{"role": "user", "content": text}]}


def pii_text() -> str:
    """Text carrying a regex-matchable phone number (mask needs a real span)."""
    return "call me at 555-123-4567 today"


def text_response(text: str, reasoning: str | None = None) -> dict:
    """Dict-shaped chat completion with optional reasoning_content."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-p4",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def multi_response(texts: list[str]) -> dict:
    """Dict-shaped chat completion with one choice per text."""
    return {
        "id": "chatcmpl-p4",
        "object": "chat.completion",
        "model": "m",
        "choices": [
            {"index": i, "message": {"role": "assistant", "content": t}, "finish_reason": "stop"}
            for i, t in enumerate(texts)
        ],
    }


def stream_chunk(text: str, index: int = 0) -> dict:
    """One dict-shaped streaming delta chunk."""
    return {"choices": [{"index": index, "delta": {"content": text}}]}


def make_guard(
    answers: dict[str, Any] | None = None, *, cfg: GateLayaConfig | None = None, **kwargs: Any
) -> tuple[GateLayaGuardrail, FakeAgent, InMemoryAuditSink]:
    """Guardrail wired to a FakeAgent and an in-memory sink."""
    agent = FakeAgent(answers)
    sink = InMemoryAuditSink()
    guard = GateLayaGuardrail(
        config=cfg if cfg is not None else GateLayaConfig(), agent=agent, audit=sink, **kwargs
    )
    return guard, agent, sink


async def drain(
    guard: GateLayaGuardrail, chunks: list[Any], request: dict | None = None
) -> list[Any]:
    """Collect every chunk the streaming hook yields."""
    collected: list[Any] = []
    async for chunk in guard.async_post_call_streaming_iterator_hook(
        None, FakeStreamResponse(chunks), request if request is not None else {}
    ):
        collected.append(chunk)
    return collected


class KeyStub:
    """Minimal stand-in for litellm's UserAPIKeyAuth (the fields policies match on)."""

    def __init__(
        self, key_alias: str | None = None, user_id: str | None = None,
        team_id: str | None = None, api_key: str | None = None,
    ) -> None:
        self.key_alias = key_alias
        self.user_id = user_id
        self.team_id = team_id
        self.api_key = api_key


def write_policies(path: Path, policies: list[dict]) -> None:
    """Write a policies YAML file (empty list = no policies)."""
    path.write_text(yaml.safe_dump({"policies": policies}, sort_keys=False), encoding="utf-8")


def routing_answers(complexity: str = "hard") -> dict[str, Any]:
    """Laya-shaped answers for the three routing questions."""
    return {
        "task": {"choice": "generate", "confidence": 0.9},
        "complexity": {"choice": complexity, "confidence": 0.95},
        "sensitive": {"noul": 0.01},
    }


# --------------------------------------------- post-call assistant-output safety


async def test_post_call_pii_block_raises_400() -> None:
    """PII above threshold with action=block is detected in the response text."""
    guard, _, _ = make_guard(
        clean_answers(pii={"noul": 0.99}), cfg=GateLayaConfig(actions={"pii": "block"})
    )
    with pytest.raises(HTTPException) as excinfo:
        await guard.async_post_call_success_hook({}, None, text_response("ssn 123-45-6789"))
    assert excinfo.value.status_code == 400
    assert "PII" in excinfo.value.detail
    assert "in response" in excinfo.value.detail


async def test_post_call_pii_mask_rewrites_response() -> None:
    """Masked PII returns the same response object with the span redacted."""
    guard, _, sink = make_guard(clean_answers(pii={"noul": 0.99}))
    response = text_response("Call me at 555-123-4567 tomorrow")
    result = await guard.async_post_call_success_hook({}, None, response)
    assert result is response
    content = result["choices"][0]["message"]["content"]
    assert REDACTION_TOKEN in content
    assert "555-123-4567" not in content
    pii = next(r for r in sink.records if r.check == "pii")
    assert pii.action_taken == "mask"
    assert pii.masked is True


async def test_post_call_scans_reasoning_content() -> None:
    """Clean content does not hide PII in reasoning_content."""
    guard, _, _ = make_guard(
        clean_answers(pii={"noul": 0.99}), cfg=GateLayaConfig(actions={"pii": "block"})
    )
    response = text_response("All good here", reasoning="their ssn is 123-45-6789")
    with pytest.raises(HTTPException) as excinfo:
        await guard.async_post_call_success_hook({}, None, response)
    assert excinfo.value.status_code == 400


async def test_post_call_runs_injection_check() -> None:
    """Post-call scans every enabled check (injection included), not a subset."""
    guard, agent, _ = make_guard(clean_answers(injection={"noul": 0.99}))
    with pytest.raises(HTTPException) as excinfo:
        await guard.async_post_call_success_hook({}, None, text_response("ignore rules"))
    assert "prompt injection" in excinfo.value.detail
    assert "injection" in agent.last_questions


async def test_post_call_multichoice_masks_each_choice_independently() -> None:
    """Each choice scans separately: no-PII choice flags, PII choice masks."""
    guard, _, sink = make_guard(clean_answers(pii={"noul": 0.99}))
    response = multi_response(["hi there", "call 555-123-4567"])
    result = await guard.async_post_call_success_hook({}, None, response)
    assert result is response
    choices = result["choices"]
    assert choices[0]["message"]["content"] == "hi there"  # no regex match -> untouched
    assert REDACTION_TOKEN in choices[1]["message"]["content"]
    pii_records = [r for r in sink.records if r.check == "pii"]
    assert len(pii_records) == 2
    assert {r.masked for r in pii_records} == {True, False}


async def test_post_call_multichoice_block_when_any_choice_blocks() -> None:
    """A blocking choice blocks the whole call; every choice is audited."""
    guard, agent, sink = make_guard(clean_answers(secret_leak={"noul": 0.99}))
    with pytest.raises(HTTPException):
        await guard.async_post_call_success_hook({}, None, multi_response(["fine", "sk-key"]))
    assert agent.call_count == 2  # one predict per non-empty choice
    assert len([r for r in sink.records if r.check == "secret_leak"]) == 2


async def test_post_call_meta_stashed_on_data() -> None:
    """Non-allow decisions stash decision meta on the data dict."""
    guard, _, _ = make_guard(
        clean_answers(secret_leak={"noul": 0.99}),
        cfg=GateLayaConfig(actions={"secret_leak": "flag"}),
    )
    data: dict = {}
    await guard.async_post_call_success_hook(data, None, text_response("sk-flag"))
    assert "gatelaya" in data
    assert any(d["check"] == "secret_leak" for d in data["gatelaya"]["decisions"])


# ------------------------------------------------------------------- streaming


async def test_streaming_clean_scan_yields_identity_chunks() -> None:
    """Clean buffered text: agent scanned once, chunks yielded as same objects."""
    guard, agent, sink = make_guard()
    chunks = [stream_chunk("Hello"), stream_chunk(" world"), {"choices": []}]
    got = await drain(guard, chunks)
    assert all(g is c for g, c in zip(got, chunks))
    assert agent.call_count == 1
    assert len(sink.records) == 4  # every check decided (all allow)
    assert all(r.action_taken == "allow" for r in sink.records)


async def test_streaming_block_raises_before_any_chunk_is_yielded() -> None:
    """Blocking assembled text raises 400; the client never receives a chunk."""
    guard, _, sink = make_guard(
        clean_answers(pii={"noul": 0.99}), cfg=GateLayaConfig(actions={"pii": "block"})
    )
    chunks = [stream_chunk("ssn 123-45-6789")]
    collected: list[Any] = []
    with pytest.raises(HTTPException) as excinfo:
        async for chunk in guard.async_post_call_streaming_iterator_hook(
            None, FakeStreamResponse(chunks), {}
        ):
            collected.append(chunk)
    assert excinfo.value.status_code == 400
    assert collected == []
    assert any(r.blocked for r in sink.records)


async def test_streaming_mask_rewrites_first_delta_and_clears_rest() -> None:
    """Masked stream: first content delta carries the redacted text, rest cleared."""
    guard, _, sink = make_guard(clean_answers(pii={"noul": 0.99}))
    chunks = [stream_chunk("Call 555-123"), stream_chunk("-4567 now")]
    request: dict = {}
    got = await drain(guard, chunks, request)
    assert len(got) == 2
    first = got[0]["choices"][0]["delta"]["content"]
    second = got[1]["choices"][0]["delta"]["content"]
    assert REDACTION_TOKEN in first
    assert "555-123-4567" not in first + second
    assert second == ""
    pii = next(r for r in sink.records if r.check == "pii")
    assert pii.masked is True
    assert "gatelaya" in request  # decision meta stashed on request_data


async def test_streaming_fail_closed_raises_without_yielding() -> None:
    """Agent failure with fail_open=False blocks the stream before delivery."""
    guard = GateLayaGuardrail(
        config=GateLayaConfig(fail_open=False),
        agent=FakeAgent(raise_error=RuntimeError("boom")),
        audit=InMemoryAuditSink(),
    )
    collected: list[Any] = []
    with pytest.raises(HTTPException) as excinfo:
        async for chunk in guard.async_post_call_streaming_iterator_hook(
            None, FakeStreamResponse([stream_chunk("hello there")]), {}
        ):
            collected.append(chunk)
    assert "fail-closed" in excinfo.value.detail
    assert collected == []
    assert guard.audit.records[0].check == "agent_error"  # type: ignore[attr-defined]


async def test_streaming_fail_open_passes_chunks_and_audits() -> None:
    """Agent failure with fail_open=True: chunks pass, agent_error recorded."""
    guard = GateLayaGuardrail(
        config=GateLayaConfig(),
        agent=FakeAgent(raise_error=RuntimeError("boom")),
        audit=InMemoryAuditSink(),
    )
    got = await drain(guard, [stream_chunk("hello")])
    assert len(got) == 1
    record = guard.audit.records[0]  # type: ignore[attr-defined]
    assert record.check == "agent_error"
    assert record.action_taken == "allow"


async def test_streaming_no_enabled_checks_passthrough() -> None:
    """enabled_checks=[]: stream passes through untouched without scanning."""
    guard, agent, sink = make_guard(cfg=GateLayaConfig(enabled_checks=[]))
    chunks = [stream_chunk("anything")]
    got = await drain(guard, chunks)
    assert got[0] is chunks[0]
    assert agent.call_count == 0
    assert sink.records == []


# ------------------------------------------------------------------ hot reload


async def test_config_file_change_reloads_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing the config YAML swaps thresholds in without a restart."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    path = tmp_path / "gatelaya.yaml"
    GateLayaConfig(thresholds={"pii": 0.95}, actions={"pii": "block"}).to_yaml(path)
    guard = GateLayaGuardrail(
        config_path=str(path), agent=FakeAgent(clean_answers(pii={"noul": 0.9})),
        audit=InMemoryAuditSink(),
    )

    assert await guard.async_pre_call_hook(None, None, pre_data("555-123-4567"), "success") is None

    GateLayaConfig(thresholds={"pii": 0.5}, actions={"pii": "block"}).to_yaml(path)
    time.sleep(0.03)
    result = await guard.async_pre_call_hook(None, None, pre_data("555-123-4567"), "success")
    assert isinstance(result, str) and "PII" in result
    assert guard.config.threshold("pii") == 0.5


async def test_hot_reload_disabled_keeps_old_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hot_reload=False: file changes are ignored until restart."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    path = tmp_path / "gatelaya.yaml"
    GateLayaConfig(thresholds={"pii": 0.95}, actions={"pii": "block"}).to_yaml(path)
    guard = GateLayaGuardrail(
        config_path=str(path), agent=FakeAgent(clean_answers(pii={"noul": 0.9})),
        audit=InMemoryAuditSink(), hot_reload=False,
    )
    assert guard._watcher is None

    GateLayaConfig(thresholds={"pii": 0.5}, actions={"pii": "block"}).to_yaml(path)
    time.sleep(0.03)
    result = await guard.async_pre_call_hook(None, None, pre_data("555-123-4567"), "success")
    assert result is None  # old threshold 0.95 still in effect
    assert guard.config.threshold("pii") == 0.95


async def test_invalid_config_file_keeps_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken mid-edit file never replaces the running config."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    path = tmp_path / "gatelaya.yaml"
    GateLayaConfig(thresholds={"pii": 0.95}, actions={"pii": "block"}).to_yaml(path)
    guard = GateLayaGuardrail(
        config_path=str(path), agent=FakeAgent(clean_answers(pii={"noul": 0.9})),
        audit=InMemoryAuditSink(),
    )

    path.write_text("thresholds: [broken", encoding="utf-8")
    time.sleep(0.03)
    result = await guard.async_pre_call_hook(None, None, pre_data("555-123-4567"), "success")
    assert result is None  # previous config (0.95) survived the bad edit
    assert guard.config.threshold("pii") == 0.95


async def test_config_mode_change_updates_event_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """event_hook follows the reloaded config mode unless explicitly pinned."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    path = tmp_path / "gatelaya.yaml"
    GateLayaConfig(mode=["pre_call"]).to_yaml(path)
    guard = GateLayaGuardrail(config_path=str(path), agent=FakeAgent(), audit=InMemoryAuditSink())
    assert guard.event_hook == ["pre_call"]

    GateLayaConfig(mode=["pre_call", "post_call"]).to_yaml(path)
    time.sleep(0.03)
    await guard.async_pre_call_hook(None, None, pre_data(), "success")
    assert guard.event_hook == ["pre_call", "post_call"]


async def test_calibration_file_change_reloads_temperatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Swapping calibration.json swaps the temperature map in place."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    cal = tmp_path / "calibration.json"
    cal.write_text('{"choice": {}, "noul": {"default": 1.0}}', encoding="utf-8")
    path = tmp_path / "gatelaya.yaml"
    GateLayaConfig(calibration_path=str(cal)).to_yaml(path)
    guard = GateLayaGuardrail(config_path=str(path), agent=FakeAgent(), audit=InMemoryAuditSink())
    assert guard.temperatures is not None
    assert guard.temperatures.noul_temperature("pii") == 1.0

    cal.write_text('{"choice": {}, "noul": {"default": 2.5}}', encoding="utf-8")
    time.sleep(0.03)
    await guard.async_pre_call_hook(None, None, pre_data(), "success")
    assert guard.temperatures.noul_temperature("pii") == 2.5


async def test_policy_file_change_hot_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a policy to the watched file applies it on the next hook call."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    ppath = tmp_path / "policies.yaml"
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(ppath))
    write_policies(ppath, [])
    guard, _, _ = make_guard(clean_answers(pii={"noul": 0.5}))

    # no policies yet: global threshold 0.85 -> allow
    result = await guard.async_pre_call_hook(
        KeyStub(key_alias="team-a"), None, pre_data(pii_text()), "success"
    )
    assert result is None

    write_policies(
        ppath,
        [{
            "match": {"key_alias": "team-a"},
            "overrides": {"thresholds": {"pii": 0.3}},
        }],
    )
    time.sleep(0.03)
    result2 = await guard.async_pre_call_hook(
        KeyStub(key_alias="team-a"), None, pre_data(pii_text()), "success"
    )
    assert isinstance(result2, dict)  # tenant threshold 0.3 <= 0.5 -> mask -> redacted data
    assert result2["gatelaya"]["policy"] == "key_alias=team-a"
    # a key outside the policy still runs globally
    assert await guard.async_pre_call_hook(
        KeyStub(key_alias="team-z"), None, pre_data(pii_text()), "success"
    ) is None


async def test_file_watcher_rate_limits_polls(tmp_path: Path) -> None:
    """poll() reports a change once; a second poll inside the interval is blank."""
    path = tmp_path / "a.yaml"
    path.write_text("x: 1", encoding="utf-8")
    watcher = FileWatcher([path], interval=10.0)
    path.write_text("x: 2", encoding="utf-8")
    assert watcher.poll([path]) is True
    path.write_text("x: 3", encoding="utf-8")
    assert watcher.poll([path]) is False  # rate-limited


async def test_file_watcher_commit_suppresses_unchanged_reload(tmp_path: Path) -> None:
    """After commit(), an unchanged file is no longer reported as changed."""
    path = tmp_path / "a.yaml"
    path.write_text("x: 1", encoding="utf-8")
    watcher = FileWatcher([path], interval=0.01)
    path.write_text("x: 2", encoding="utf-8")
    assert watcher.poll([path]) is True
    watcher.commit()
    time.sleep(0.02)
    assert watcher.poll([path]) is False


async def test_file_watcher_uncommitted_change_is_reported_again(tmp_path: Path) -> None:
    """A detected change without commit() keeps polling True (failed reload retries)."""
    path = tmp_path / "a.yaml"
    path.write_text("x: 1", encoding="utf-8")
    watcher = FileWatcher([path], interval=0.01)
    path.write_text("x: 2", encoding="utf-8")
    assert watcher.poll([path]) is True
    time.sleep(0.02)
    assert watcher.poll([path]) is True  # never committed -> still differs
    watcher.commit()
    time.sleep(0.02)
    assert watcher.poll([path]) is False


async def test_file_watcher_seeds_file_created_later(tmp_path: Path) -> None:
    """A path tracked as missing becomes a change when it appears."""
    path = tmp_path / "later.yaml"
    watcher = FileWatcher([path], interval=0.01)
    time.sleep(0.02)
    path.write_text("policies: []", encoding="utf-8")
    assert watcher.poll([path]) is True


def test_build_watcher_disabled_cases(tmp_path: Path) -> None:
    """interval<=0, enabled=False, or no paths all yield no watcher."""
    path = tmp_path / "a.yaml"
    path.write_text("x: 1", encoding="utf-8")
    assert build_watcher([path], enabled=True, interval=0.0) is None
    assert build_watcher([path], enabled=False, interval=1.0) is None
    assert build_watcher([], enabled=True, interval=1.0) is None
    assert build_watcher([path], enabled=True, interval=1.0) is not None


# ------------------------------------------------------------- tenant policies


async def test_policy_overrides_thresholds_and_audits_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matched key_alias gets the tenant threshold/action; audit names the policy."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    write_policies(
        tmp_path / "policies.yaml",
        [{
            "match": {"key_alias": "team-a"},
            "overrides": {"thresholds": {"pii": 0.3}, "actions": {"pii": "block"}},
        }],
    )
    guard, _, sink = make_guard(clean_answers(pii={"noul": 0.5}))

    data = pre_data(pii_text())
    result = await guard.async_pre_call_hook(KeyStub(key_alias="team-a"), None, data, "success")
    assert isinstance(result, str) and "PII" in result  # tenant threshold 0.3 <= 0.5 blocks
    assert data["gatelaya"]["policy"] == "key_alias=team-a"
    assert sink.records[0].detail["policy"] == "key_alias=team-a"

    # same shape, unmatched key: global threshold 0.85 -> allow
    data2: dict = {}
    result2 = await guard.async_pre_call_hook(KeyStub(key_alias="team-z"), None, data2, "success")
    assert result2 is None
    assert "policy" not in data2.get("gatelaya", {})


async def test_policy_enabled_checks_replaces_and_fail_open_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant enabled_checks replaces the list; fail_open=false blocks on errors."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    write_policies(
        tmp_path / "policies.yaml",
        [{
            "match": {"key_alias": "team-b"},
            "overrides": {"enabled_checks": ["pii"], "fail_open": False},
        }],
    )
    agent = FakeAgent(clean_answers(pii={"noul": 0.99}))
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=InMemoryAuditSink())

    result = await guard.async_pre_call_hook(
        KeyStub(key_alias="team-b"), None, pre_data(pii_text()), "success"
    )
    assert set(agent.last_questions) == {"pii", "pii_type"}  # replacement, not merge
    assert isinstance(result, dict)  # pii 0.99 >= 0.85 -> mask (phone span redacted)

    # agent error with the tenant's fail_open=false -> fail-closed block string
    guard2 = GateLayaGuardrail(
        config=GateLayaConfig(),
        agent=FakeAgent(raise_error=RuntimeError("boom")),
        audit=InMemoryAuditSink(),
    )
    closed = await guard2.async_pre_call_hook(
        KeyStub(key_alias="team-b"), None, pre_data(pii_text()), "success"
    )
    assert closed == "GateLaya: internal guardrail error (fail-closed)"


async def test_policy_api_key_hash_matches_plaintext_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sha256 prefix in the policy matches a plaintext key (never stored)."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    plaintext = "sk-test-1234567890"
    digest = hashlib.sha256(plaintext.encode()).hexdigest()
    write_policies(
        tmp_path / "policies.yaml",
        [{
            "match": {"api_key_hash": digest[:16]},
            "overrides": {"thresholds": {"pii": 0.1}},
        }],
    )
    guard, _, _ = make_guard(clean_answers(pii={"noul": 0.9}))
    data: dict = pre_data(pii_text())
    # threshold 0.1 <= 0.9 with default action mask -> redacted data returned
    result = await guard.async_pre_call_hook(KeyStub(api_key=plaintext), None, data, "success")
    assert isinstance(result, dict)
    assert data["gatelaya"]["policy"] == f"api_key_hash={digest[:16]}"


async def test_policy_first_match_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Policies resolve in file order: the first full match supplies the overrides."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    write_policies(
        tmp_path / "policies.yaml",
        [
            {"match": {"key_alias": "dup"}, "overrides": {"thresholds": {"pii": 0.2}}},
            {"match": {"key_alias": "dup"}, "overrides": {"thresholds": {"pii": 0.9}}},
        ],
    )
    policies = PolicySet.from_yaml(tmp_path / "policies.yaml")
    tenant, label = policies.resolve(KeyStub(key_alias="dup"))
    assert tenant is not None
    assert tenant.overrides.thresholds == {"pii": 0.2}
    assert label == "key_alias=dup"


async def test_missing_policy_file_is_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-missing policy file leaves the guardrail running globally."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "absent.yaml"))
    guard, _, _ = make_guard()
    assert guard.policies is None
    result = await guard.async_pre_call_hook(KeyStub(key_alias="x"), None, pre_data(), "success")
    assert result is None


def test_policy_match_requires_a_field() -> None:
    """An empty match block is rejected at model validation."""
    with pytest.raises(ValidationError, match="at least one identity field"):
        PolicyMatch()


def test_policy_rejects_plaintext_api_key_hash(tmp_path: Path) -> None:
    """api_key_hash must be hex (sha256), never a raw key."""
    path = tmp_path / "bad.yaml"
    write_policies(path, [{"match": {"api_key_hash": "not-a-hash-at-all"}, "overrides": {}}])
    with pytest.raises(GuardrailConfigurationError, match="invalid policy file"):
        PolicySet.from_yaml(path)


async def test_policy_post_call_uses_tenant_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-call scans also honor the matched tenant's thresholds/actions."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    write_policies(
        tmp_path / "policies.yaml",
        [{
            "match": {"user_id": "u-1"},
            "overrides": {"thresholds": {"secret_leak": 0.2}, "actions": {"secret_leak": "block"}},
        }],
    )
    guard, _, _ = make_guard(clean_answers(secret_leak={"noul": 0.5}))
    with pytest.raises(HTTPException) as excinfo:
        await guard.async_post_call_success_hook({}, KeyStub(user_id="u-1"), text_response("sk-1"))
    assert "secret leak" in excinfo.value.detail
    # global key (0.5 < 0.85) passes
    assert await guard.async_post_call_success_hook({}, None, text_response("sk-1")) is None


# ----------------------------------------------------- router: reload + policies


async def test_router_routing_file_change_hot_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing routing.yaml swaps tier targets without a restart."""
    monkeypatch.setenv(FAST_POLL, "0.01")
    path = tmp_path / "routing.yaml"
    RoutingPolicy(tiers={"hard": "frontier"}).to_yaml(path)
    router = GateLayaRouter(
        agent=FakeAgent(routing_answers()), audit=InMemoryAuditSink(), routing_path=str(path)
    )
    data = pre_data()
    await router.async_pre_call_hook(None, None, data, "success")
    assert data["model"] == "frontier"

    RoutingPolicy(tiers={"hard": "new-frontier"}).to_yaml(path)
    time.sleep(0.03)
    data2 = pre_data()
    await router.async_pre_call_hook(None, None, data2, "success")
    assert data2["model"] == "new-frontier"


async def test_router_policy_disables_routing_for_tenant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tenant routing override (enabled=false) pins the original model."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    write_policies(
        tmp_path / "policies.yaml",
        [{
            "match": {"team_id": "team-a"},
            "overrides": {"routing": {"enabled": False}},
        }],
    )
    router = GateLayaRouter(
        policy=RoutingPolicy(tiers={"hard": "frontier"}),
        agent=FakeAgent(routing_answers()), audit=InMemoryAuditSink(),
    )
    data = pre_data()
    result = await router.async_pre_call_hook(KeyStub(team_id="team-a"), None, data, "success")
    assert result is None
    assert "gatelaya_routing" not in data
    assert data["model"] == "m"  # untouched

    # unmatched key still routes
    data2 = pre_data()
    await router.async_pre_call_hook(KeyStub(team_id="team-z"), None, data2, "success")
    assert data2["model"] == "frontier"


async def test_router_policy_tenant_tiers_and_audit_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant routing overrides replace tier targets; the route record names the policy."""
    monkeypatch.setenv("GATELAYA_POLICY_PATH", str(tmp_path / "policies.yaml"))
    write_policies(
        tmp_path / "policies.yaml",
        [{
            "match": {"key_alias": "team-a"},
            "overrides": {"routing": {"tiers": {"hard": "tenant-model"}}},
        }],
    )
    sink = InMemoryAuditSink()
    router = GateLayaRouter(
        policy=RoutingPolicy(tiers={"hard": "frontier"}),
        agent=FakeAgent(routing_answers()), audit=sink,
    )
    data = pre_data()
    await router.async_pre_call_hook(KeyStub(key_alias="team-a"), None, data, "success")
    assert data["model"] == "tenant-model"
    route = next(r for r in sink.records if r.check == "routing")
    assert route.detail["policy"] == "key_alias=team-a"
