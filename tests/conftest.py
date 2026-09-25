"""Shared fixtures and fakes for the GateLaya test suite.

No network, no model downloads: every test injects a ``FakeAgent`` that
implements the ``LayaAgent`` protocol from ``gatelaya/agent.py``.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from gatelaya.audit import InMemoryAuditSink
from gatelaya.config import GateLayaConfig
from gatelaya.guardrail import GateLayaGuardrail

# --------------------------------------------------------------------- fakes


class FakeAgent:
    """Configurable ``LayaAgent`` stand-in.

    Records every ``predict`` call, and supports failure modes:
    ``raise_error`` (raise instead of answering) and ``delay`` (sleep before
    answering — keep < 0.1s per project rules).
    """

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        *,
        delay: float = 0.0,
        raise_error: Exception | None = None,
    ) -> None:
        self.answers: dict[str, Any] = dict(answers) if answers is not None else clean_answers()
        self.delay = delay
        self.raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """Number of predict invocations so far."""
        return len(self.calls)

    @property
    def last_questions(self) -> dict[str, Any]:
        """Questions dict from the most recent predict call."""
        assert self.calls, "FakeAgent.predict was never called"
        return self.calls[-1]["questions"]

    @property
    def last_state(self) -> dict[str, Any]:
        """State dict from the most recent predict call."""
        assert self.calls, "FakeAgent.predict was never called"
        return self.calls[-1]["state"]

    def predict(self, state: dict, questions: dict) -> dict:
        """Implements the ``LayaAgent`` protocol."""
        self.calls.append(
            {
                "state": dict(state),
                "questions": {name: dict(q) for name, q in questions.items()},
            }
        )
        if self.delay:
            time.sleep(self.delay)
        if self.raise_error is not None:
            raise self.raise_error
        return {"answers": dict(self.answers)}


def clean_answers(**overrides: Any) -> dict[str, Any]:
    """All four checks confidently negative, plus a neutral pii_type answer."""
    answers: dict[str, Any] = {
        "pii": {"noul": 0.01},
        "pii_type": {"choice": "none", "confidence": 0.95},
        "injection": {"noul": 0.01},
        "toxicity": {"noul": 0.01},
        "secret_leak": {"noul": 0.01},
    }
    answers.update(overrides)
    return answers


class FakeStreamResponse:
    """Minimal async-iterator matching what the streaming hook consumes."""

    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = list(chunks)

    def __aiter__(self) -> "FakeStreamResponse":
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self._index]
        self._index += 1
        return chunk


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def config() -> GateLayaConfig:
    """Default GateLayaConfig (thresholds/actions straight from source defaults)."""
    return GateLayaConfig()


@pytest.fixture
def fake_agent() -> FakeAgent:
    """FakeAgent answering every check with a clean (0.01) probability."""
    return FakeAgent(clean_answers())


@pytest.fixture
def audit_sink() -> InMemoryAuditSink:
    """In-memory audit sink for asserting recorded decisions."""
    return InMemoryAuditSink()


@pytest.fixture
def guardrail(
    config: GateLayaConfig, fake_agent: FakeAgent, audit_sink: InMemoryAuditSink
) -> GateLayaGuardrail:
    """Guardrail wired to the default fake agent and in-memory audit sink."""
    return GateLayaGuardrail(config=config, agent=fake_agent, audit=audit_sink)


@pytest.fixture
def make_guardrail(
    config: GateLayaConfig, audit_sink: InMemoryAuditSink
) -> Any:
    """Factory: ``g, agent = make_guardrail(answers=..., cfg=..., audit=...)``."""

    def _make(
        answers: dict[str, Any] | None = None,
        *,
        cfg: GateLayaConfig | None = None,
        agent: FakeAgent | None = None,
        audit: InMemoryAuditSink | None = None,
        **kwargs: Any,
    ) -> tuple[GateLayaGuardrail, FakeAgent]:
        real_agent = agent if agent is not None else FakeAgent(answers)
        guard = GateLayaGuardrail(
            config=cfg if cfg is not None else config,
            agent=real_agent,
            audit=audit if audit is not None else audit_sink,
            **kwargs,
        )
        return guard, real_agent

    return _make


@pytest.fixture
def pre_call_data() -> dict:
    """Sample LiteLLM pre-call payload with a system + user message."""
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    }


@pytest.fixture
def sample_response() -> dict:
    """Sample dict-shaped chat completion response for post-call hooks."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi! How can I help?"},
                "finish_reason": "stop",
            }
        ],
    }
