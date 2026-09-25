"""Tests for the streaming hook: buffered end-of-stream scan, clean passthrough."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from gatelaya.audit import InMemoryAuditSink
from gatelaya.config import GateLayaConfig
from gatelaya.guardrail import GateLayaGuardrail

from .conftest import FakeAgent, FakeStreamResponse


async def collect_chunks(guard: GateLayaGuardrail, chunks: list[Any]) -> list[Any]:
    """Drain the streaming hook and return every yielded chunk."""
    stream = FakeStreamResponse(chunks)
    collected: list[Any] = []
    async for chunk in guard.async_post_call_streaming_iterator_hook(None, stream, {}):
        collected.append(chunk)
    return collected


async def test_streaming_yields_all_chunks_unchanged(
    guardrail: GateLayaGuardrail,
) -> None:
    """Arrange: three dict chunks. Act: drain the streaming hook.
    Assert: same objects, same order."""
    chunks = [{"delta": "Hello"}, {"delta": " world"}, {"finish": "stop"}]
    # Act
    got = await collect_chunks(guardrail, chunks)
    # Assert
    assert got == chunks
    assert all(g is c for g, c in zip(got, chunks))  # identity, not copies


async def test_streaming_does_not_call_agent() -> None:
    """Arrange: fake agent that would record any call. Act: stream chunks.
    Assert: agent never invoked (v1 streaming is audit-only)."""
    sink = InMemoryAuditSink()
    agent = FakeAgent()
    guard = GateLayaGuardrail(config=GateLayaConfig(), agent=agent, audit=sink)
    # Act
    got = await collect_chunks(guard, [{"a": 1}, {"b": 2}])
    # Assert
    assert len(got) == 2
    assert agent.call_count == 0


async def test_streaming_writes_no_audit_records() -> None:
    """Arrange: in-memory sink. Act: stream chunks. Assert: sink stays empty —
    pass-through performs no decisions."""
    sink = InMemoryAuditSink()
    guard = GateLayaGuardrail(
        config=GateLayaConfig(), agent=FakeAgent(), audit=sink
    )
    # Act
    await collect_chunks(guard, [{"x": 1}])
    # Assert
    assert sink.records == []


async def test_streaming_logs_pass_through_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Arrange: logging capture at INFO. Act: two streaming sessions.
    Assert: end-of-stream buffering notice logged exactly once per guardrail."""
    guard = GateLayaGuardrail(
        config=GateLayaConfig(), agent=FakeAgent(), audit=InMemoryAuditSink()
    )
    with caplog.at_level(logging.INFO, logger="gatelaya"):
        # Act
        await collect_chunks(guard, [{"a": 1}])
        await collect_chunks(guard, [{"b": 2}])
    # Assert
    notices = [r for r in caplog.records if "buffered and scanned" in r.getMessage()]
    assert len(notices) == 1


async def test_streaming_empty_stream_yields_nothing() -> None:
    """Arrange: empty chunk list. Act: drain hook. Assert: no chunks."""
    guard = GateLayaGuardrail(
        config=GateLayaConfig(), agent=FakeAgent(), audit=InMemoryAuditSink()
    )
    # Act
    got = await collect_chunks(guard, [])
    # Assert
    assert got == []
