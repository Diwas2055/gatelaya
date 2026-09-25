"""Tests for DecisionRecord and audit sinks (in-memory + SQLAlchemy)."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from gatelaya.audit import (
    DecisionRecord,
    InMemoryAuditSink,
    SqlAlchemyAuditSink,
    guardrail_decisions,
)
from gatelaya.errors import GuardrailConfigurationError

AIOSQLITE_AVAILABLE = importlib.util.find_spec("aiosqlite") is not None


def make_record(**overrides: Any) -> DecisionRecord:
    """Build a DecisionRecord with sensible defaults (per-field overrides allowed)."""
    fields: dict[str, Any] = {
        "check": "injection",
        "language": "english",
        "probs": {"raw": 0.95, "calibrated": 0.93},
        "action_taken": "block",
        "blocked": True,
        "masked": False,
        "input_sha256": "a" * 64,
        "latency_ms": 12.5,
        "mode": "pre_call",
    }
    fields.update(overrides)
    return DecisionRecord(**fields)


# ---------------------------------------------------------- DecisionRecord


def test_decision_record_defaults() -> None:
    """Arrange/act: minimal record. Assert: id/timestamp auto-populated."""
    rec = make_record()
    assert rec.id
    assert len(rec.id) == 32  # uuid4 hex
    assert rec.timestamp.tzinfo is not None
    assert rec.timestamp <= datetime.now(timezone.utc)


def test_decision_record_model_dump_is_serializable() -> None:
    """Arrange: record. Act: model_dump / model_dump_json.
    Assert: JSON round-trips with all columns present."""
    rec = make_record()
    # Act
    dumped = rec.model_dump()
    as_json = json.loads(rec.model_dump_json())
    # Assert
    expected = {
        "id",
        "timestamp",
        "check",
        "language",
        "probs",
        "action_taken",
        "blocked",
        "masked",
        "input_sha256",
        "latency_ms",
        "mode",
    }
    assert expected <= set(dumped)
    assert as_json["check"] == "injection"
    assert as_json["probs"] == {"raw": 0.95, "calibrated": 0.93}
    assert as_json["blocked"] is True


def test_decision_record_rejects_bad_assignment() -> None:
    """Arrange: record. Act: assign wrong-typed field.
    Assert (CORRECT behavior): pydantic rejects the mutation so records stay
    consistent. Currently assignment silently succeeds."""
    from pydantic import ValidationError

    rec = make_record()
    with pytest.raises(ValidationError):
        rec.latency_ms = "fast"  # type: ignore[assignment]
    assert rec.latency_ms == 12.5


# -------------------------------------------------------- InMemoryAuditSink


async def test_in_memory_sink_appends_in_order() -> None:
    """Arrange: sink. Act: three records. Assert: same order, all kept."""
    sink = InMemoryAuditSink()
    first, second, third = make_record(), make_record(check="pii"), make_record(check="toxicity")
    # Act
    await sink.record(first)
    await sink.record(second)
    await sink.record(third)
    # Assert
    assert sink.records == [first, second, third]


async def test_in_memory_sink_caps_at_max_records_dropping_oldest() -> None:
    """Arrange: sink capped at 3. Act: record 5.
    Assert: only the newest 3 survive (oldest two dropped)."""
    sink = InMemoryAuditSink(max_records=3)
    records = [make_record(check=f"check_{i}") for i in range(5)]
    # Act
    for rec in records:
        await sink.record(rec)
    # Assert
    assert len(sink.records) == 3
    assert [r.check for r in sink.records] == ["check_2", "check_3", "check_4"]


async def test_in_memory_sink_rejects_nonpositive_cap() -> None:
    """Arrange/act: max_records < 1. Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="max_records"):
        InMemoryAuditSink(max_records=0)


async def test_in_memory_sink_protocol_conformance() -> None:
    """Arrange: sink. Act: isinstance check against runtime-checkable protocol.
    Assert: satisfies AuditSink."""
    from gatelaya.audit import AuditSink

    assert isinstance(InMemoryAuditSink(), AuditSink)


# ------------------------------------------------------ SqlAlchemyAuditSink


@pytest.mark.skipif(not AIOSQLITE_AVAILABLE, reason="aiosqlite driver not installed")
async def test_sqlalchemy_sink_roundtrip_sqlite_memory() -> None:
    """Arrange: async sqlite in-memory sink. Act: record one decision, read rows.
    Assert: row comes back with every field intact."""
    from sqlalchemy import select

    sink = SqlAlchemyAuditSink("sqlite+aiosqlite:///:memory:")
    rec = make_record()
    try:
        # Act
        await sink.record(rec)
        async with sink._engine.connect() as conn:
            rows = (await conn.execute(select(guardrail_decisions))).fetchall()
        # Assert
        assert len(rows) == 1
        row = rows[0]
        assert row.check == "injection"
        assert row.language == "english"
        assert row.action_taken == "block"
        assert row.blocked is True
        assert row.masked is False
        assert row.probs == {"raw": 0.95, "calibrated": 0.93}
        assert row.input_sha256 == "a" * 64
        assert row.latency_ms == pytest.approx(12.5)
        assert row.mode == "pre_call"
    finally:
        await sink.close()


@pytest.mark.skipif(not AIOSQLITE_AVAILABLE, reason="aiosqlite driver not installed")
async def test_sqlalchemy_sink_multiple_records_and_lazy_table_creation() -> None:
    """Arrange: fresh sink. Act: two records without explicit start().
    Assert: table auto-created, both rows persisted."""
    from sqlalchemy import func, select

    sink = SqlAlchemyAuditSink("sqlite+aiosqlite:///:memory:")
    try:
        # Act
        await sink.record(make_record(check="pii"))
        await sink.record(make_record(check="toxicity", blocked=False, action_taken="flag"))
        async with sink._engine.connect() as conn:
            count = (await conn.execute(select(func.count(guardrail_decisions.c.id)))).scalar()
            checks = (await conn.execute(select(guardrail_decisions.c.check))).scalars().all()
        # Assert
        assert count == 2
        assert list(checks) == ["pii", "toxicity"]
    finally:
        await sink.close()


async def test_sqlalchemy_sink_missing_driver_message() -> None:
    """Arrange/act: URL whose async driver is not installed (aiomysql absent).
    Assert: GuardrailConfigurationError naming the driver."""
    with pytest.raises(GuardrailConfigurationError, match="missing async database driver"):
        SqlAlchemyAuditSink("mysql+aiomysql://user:pass@localhost/db")
