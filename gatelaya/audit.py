"""Decision audit records and pluggable sinks (in-memory / SQLAlchemy)."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .errors import GuardrailConfigurationError

decision_metadata = MetaData()

guardrail_decisions = Table(
    "guardrail_decisions",
    decision_metadata,
    Column("id", String(64), primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("check", String(32), nullable=False),
    Column("language", String(32), nullable=False),
    Column("probs", JSON, nullable=False),
    Column("action_taken", String(16), nullable=False),
    Column("blocked", Boolean, nullable=False),
    Column("masked", Boolean, nullable=False),
    Column("input_sha256", String(64), nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("mode", String(32), nullable=False),
    Column("detail", JSON, nullable=False),
)


class DecisionRecord(BaseModel):
    """One guardrail decision: what ran, what happened, how long it took."""

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    check: str
    language: str
    probs: dict[str, float]
    action_taken: str
    blocked: bool
    masked: bool
    input_sha256: str
    latency_ms: float
    mode: str
    detail: dict[str, str] = {}


@runtime_checkable
class AuditSink(Protocol):
    """Async sink for decision records."""

    async def record(self, rec: DecisionRecord) -> None:
        """Persist one decision record."""
        ...


class InMemoryAuditSink:
    """Capped in-memory sink for tests and audit-disabled runs."""

    def __init__(self, max_records: int = 1000) -> None:
        if max_records < 1:
            raise GuardrailConfigurationError("max_records must be >= 1")
        self.max_records = max_records
        self.records: list[DecisionRecord] = []

    async def record(self, rec: DecisionRecord) -> None:
        """Append a record, evicting oldest entries past the cap."""
        self.records.append(rec)
        overflow = len(self.records) - self.max_records
        if overflow > 0:
            del self.records[:overflow]


class SqlAlchemyAuditSink:
    """Async SQLAlchemy sink writing to the `guardrail_decisions` table."""

    def __init__(self, url: str) -> None:
        try:
            self._engine: AsyncEngine = create_async_engine(url)
        except ModuleNotFoundError as exc:
            raise GuardrailConfigurationError(
                f"missing async database driver for {url!r}: {exc}. "
                "For Postgres install with: pip install asyncpg"
            ) from exc
        self._ready = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Create the decisions table if it does not exist."""
        async with self._lock:
            if self._ready:
                return
            try:
                async with self._engine.begin() as conn:
                    await conn.run_sync(decision_metadata.create_all)
            except ModuleNotFoundError as exc:
                raise GuardrailConfigurationError(
                    f"missing async database driver: {exc}. "
                    "For Postgres install with: pip install asyncpg; "
                    "for SQLite: pip install aiosqlite"
                ) from exc
            self._ready = True

    async def record(self, rec: DecisionRecord) -> None:
        """Insert one decision row (lazily creating the table on first write)."""
        await self.start()
        values = rec.model_dump()
        values["timestamp"] = rec.timestamp
        async with self._engine.begin() as conn:
            await conn.execute(guardrail_decisions.insert().values(**values))

    async def close(self) -> None:
        """Dispose the underlying engine pool."""
        await self._engine.dispose()
