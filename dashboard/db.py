"""Async engine, session factory, and dashboard schema (audit + review tables)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from gatelaya.audit import DecisionRecord, decision_metadata, guardrail_decisions

from .settings import get_settings

review_metadata = MetaData()

review_labels = Table(
    "review_labels",
    review_metadata,
    Column("id", String(64), primary_key=True),
    Column("decision_id", String(64), nullable=False, unique=True, index=True),
    Column("label", String(16), nullable=False),
    Column("note", Text, nullable=True),
    Column("text", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def utc_dt(value: datetime) -> datetime:
    """Attach/convert to UTC — sqlite returns naive UTC wall times."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def row_to_record(row: Mapping[str, Any]) -> DecisionRecord:
    """Build a DecisionRecord from a guardrail_decisions row mapping."""
    data = dict(row)
    data["timestamp"] = utc_dt(data["timestamp"])
    return DecisionRecord(**data)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Cached async engine for GATELAYA_DATABASE_URL."""
    return create_async_engine(get_settings().database_url)


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Cached session factory used by the per-request dependency."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def init_db() -> None:
    """Create the audit and review tables if they do not exist."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(decision_metadata.create_all)
        await conn.run_sync(review_metadata.create_all)
