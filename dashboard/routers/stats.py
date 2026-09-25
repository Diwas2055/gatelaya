"""Aggregate decision statistics for a rolling time window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from gatelaya.audit import guardrail_decisions

from ..db import utc_dt
from ..dependencies import SessionDep

router = APIRouter(prefix="/api", tags=["stats"])


class StatsOut(BaseModel):
    """Counts, latency average, and group-by breakdowns for a window."""

    total: int
    blocked: int
    flagged: int
    masked: int
    allowed: int
    avg_latency_ms: float
    by_check: dict[str, int]
    by_action: dict[str, int]
    by_mode: dict[str, int]
    window_hours: int


def _count_when(condition: ColumnElement[bool]) -> ColumnElement[int]:
    """Count rows where `condition` holds."""
    return func.coalesce(func.sum(case((condition, 1), else_=0)), 0)


async def _group_counts(
    session: AsyncSession, column: ColumnElement[str], *clauses: ColumnElement[bool]
) -> dict[str, int]:
    """Count rows grouped by one column, restricted to the window clauses."""
    rows = (
        await session.execute(select(column, func.count()).where(*clauses).group_by(column))
    ).all()
    return {str(key): int(count) for key, count in rows}


@router.get("/stats", response_model=StatsOut)
async def stats(session: SessionDep, hours: int = Query(24, ge=1, le=8760)) -> StatsOut:
    """Summarize decisions recorded in the last `hours` hours."""
    cutoff = utc_dt(datetime.now(timezone.utc) - timedelta(hours=hours))
    window = guardrail_decisions.c.timestamp >= cutoff
    row = (
        await session.execute(
            select(
                func.count().label("total"),
                _count_when(guardrail_decisions.c.blocked.is_(True)).label("blocked"),
                _count_when(guardrail_decisions.c.action_taken == "flag").label("flagged"),
                _count_when(guardrail_decisions.c.action_taken == "mask").label("masked"),
                _count_when(guardrail_decisions.c.action_taken == "allow").label("allowed"),
                func.avg(guardrail_decisions.c.latency_ms).label("avg_latency_ms"),
            ).where(window)
        )
    ).one()
    return StatsOut(
        total=int(row.total),
        blocked=int(row.blocked),
        flagged=int(row.flagged),
        masked=int(row.masked),
        allowed=int(row.allowed),
        avg_latency_ms=float(row.avg_latency_ms or 0.0),
        by_check=await _group_counts(session, guardrail_decisions.c.check, window),
        by_action=await _group_counts(session, guardrail_decisions.c.action_taken, window),
        by_mode=await _group_counts(session, guardrail_decisions.c.mode, window),
        window_hours=hours,
    )
