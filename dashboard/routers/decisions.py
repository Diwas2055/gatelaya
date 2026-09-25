"""Decision search over the guardrail_decisions audit log (sha256 only)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.sql.elements import ColumnElement

from gatelaya.audit import DecisionRecord, guardrail_decisions

from ..db import row_to_record, utc_dt
from ..dependencies import SessionDep

router = APIRouter(prefix="/api/decisions", tags=["decisions"])

MAX_LIMIT = 200

# Full DecisionRecord serialization (sha256 only — raw text never leaves the audit log).
DecisionOut = DecisionRecord


class DecisionListOut(BaseModel):
    """A page of decisions plus the filtered total."""

    items: list[DecisionOut]
    total: int


def _clauses(
    check: str | None,
    action: str | None,
    mode: str | None,
    sha256: str | None,
    since: datetime | None,
    until: datetime | None,
) -> list[ColumnElement[bool]]:
    """Build the exact-match/time-window WHERE clauses for a query."""
    clauses: list[ColumnElement[bool]] = []
    if check is not None:
        clauses.append(guardrail_decisions.c.check == check)
    if action is not None:
        clauses.append(guardrail_decisions.c.action_taken == action)
    if mode is not None:
        clauses.append(guardrail_decisions.c.mode == mode)
    if sha256 is not None:
        clauses.append(guardrail_decisions.c.input_sha256 == sha256)
    if since is not None:
        clauses.append(guardrail_decisions.c.timestamp >= utc_dt(since))
    if until is not None:
        clauses.append(guardrail_decisions.c.timestamp <= utc_dt(until))
    return clauses


@router.get("", response_model=DecisionListOut)
async def list_decisions(
    session: SessionDep,
    check: str | None = Query(None),
    action: str | None = Query(None),
    mode: str | None = Query(None),
    sha256: str | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
) -> DecisionListOut:
    """List decisions newest-first with optional filters (limit capped at 200)."""
    clauses = _clauses(check, action, mode, sha256, since, until)
    total = (
        await session.execute(
            select(func.count()).select_from(guardrail_decisions).where(*clauses)
        )
    ).scalar_one()
    rows = (
        await session.execute(
            select(guardrail_decisions)
            .where(*clauses)
            .order_by(guardrail_decisions.c.timestamp.desc(), guardrail_decisions.c.id.desc())
            .limit(min(limit, MAX_LIMIT))
            .offset(offset)
        )
    ).mappings()
    return DecisionListOut(items=[row_to_record(row) for row in rows], total=total)


@router.get("/{decision_id}", response_model=DecisionOut)
async def get_decision(decision_id: str, session: SessionDep) -> DecisionOut:
    """Fetch one decision by id (404 when unknown)."""
    row = (
        await session.execute(
            select(guardrail_decisions).where(guardrail_decisions.c.id == decision_id)
        )
    ).mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"decision {decision_id!r} not found")
    return row_to_record(row)
