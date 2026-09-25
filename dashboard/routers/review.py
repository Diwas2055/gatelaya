"""Human-review queue for flagged decisions: labels + calibration JSONL export."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gatelaya.audit import guardrail_decisions

from ..db import review_labels, row_to_record, utc_dt
from ..dependencies import SessionDep
from .decisions import DecisionOut

router = APIRouter(prefix="/api/review", tags=["review"])

# operator verdict -> noul label used by scripts/calibrate.py
EXPORT_LABELS = {"allow": "no", "block": "yes", "mask": "yes"}


class LabelOut(BaseModel):
    """A human label attached to a flagged decision."""

    id: str
    decision_id: str
    label: str
    note: str | None = None
    text: str | None = None
    created_at: datetime


class ReviewItem(BaseModel):
    """One queue entry: the flagged decision plus its label (null while pending)."""

    decision: DecisionOut
    label: LabelOut | None = None


class ReviewListOut(BaseModel):
    """A page of review queue entries plus the filtered total."""

    items: list[ReviewItem]
    total: int


class LabelIn(BaseModel):
    """Operator verdict for a flagged decision; `text` feeds calibration export."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["allow", "block", "mask"]
    note: str | None = None
    text: str | None = None


class LabelAck(BaseModel):
    """Acknowledgement of a stored label."""

    ok: bool
    id: str


def _label_out(row: Mapping[str, Any]) -> LabelOut:
    """Build a LabelOut from a review_labels row."""
    data = dict(row)
    data["created_at"] = utc_dt(data["created_at"])
    return LabelOut(**data)


async def _labels_by_decision(session: AsyncSession, ids: list[str]) -> dict[str, LabelOut]:
    """Fetch labels for the given decision ids, keyed by decision_id."""
    if not ids:
        return {}
    rows = (
        await session.execute(select(review_labels).where(review_labels.c.decision_id.in_(ids)))
    ).mappings()
    return {row["decision_id"]: _label_out(row) for row in rows}


@router.get("", response_model=ReviewListOut)
async def list_review(
    session: SessionDep,
    status: str = Query("pending", pattern="^(pending|resolved|all)$"),
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
) -> ReviewListOut:
    """List flagged decisions: pending (unlabeled), resolved (labeled), or all."""
    limit = min(limit, 200)
    labeled = select(review_labels.c.decision_id).where(
        review_labels.c.decision_id == guardrail_decisions.c.id
    )
    base = select(guardrail_decisions).where(guardrail_decisions.c.action_taken == "flag")
    if status == "pending":
        base = base.where(~labeled.exists())
    elif status == "resolved":
        base = base.where(labeled.exists())
    total = (
        await session.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await session.execute(
            base.order_by(
                guardrail_decisions.c.timestamp.desc(), guardrail_decisions.c.id.desc()
            )
            .limit(limit)
            .offset(offset)
        )
    ).mappings()
    decisions = [row_to_record(row) for row in rows]
    labels = await _labels_by_decision(session, [rec.id for rec in decisions])
    return ReviewListOut(
        items=[ReviewItem(decision=rec, label=labels.get(rec.id)) for rec in decisions],
        total=total,
    )


@router.post("/{decision_id}/label", response_model=LabelAck)
async def label_decision(decision_id: str, body: LabelIn, session: SessionDep) -> LabelAck:
    """Attach an operator label (404 unknown decision, 409 already labeled)."""
    known = (
        await session.execute(
            select(func.count()).select_from(guardrail_decisions).where(
                guardrail_decisions.c.id == decision_id
            )
        )
    ).scalar_one()
    if known == 0:
        raise HTTPException(status_code=404, detail=f"decision {decision_id!r} not found")
    label_id = uuid.uuid4().hex
    try:
        await session.execute(
            insert(review_labels).values(
                id=label_id,
                decision_id=decision_id,
                label=body.label,
                note=body.note,
                text=body.text,
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail=f"decision {decision_id!r} already labeled"
        ) from None
    return LabelAck(ok=True, id=label_id)


@router.get("/export.jsonl")
async def export_labels(
    session: SessionDep,
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
) -> Response:
    """JSONL calibration export: labeled flagged decisions that carry text."""
    stmt = (
        select(guardrail_decisions.c.check, review_labels.c.text, review_labels.c.label)
        .select_from(
            guardrail_decisions.join(
                review_labels, review_labels.c.decision_id == guardrail_decisions.c.id
            )
        )
        .where(
            guardrail_decisions.c.action_taken == "flag",
            review_labels.c.text.is_not(None),
        )
    )
    if since is not None:
        stmt = stmt.where(guardrail_decisions.c.timestamp >= utc_dt(since))
    if until is not None:
        stmt = stmt.where(guardrail_decisions.c.timestamp <= utc_dt(until))
    rows = (
        await session.execute(stmt.order_by(guardrail_decisions.c.timestamp.asc()))
    ).all()
    lines = [
        json.dumps(
            {
                "text": text,
                "label": EXPORT_LABELS[label],
                "check": check,
                "question": check,
            },
            ensure_ascii=False,
        )
        for check, text, label in rows
    ]
    content = "\n".join(lines) + ("\n" if lines else "")
    return Response(content=content, media_type="text/plain")
