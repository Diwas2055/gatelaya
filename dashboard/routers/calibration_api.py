"""Calibration map read/upload: fits temperatures via Laya (lazy, thread-offloaded)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from gatelaya.calibration import TemperatureMap, fit
from gatelaya.config import ALL_CHECKS
from gatelaya.errors import GuardrailConfigurationError, LayaNotInstalledError
from gatelaya.hotreload import env_flag
from gatelaya.questions import build_questions, choice_probs, noul_probs

from ..dependencies import SettingsDep
from .config_api import load_config

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

YES_LABELS = {"yes", "y", "true", "1", "positive", "pos"}
NO_LABELS = {"no", "n", "false", "0", "negative", "neg"}


class CalibrationView(BaseModel):
    """Stored temperature map plus its file path."""

    temperatures: dict[str, dict[str, float]] | None
    path: str
    exists: bool


class UploadAck(BaseModel):
    """Fitted temperatures written to disk (hot-reloaded unless disabled)."""

    temperatures: dict[str, dict[str, float]]
    n_rows: int
    path: str
    restart_required: bool
    hot_reload: bool


def _parse_rows(content: str, default_check: str | None) -> list[dict[str, Any]]:
    """Parse labeled JSONL rows; ValueError messages carry line context."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        where = f"line {line_no}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{where}: each line must be a JSON object")
        for field in ("text", "label"):
            if field not in row:
                raise ValueError(f"{where}: missing required field {field!r}")
        check = row.get("check") or default_check
        if check not in ALL_CHECKS:
            raise ValueError(
                f"{where}: missing/unknown check {check!r}; set 'check' or the form field"
            )
        question_name = row.get("question", check)
        questions, _ = build_questions([check])
        if question_name not in questions:
            raise ValueError(f"{where}: unknown question {question_name!r} for check {check!r}")
        rows.append({**row, "check": check, "question": question_name, "_where": where})
    if not rows:
        raise ValueError("no labeled rows found in input")
    return rows


def _label_index(label: Any, question: dict[str, Any], where: str) -> int:
    """Map a JSONL label onto the question's true option index."""
    if question["type"] == "noul":
        if isinstance(label, bool):
            return 1 if label else 0
        if isinstance(label, (int, float)) and label in (0, 1):
            return int(label)
        if isinstance(label, str):
            lowered = label.strip().lower()
            if lowered in YES_LABELS:
                return 1
            if lowered in NO_LABELS:
                return 0
        raise ValueError(f"{where}: cannot parse noul label {label!r} (use yes/no or 0/1)")
    options = list(question["criteria"])
    if isinstance(label, int) and 0 <= label < len(options):
        return label
    if isinstance(label, str) and label in options:
        return options.index(label)
    raise ValueError(f"{where}: label {label!r} not in options {options}")


def _fit_rows(
    rows: list[dict[str, Any]], english_checkpoint: str, multilingual_checkpoint: str
) -> tuple[TemperatureMap, int]:
    """Run Laya on every row and grid-fit per-bucket temperatures (blocking)."""
    from gatelaya.agent import LayaRouterAgent  # lazy: only the upload path needs it

    agent = LayaRouterAgent(
        english_checkpoint=english_checkpoint,
        multilingual_checkpoint=multilingual_checkpoint,
    )
    # bucket -> list[(probs, true_label)]; bucket is ("noul", None) or ("choice", n)
    buckets: dict[tuple[str, int | None], list[tuple[list[float], int]]] = {}
    for row in rows:
        question_name = row["question"]
        questions, _ = build_questions([row["check"]])
        question = questions[question_name]
        result = agent.predict({"text": row["text"]}, {question_name: question})
        answers = result.get("answers", {})
        if question_name not in answers:
            raise ValueError(f"{row['_where']}: agent returned no answer for {question_name!r}")
        answer = answers[question_name]
        label_idx = _label_index(row["label"], question, row["_where"])
        if question["type"] == "noul":
            probs = noul_probs(answer)
            bucket: tuple[str, int | None] = ("noul", None)
        else:
            options = list(question["criteria"])
            probs = choice_probs(answer, options)
            bucket = ("choice", len(options))
        buckets.setdefault(bucket, []).append((probs, label_idx))

    temperature_map = TemperatureMap()
    for (qtype, option_count), samples in sorted(buckets.items(), key=str):
        temperature = fit(samples)
        if qtype == "noul":
            temperature_map.noul["default"] = temperature
        else:
            temperature_map.choice[str(option_count)] = temperature
    return temperature_map, len(rows)


@router.get("", response_model=CalibrationView)
async def get_calibration(settings: SettingsDep) -> CalibrationView:
    """Return the current temperature map (null when not yet fitted)."""
    path = Path(settings.calibration_path)
    if not path.exists():
        return CalibrationView(temperatures=None, path=str(path), exists=False)
    try:
        temperatures = TemperatureMap.load(path).to_dict()
    except GuardrailConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CalibrationView(temperatures=temperatures, path=str(path), exists=True)


@router.post("/upload", response_model=UploadAck)
async def upload_calibration(
    settings: SettingsDep,
    file: UploadFile = File(...),
    check: str | None = Form(None),
) -> UploadAck:
    """Fit temperatures from an uploaded JSONL dataset and write calibration.json."""
    raw = await file.read()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"file is not UTF-8: {exc}") from exc
    try:
        rows = _parse_rows(content, check)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cfg, _ = load_config(Path(settings.config_path))
    try:
        temperatures, n_rows = await asyncio.to_thread(
            _fit_rows, rows, cfg.english_checkpoint, cfg.multilingual_checkpoint
        )
    except LayaNotInstalledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, GuardrailConfigurationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    temperatures.save(settings.calibration_path)
    hot_reload = env_flag("GATELAYA_HOT_RELOAD", True)
    return UploadAck(
        temperatures=temperatures.to_dict(),
        n_rows=n_rows,
        path=settings.calibration_path,
        restart_required=not hot_reload,
        hot_reload=hot_reload,
    )
