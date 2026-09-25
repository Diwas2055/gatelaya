#!/usr/bin/env python3
"""Fit GateLaya temperature calibration from a labeled JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, fit, save_temperature_map
    from gatelaya.questions import build_questions, choice_probs, noul_probs
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, fit, save_temperature_map
    from gatelaya.questions import build_questions, choice_probs, noul_probs

VALID_CHECKS = ("pii", "injection", "toxicity", "secret_leak")

YES_LABELS = {"yes", "y", "true", "1", "positive", "pos"}
NO_LABELS = {"no", "n", "false", "0", "negative", "neg"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fit GateLaya temperatures from labeled JSONL and write calibration.json"
    )
    parser.add_argument(
        "--input", required=True, help='JSONL: {"text": ..., "label": ..., "check": ...}'
    )
    parser.add_argument(
        "--output", default="calibration.json", help="output temperature map JSON (default: calibration.json)"
    )
    parser.add_argument(
        "--check", choices=VALID_CHECKS, help="default check for rows without a 'check' field"
    )
    parser.add_argument("--english-checkpoint", default="convaiinnovations/laya")
    parser.add_argument("--multilingual-checkpoint", default="convaiinnovations/laya-multilingual")
    return parser.parse_args(argv)


def parse_label(label: Any, question: dict[str, Any], where: str) -> int:
    """Convert a JSONL label to the true option index for a question."""
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
        raise SystemExit(f"{where}: cannot parse noul label {label!r} (use yes/no or 0/1)")
    options = list(question["criteria"])
    if isinstance(label, int) and 0 <= label < len(options):
        return label
    if isinstance(label, str) and label in options:
        return options.index(label)
    raise SystemExit(f"{where}: label {label!r} not in options {options}")


def main(argv: list[str] | None = None) -> int:
    """Fit temperatures from --input JSONL and write the calibration map."""
    args = parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input file not found: {input_path}")

    agent = LayaRouterAgent(
        english_checkpoint=args.english_checkpoint,
        multilingual_checkpoint=args.multilingual_checkpoint,
    )

    # bucket -> list[(probs, true_label)]; bucket is ("noul", None) or ("choice", n)
    buckets: dict[tuple[str, int | None], list[tuple[list[float], int]]] = {}
    row_count = 0

    for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        where = f"{input_path}:{line_no}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{where}: invalid JSON ({exc})") from exc
        for field in ("text", "label"):
            if field not in row:
                raise SystemExit(f"{where}: missing required field {field!r}")
        check = row.get("check") or args.check
        if check not in VALID_CHECKS:
            raise SystemExit(f"{where}: missing/unknown check {check!r}; pass --check or add 'check'")
        question_name = row.get("question", check)
        questions, _ = build_questions([check])
        if question_name not in questions:
            raise SystemExit(f"{where}: unknown question {question_name!r} for check {check!r}")
        question = questions[question_name]

        result = agent.predict({"text": row["text"]}, {question_name: question})
        answers = result.get("answers", {})
        if question_name not in answers:
            raise SystemExit(f"{where}: agent returned no answer for {question_name!r}")
        answer = answers[question_name]
        label_idx = parse_label(row["label"], question, where)

        if question["type"] == "noul":
            probs = noul_probs(answer)
            bucket: tuple[str, int | None] = ("noul", None)
        else:
            options = list(question["criteria"])
            probs = choice_probs(answer, options)
            bucket = ("choice", len(options))
        buckets.setdefault(bucket, []).append((probs, label_idx))
        row_count += 1

    if row_count == 0:
        raise SystemExit("no labeled rows found in input")

    temperature_map = TemperatureMap()
    for (qtype, option_count), samples in sorted(buckets.items(), key=str):
        temperature = fit(samples)
        if qtype == "noul":
            temperature_map.noul["default"] = temperature
            label = "noul/default"
        else:
            temperature_map.choice[str(option_count)] = temperature
            label = f"choice/{option_count}"
        print(f"fitted {label}: n={len(samples)}  T={temperature:.3f}")

    save_temperature_map(temperature_map, args.output)
    print(f"wrote {args.output} ({row_count} labeled rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
