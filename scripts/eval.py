#!/usr/bin/env python3
"""Run GateLaya checks over a labeled eval dataset; report metrics and gate verdict."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

try:
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, calibrated
    from gatelaya.config import ALL_CHECKS, GateLayaConfig
    from gatelaya.errors import LayaNotInstalledError
    from gatelaya.metrics import CheckResult, EvalReport, build_report
    from gatelaya.questions import build_questions, noul_probs
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, calibrated
    from gatelaya.config import ALL_CHECKS, GateLayaConfig
    from gatelaya.errors import LayaNotInstalledError
    from gatelaya.metrics import CheckResult, EvalReport, build_report
    from gatelaya.questions import build_questions, noul_probs

VALID_CHECKS: tuple[str, ...] = ALL_CHECKS
LANGS: tuple[str, ...] = ("en", "np", "es", "fr", "de", "hi", "ar")
SANITY_WARNING = "*** SANITY MODE (--agent fake): deterministic stub — NOT real results ***"


class SanityAgent:
    """Deterministic stub: P(true) = 0.9 when the current row's gold label is 1 else 0.1."""

    def __init__(self) -> None:
        self.truth = 0

    def predict(self, state: dict, questions: dict) -> dict:
        """Answer every requested question with the label-derived probability."""
        p_true = 0.9 if self.truth else 0.1
        return {"answers": {name: {"noul": p_true} for name in questions}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate GateLaya checks on labeled JSONL and report metrics/gate"
    )
    parser.add_argument(
        "--data", default="evals/data",
        help="dataset directory (all *.jsonl) or a single JSONL file (default: evals/data)",
    )
    parser.add_argument(
        "--check", action="append", choices=VALID_CHECKS,
        help="only evaluate this check (repeatable)",
    )
    parser.add_argument("--config", help="gatelaya.yaml for thresholds (default: built-in config)")
    parser.add_argument("--calibration", help="calibration.json temperature map (optional)")
    parser.add_argument("--limit", type=int, help="max rows per check (smoke runs)")
    parser.add_argument("--report", help="write the EvalReport JSON to this path")
    parser.add_argument("--lang", choices=LANGS, help="only evaluate rows in this language")
    parser.add_argument(
        "--gate", action="store_true",
        help="exit 1 when the gate fails (PRODUCT.md: accuracy >= 0.90, ECE <= 0.15)",
    )
    parser.add_argument("--bins", type=int, default=10, help="ECE bins (default: 10)")
    parser.add_argument(
        "--agent", choices=("laya", "fake"), default="laya",
        help="predict with the real Laya model or a deterministic sanity stub",
    )
    return parser.parse_args(argv)


def load_rows(
    data: Path, checks: list[str] | None, lang: str | None, limit: int | None
) -> dict[str, list[dict[str, Any]]]:
    """Load and filter JSONL rows, grouped by check in first-seen order."""
    if not data.exists():
        raise SystemExit(f"data path not found: {data}")
    paths = sorted(data.glob("*.jsonl")) if data.is_dir() else [data]
    if not paths:
        raise SystemExit(f"no *.jsonl files in {data}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            where = f"{path}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{where}: invalid JSON ({exc})") from exc
            for field in ("text", "label", "check", "lang"):
                if field not in row:
                    raise SystemExit(f"{where}: missing required field {field!r}")
            if row["label"] not in (0, 1):
                raise SystemExit(f"{where}: label must be 0 or 1, got {row['label']!r}")
            if row["check"] not in VALID_CHECKS:
                raise SystemExit(f"{where}: unknown check {row['check']!r}")
            if checks and row["check"] not in checks:
                continue
            if lang and row["lang"] != lang:
                continue
            grouped.setdefault(row["check"], []).append(row)

    if not grouped:
        raise SystemExit("no rows matched the given filters")
    if limit is not None:
        if limit < 1:
            raise SystemExit(f"--limit must be >= 1, got {limit}")
        grouped = {check: rows[:limit] for check, rows in grouped.items()}
    return grouped


async def _predict(agent: Any, text: str, questions: dict) -> dict:
    """Thread-offload one forward pass so the event loop never blocks on inference."""
    return await asyncio.to_thread(agent.predict, {"text": text}, questions)


async def evaluate(
    grouped: dict[str, list[dict[str, Any]]],
    agent: Any,
    config: GateLayaConfig,
    temperatures: TemperatureMap | None,
) -> list[CheckResult]:
    """Predict every row and collect per-check probabilities/labels/thresholds."""
    results: list[CheckResult] = []
    for check, rows in grouped.items():
        questions, _ = build_questions([check])
        noul_question = {check: questions[check]}
        probs: list[float] = []
        labels: list[int] = []
        for row in rows:
            if isinstance(agent, SanityAgent):
                agent.truth = int(row["label"])
            result = await _predict(agent, row["text"], noul_question)
            answers = result.get("answers", {}) if isinstance(result, dict) else {}
            if check not in answers:
                raise SystemExit(f"agent returned no answer for {check!r}")
            vector = noul_probs(answers[check])
            if temperatures is not None:
                vector = calibrated(vector, temperatures.temperature_for("noul"))
            probs.append(vector[1])
            labels.append(int(row["label"]))
        results.append(
            CheckResult(
                check=check, probs=probs, labels=labels, threshold=config.threshold(check)
            )
        )
    return results


def print_table(report: EvalReport) -> None:
    """Print per-check metrics, macro averages, and the gate verdict."""
    header = f"{'check':<14}{'n':>6}{'acc':>9}{'P':>9}{'R':>9}{'F1':>9}{'ECE':>9}"
    print(header)
    print("-" * len(header))
    for check in report.checks:
        m = check.metrics
        print(
            f"{check.check:<14}{m.n:>6}{m.accuracy:>9.3f}{m.precision:>9.3f}"
            f"{m.recall:>9.3f}{m.f1:>9.3f}{check.ece:>9.3f}"
        )
    macro = report.macro
    print("-" * len(header))
    print(
        f"{'macro':<14}{report.n:>6}{macro.accuracy:>9.3f}{macro.precision:>9.3f}"
        f"{macro.recall:>9.3f}{macro.f1:>9.3f}{macro.ece:>9.3f}"
    )
    gate = report.gate
    verdict = "PASS" if gate.passed else "FAIL"
    print(
        f"\nGATE {verdict} (min_accuracy {gate.target['min_accuracy']:.2f}, "
        f"max_ece {gate.target['max_ece']:.2f})"
    )
    for failure in gate.failures:
        print(f"  - {failure}")


def main(argv: list[str] | None = None) -> int:
    """Run the eval; returns 0 ok, 1 gate failed, 2 laya missing."""
    args = parse_args(argv)
    if args.bins < 1:
        raise SystemExit(f"--bins must be >= 1, got {args.bins}")

    grouped = load_rows(Path(args.data), args.check, args.lang, args.limit)
    config = GateLayaConfig.from_yaml(args.config) if args.config else GateLayaConfig()
    temperatures = TemperatureMap.load(args.calibration) if args.calibration else None
    agent: Any = SanityAgent() if args.agent == "fake" else LayaRouterAgent(
        english_checkpoint=config.english_checkpoint,
        multilingual_checkpoint=config.multilingual_checkpoint,
    )
    if args.agent == "fake":
        print(SANITY_WARNING, file=sys.stderr)

    try:
        results = asyncio.run(evaluate(grouped, agent, config, temperatures))
    except LayaNotInstalledError:
        print(
            "Laya is not installed. Install it with: pip install laya\n"
            "(or: pip install 'gatelaya[model]')",
            file=sys.stderr,
        )
        return 2

    report = build_report(results, gate=None, n_bins=args.bins)
    print_table(report)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")
    return 1 if (args.gate and not report.gate.passed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
