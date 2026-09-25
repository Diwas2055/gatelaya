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
    from gatelaya.calibration import TemperatureMap, calibrated, save_temperature_map
    from gatelaya.config import ALL_CHECKS, GateLayaConfig
    from gatelaya.errors import LayaNotInstalledError
    from gatelaya.metrics import CheckResult, EvalReport, build_report
    from gatelaya.questions import build_questions, noul_probs
    from gatelaya.tune import CheckData, TuneResult, run_tune
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, calibrated, save_temperature_map
    from gatelaya.config import ALL_CHECKS, GateLayaConfig
    from gatelaya.errors import LayaNotInstalledError
    from gatelaya.metrics import CheckResult, EvalReport, build_report
    from gatelaya.questions import build_questions, noul_probs
    from gatelaya.tune import CheckData, TuneResult, run_tune

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
    tune = parser.add_argument_group("tuning")
    tune.add_argument(
        "--tune", action="store_true",
        help="run leakage-free per-check CV threshold+temperature tuning on the raw "
        "probabilities (predicts each row once); with --gate, the exit code judges "
        "the honest CV metrics",
    )
    tune.add_argument("--folds", type=int, default=5, help="CV folds for --tune (default: 5)")
    tune.add_argument(
        "--grid-step", type=float, default=0.05,
        help="threshold sweep step for --tune (default: 0.05 → grid 0.05..0.95)",
    )
    tune.add_argument("--seed", type=int, default=42, help="fold shuffle seed (default: 42)")
    tune.add_argument(
        "--tune-report", default="evals/results/tuned.json",
        help="tuning report JSON (default: evals/results/tuned.json)",
    )
    tune.add_argument(
        "--tune-config", default="evals/results/tuned-config.yaml",
        help="deployable GateLayaConfig YAML with tuned thresholds "
        "(default: evals/results/tuned-config.yaml)",
    )
    tune.add_argument(
        "--tune-calibration", default="evals/results/tuned-calibration.json",
        help="temperature map JSON (default: evals/results/tuned-calibration.json)",
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


async def predict_raw(
    grouped: dict[str, list[dict[str, Any]]], agent: Any
) -> dict[str, CheckData]:
    """Predict every row exactly ONCE; raw [P(false), P(true)] vectors per check.

    All analyses (baseline / CV / final fit) reuse this single pass.
    """
    raw: dict[str, CheckData] = {}
    for check, rows in grouped.items():
        questions, _ = build_questions([check])
        noul_question = {check: questions[check]}
        vectors: list[list[float]] = []
        labels: list[int] = []
        for row in rows:
            if isinstance(agent, SanityAgent):
                agent.truth = int(row["label"])
            result = await _predict(agent, row["text"], noul_question)
            answers = result.get("answers", {}) if isinstance(result, dict) else {}
            if check not in answers:
                raise SystemExit(f"agent returned no answer for {check!r}")
            vectors.append(noul_probs(answers[check]))
            labels.append(int(row["label"]))
        raw[check] = CheckData(check=check, probs=vectors, labels=labels)
    return raw


async def evaluate(
    grouped: dict[str, list[dict[str, Any]]],
    agent: Any,
    config: GateLayaConfig,
    temperatures: TemperatureMap | None,
) -> list[CheckResult]:
    """Predict every row and collect per-check probabilities/labels/thresholds."""
    raw = await predict_raw(grouped, agent)
    return results_from_raw(raw, config, temperatures)


def results_from_raw(
    raw: dict[str, CheckData],
    config: GateLayaConfig,
    temperatures: TemperatureMap | None,
) -> list[CheckResult]:
    """Apply temperatures/thresholds to raw predictions (no extra model passes)."""
    results: list[CheckResult] = []
    for check, data in raw.items():
        if temperatures is not None:
            temperature = temperatures.noul_temperature(check)
            p_true = [calibrated(vector, temperature)[1] for vector in data.probs]
        else:
            p_true = [vector[1] for vector in data.probs]
        results.append(
            CheckResult(
                check=check, probs=p_true, labels=data.labels, threshold=config.threshold(check)
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


def print_tune_table(result: TuneResult) -> None:
    """Print baseline vs CV vs final per check, plus the three gate verdicts."""
    header = (
        f"{'check':<14}{'acc base':>10}{'acc cv':>16}{'acc final':>11}"
        f"{'ece base':>10}{'ece cv':>16}{'ece final':>11}{'roc_auc':>9}{'thr':>7}"
    )
    print(header)
    print("-" * len(header))
    baseline_by_check = {check.check: check for check in result.baseline.checks}
    for check, cv in result.cv.checks.items():
        baseline = baseline_by_check[check]
        final = result.final.checks[check]
        auc = result.roc_auc[check]
        auc_text = f"{auc:>9.3f}" if auc is not None else f"{'n/a':>9}"
        print(
            f"{check:<14}"
            f"{baseline.metrics.accuracy:>10.3f}"
            f"{cv.accuracy.mean:>11.3f}±{cv.accuracy.std:<4.3f}"
            f"{final.metrics.accuracy:>11.3f}"
            f"{baseline.ece:>10.3f}"
            f"{cv.ece.mean:>11.3f}±{cv.ece.std:<4.3f}"
            f"{final.ece:>11.3f}"
            f"{auc_text}"
            f"{final.threshold:>7.2f}"
        )
    print("-" * len(header))
    base, cvm, fin = result.baseline.macro, result.cv.macro, result.final.macro
    print(
        f"{'macro':<14}{base.accuracy:>10.3f}{cvm.accuracy:>16.3f}{fin.accuracy:>11.3f}"
        f"{base.ece:>10.3f}{cvm.ece:>16.3f}{fin.ece:>11.3f}"
    )
    print(
        f"\nGATE baseline: {'PASS' if result.baseline.gate.passed else 'FAIL'} | "
        f"CV (honest): {'PASS' if result.cv.gate.passed else 'FAIL'} | "
        f"final (optimistic): {'PASS' if result.final.gate.passed else 'FAIL'}"
    )
    for label, gate in (
        ("baseline", result.baseline.gate),
        ("CV", result.cv.gate),
        ("final", result.final.gate),
    ):
        for failure in gate.failures:
            print(f"  - {label}: {failure}")
    print(f"\n{result.honest_assessment}")


def write_tune_artifacts(args: argparse.Namespace, config: GateLayaConfig, result: TuneResult) -> None:
    """Write the tuning report JSON, deployable config YAML, and calibration JSON."""
    report_path = Path(args.tune_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {report_path}")

    config_path = Path(args.tune_config)
    tuned_config = GateLayaConfig(
        thresholds=result.tuned_thresholds(),
        calibration_path=Path(args.tune_calibration),
        english_checkpoint=config.english_checkpoint,
        multilingual_checkpoint=config.multilingual_checkpoint,
        enabled_checks=config.enabled_checks,
    )
    tuned_config.to_yaml(config_path)
    print(f"wrote {config_path}")

    save_temperature_map(result.temperature_map(), args.tune_calibration)
    print(f"wrote {args.tune_calibration}")


def main(argv: list[str] | None = None) -> int:
    """Run the eval; returns 0 ok, 1 gate failed, 2 laya missing."""
    args = parse_args(argv)
    if args.bins < 1:
        raise SystemExit(f"--bins must be >= 1, got {args.bins}")
    if args.tune:
        if args.folds < 2:
            raise SystemExit(f"--folds must be >= 2, got {args.folds}")
        if not 0.0 < args.grid_step <= 1.0:
            raise SystemExit(f"--grid-step must be in (0, 1], got {args.grid_step}")

    grouped = load_rows(Path(args.data), args.check, args.lang, args.limit)
    config = GateLayaConfig.from_yaml(args.config) if args.config else GateLayaConfig()
    temperatures = TemperatureMap.load(args.calibration) if args.calibration else None
    if args.tune and temperatures is not None:
        print(
            "note: --calibration is ignored with --tune (baseline is uncalibrated)",
            file=sys.stderr,
        )
        temperatures = None
    agent: Any = SanityAgent() if args.agent == "fake" else LayaRouterAgent(
        english_checkpoint=config.english_checkpoint,
        multilingual_checkpoint=config.multilingual_checkpoint,
    )
    if args.agent == "fake":
        print(SANITY_WARNING, file=sys.stderr)

    try:
        raw = asyncio.run(predict_raw(grouped, agent))
    except LayaNotInstalledError:
        print(
            "Laya is not installed. Install it with: pip install laya\n"
            "(or: pip install 'gatelaya[model]')",
            file=sys.stderr,
        )
        return 2

    if args.tune:
        try:
            result = run_tune(
                list(raw.values()),
                folds=args.folds,
                step=args.grid_step,
                seed=args.seed,
                n_bins=args.bins,
                thresholds=config.thresholds,
                meta={
                    "agent": args.agent,
                    "sanity_mode": args.agent == "fake",
                    "data": str(args.data),
                    "config": args.config,
                },
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print_tune_table(result)
        write_tune_artifacts(args, config, result)
        if args.report:
            _write_json(Path(args.report), result.baseline)
        # honest gate: with --gate, the exit code follows the CV estimate
        return 1 if (args.gate and not result.cv.gate.passed) else 0

    results = results_from_raw(raw, config, temperatures)
    report = build_report(results, gate=None, n_bins=args.bins)
    print_table(report)
    if args.report:
        _write_json(Path(args.report), report)
    return 1 if (args.gate and not report.gate.passed) else 0


def _write_json(path: Path, payload: EvalReport) -> None:
    """Write an EvalReport (or report-shaped model) as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
