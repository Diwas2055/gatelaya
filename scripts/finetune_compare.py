#!/usr/bin/env python3
"""Compare base vs fine-tuned Laya on the held-out TEST split.

Honest protocol for both models:
  1. predict VAL and TEST exactly once;
  2. fit per-check temperature scaling on VAL only;
  3. sweep thresholds on VAL only;
  4. evaluate TEST once with those temperatures/thresholds.

Writes evals/results/finetune-test.json: per-check accuracy/ECE/ROC AUC +
macro gate verdict for both models, plus the delta.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

try:
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, calibrated, fit
    from gatelaya.metrics import CheckResult, EvalReport, build_report, roc_auc, sweep_thresholds
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gatelaya.agent import LayaRouterAgent
    from gatelaya.calibration import TemperatureMap, calibrated, fit
    from gatelaya.metrics import CheckResult, EvalReport, build_report, roc_auc, sweep_thresholds

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval.py"
DEFAULT_SPLITS = ROOT / "evals" / "data" / "splits"
DEFAULT_OUT = ROOT / "evals" / "results" / "finetune-test.json"
DEFAULT_MODEL = ROOT / "evals" / "models" / "gatelaya-ft"


def _load_eval_module() -> Any:
    """Import scripts/eval.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gatelaya_eval_for_compare", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="fine-tuned checkpoint")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args(argv)


def load_split_grouped(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load one JSONL split grouped by check (first-seen order)."""
    if not path.exists():
        raise SystemExit(f"missing split: {path} (run scripts/prepare_data.py)")
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(row["check"], []).append(row)
    return grouped


def fit_temperatures(raw_val: dict[str, Any]) -> TemperatureMap:
    """Grid-fit scalar temperature per check on VAL (pooled value as default)."""
    pooled: list[tuple[list[float], int]] = []
    noul: dict[str, float] = {}
    for check, data in raw_val.items():
        samples = list(zip(data.probs, data.labels))
        noul[check] = fit(samples)
        pooled.extend(samples)
    noul["default"] = fit(pooled)
    return TemperatureMap(noul=noul)


def val_thresholds(raw_val: dict[str, Any], temperatures: TemperatureMap) -> dict[str, float]:
    """Best threshold per check on temperature-scaled VAL probabilities."""
    thresholds: dict[str, float] = {}
    for check, data in raw_val.items():
        t = temperatures.noul_temperature(check)
        cal = [calibrated(vector, t)[1] for vector in data.probs]
        thresholds[check] = sweep_thresholds(cal, data.labels).best_threshold
    return thresholds


def evaluate_split(
    raw: dict[str, Any],
    temperatures: TemperatureMap,
    thresholds: dict[str, float],
    n_bins: int,
) -> tuple[EvalReport, dict[str, float]]:
    """Apply val-fit temps/thresholds to a split; returns (report, per-check ROC AUC)."""
    results: list[CheckResult] = []
    aucs: dict[str, float] = {}
    for check, data in raw.items():
        t = temperatures.noul_temperature(check)
        cal = [calibrated(vector, t)[1] for vector in data.probs]
        results.append(
            CheckResult(check=check, probs=cal, labels=data.labels, threshold=thresholds[check])
        )
        aucs[check] = round(roc_auc(cal, data.labels), 4)
    return build_report(results, gate=None, n_bins=n_bins), aucs


async def evaluate_model(
    agent: Any, val_grouped: dict[str, Any], test_grouped: dict[str, Any], n_bins: int
) -> dict[str, Any]:
    """Full honest protocol for one model: predict, fit on val, evaluate on test."""
    eval_mod = _load_eval_module()
    raw_val = await eval_mod.predict_raw(val_grouped, agent)
    raw_test = await eval_mod.predict_raw(test_grouped, agent)
    temperatures = fit_temperatures(raw_val)
    thresholds = val_thresholds(raw_val, temperatures)
    report, aucs = evaluate_split(raw_test, temperatures, thresholds, n_bins)
    return {
        "temperatures": temperatures.noul,
        "thresholds": thresholds,
        "val": {
            check: {"n": len(d.labels), "pos": sum(d.labels)} for check, d in raw_val.items()
        },
        "test_report": report.model_dump(mode="json"),
        "test_roc_auc": aucs,
    }


def comparison_table(baseline: dict[str, Any], finetuned: dict[str, Any]) -> str:
    """Per-check + macro comparison lines for stdout."""
    lines = [
        f"{'check':<12} {'base acc':>9} {'ft acc':>8} {'base ece':>9} "
        f"{'ft ece':>8} {'base auc':>9} {'ft auc':>8}",
        "-" * 67,
    ]
    b_checks = {c["check"]: c for c in baseline["test_report"]["checks"]}
    f_checks = {c["check"]: c for c in finetuned["test_report"]["checks"]}
    for check in sorted(b_checks):
        b, f = b_checks[check], f_checks[check]
        lines.append(
            f"{check:<12} {b['metrics']['accuracy']:>9.3f} {f['metrics']['accuracy']:>8.3f} "
            f"{b['ece']:>9.3f} {f['ece']:>8.3f} "
            f"{baseline['test_roc_auc'][check]:>9.3f} {finetuned['test_roc_auc'][check]:>8.3f}"
        )
    bm, fm = baseline["test_report"]["macro"], finetuned["test_report"]["macro"]
    lines.append("-" * 67)
    lines.append(
        f"{'macro':<12} {bm['accuracy']:>9.3f} {fm['accuracy']:>8.3f} "
        f"{bm['ece']:>9.3f} {fm['ece']:>8.3f}"
    )
    for name, block in (("baseline", baseline), ("finetuned", finetuned)):
        gate = block["test_report"]["gate"]
        verdict = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"gate {name:<9}: {verdict}  {gate['failures'] or ''}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the baseline-vs-finetuned comparison and write the JSON report."""
    args = parse_args(argv)
    if not args.model.exists():
        raise SystemExit(f"fine-tuned checkpoint not found: {args.model}")

    val_grouped = load_split_grouped(args.splits / "val.jsonl")
    test_grouped = load_split_grouped(args.splits / "test.jsonl")
    test_rows = sum(len(v) for v in test_grouped.values())
    print(f"val rows {sum(len(v) for v in val_grouped.values())} | test rows {test_rows}")

    print("evaluating BASELINE (routed base checkpoints) ...")
    baseline_agent = LayaRouterAgent()
    baseline = asyncio.run(evaluate_model(baseline_agent, val_grouped, test_grouped, args.bins))
    del baseline_agent
    gc.collect()

    print(f"evaluating FINETUNED ({args.model}) ...")
    ft_agent = LayaRouterAgent(model_path=str(args.model))
    finetuned = asyncio.run(evaluate_model(ft_agent, val_grouped, test_grouped, args.bins))
    del ft_agent
    gc.collect()

    bm = baseline["test_report"]["macro"]
    fm = finetuned["test_report"]["macro"]
    payload = {
        "protocol": "temps+thresholds fit on val, evaluated once on test",
        "test_rows": test_rows,
        "model": str(args.model),
        "baseline": baseline,
        "finetuned": finetuned,
        "delta": {
            "macro_accuracy": round(fm["accuracy"] - bm["accuracy"], 4),
            "macro_ece": round(fm["ece"] - bm["ece"], 4),
            "baseline_gate_passed": baseline["test_report"]["gate"]["passed"],
            "finetuned_gate_passed": finetuned["test_report"]["gate"]["passed"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(comparison_table(baseline, finetuned))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
