"""Leakage-free threshold + temperature tuning for the eval gate.

Pipeline per check, all driven from a single raw-probability pass:

1. **5-fold CV** (seeded, deterministic): fit a temperature on the train fold
   only, pick the best threshold on the calibrated train fold only, then score
   the val fold with those train-fit artifacts. Fold metrics aggregate to an
   honest mean ± std estimate — this is the number the gate should judge.
2. **Final deployable config**: refit temperature + threshold on ALL rows per
   check (optimistic — selected on the same data it scores).
3. **Baseline**: the same raw probabilities at default thresholds, no
   calibration, for the delta table.

`gate_verdict` from `metrics` produces every pass/fail so baseline, CV, and
final verdicts share one code path with `build_report`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict

from .calibration import TemperatureMap, calibrated, fit
from .config import DEFAULT_THRESHOLDS
from .metrics import (
    MAX_ECE,
    MIN_ACCURACY,
    BinaryMetrics,
    CheckResult,
    EvalReport,
    GateSection,
    MacroAverages,
    binary_metrics,
    build_report,
    ece,
    gate_verdict,
    mean_std,
    roc_auc,
    sweep_thresholds,
)

__all__ = [
    "CheckCV",
    "CheckData",
    "CheckFinal",
    "CvSection",
    "FinalSection",
    "MetricStats",
    "TuneResult",
    "cv_check",
    "final_check",
    "make_folds",
    "run_tune",
]


# ------------------------------------------------------------------ inputs


@dataclass
class CheckData:
    """Raw model output for one check: [P(false), P(true)] vectors + gold labels."""

    check: str
    probs: list[list[float]]
    labels: list[int]

    @property
    def n(self) -> int:
        return len(self.labels)


def make_folds(n: int, folds: int = 5, seed: int = 42, key: str = "") -> list[list[int]]:
    """Deterministic shuffled round-robin validation folds (sorted index lists).

    Same (n, folds, seed, key) always yields the same split; `key` (the check
    name) gives each check an independent but reproducible shuffle.
    """
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    if n < folds:
        raise ValueError(f"need at least folds={folds} rows to split, got {n}")
    rng = random.Random(f"{seed}:{key}")
    indices = list(range(n))
    rng.shuffle(indices)
    return [sorted(indices[fold::folds]) for fold in range(folds)]


def fit_temperature(
    vectors: Sequence[Sequence[float]], labels: Sequence[int], rows: Sequence[int]
) -> float:
    """NLL-fit a scalar temperature using ONLY the given row indices."""
    return fit([(vectors[i], labels[i]) for i in rows])


# ------------------------------------------------------------------ CV


class MetricStats(BaseModel):
    """Mean ± sample std of a metric across CV folds."""

    model_config = ConfigDict(extra="forbid")

    mean: float
    std: float


class CheckCV(BaseModel):
    """Honest CV estimate for one check: fold-aggregated metrics + fold choices."""

    model_config = ConfigDict(extra="forbid")

    check: str
    n: int
    folds: int
    accuracy: MetricStats
    ece: MetricStats
    precision: float
    recall: float
    f1: float
    chosen_thresholds: list[float]
    temperatures: list[float]


class CvSection(BaseModel):
    """CV aggregation: per-check estimates, macro means, and the honest gate verdict."""

    model_config = ConfigDict(extra="forbid")

    checks: dict[str, CheckCV]
    macro: MacroAverages
    gate: GateSection


def cv_check(
    data: CheckData,
    folds: int = 5,
    step: float = 0.05,
    seed: int = 42,
    n_bins: int = 10,
) -> CheckCV:
    """Leakage-free CV for one check: train-fit (temperature, threshold) → val score."""
    fold_ids = make_folds(data.n, folds=folds, seed=seed, key=data.check)
    all_rows = set(range(data.n))

    accuracies: list[float] = []
    eces: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    thresholds: list[float] = []
    temperatures: list[float] = []

    for val_rows in fold_ids:
        train_rows = sorted(all_rows - set(val_rows))

        # 1. fit temperature on the train fold only
        temperature = fit_temperature(data.probs, data.labels, train_rows)
        train_cal = [calibrated(data.probs[i], temperature)[1] for i in train_rows]
        train_labels = [data.labels[i] for i in train_rows]

        # 2. pick the threshold on the calibrated train fold only
        threshold = sweep_thresholds(train_cal, train_labels, step=step).best_threshold

        # 3. score the val fold with the train-fit artifacts
        val_cal = [calibrated(data.probs[i], temperature)[1] for i in val_rows]
        val_labels = [data.labels[i] for i in val_rows]
        metrics = binary_metrics(val_cal, val_labels, threshold)

        accuracies.append(metrics.accuracy)
        precisions.append(metrics.precision)
        recalls.append(metrics.recall)
        f1s.append(metrics.f1)
        eces.append(ece(val_cal, val_labels, n_bins))
        thresholds.append(threshold)
        temperatures.append(temperature)

    acc_mean, acc_std = mean_std(accuracies)
    ece_mean, ece_std = mean_std(eces)
    fold_count = len(fold_ids)
    return CheckCV(
        check=data.check,
        n=data.n,
        folds=fold_count,
        accuracy=MetricStats(mean=acc_mean, std=acc_std),
        ece=MetricStats(mean=ece_mean, std=ece_std),
        precision=sum(precisions) / fold_count,
        recall=sum(recalls) / fold_count,
        f1=sum(f1s) / fold_count,
        chosen_thresholds=thresholds,
        temperatures=temperatures,
    )


# ------------------------------------------------------------------ final fit


class CheckFinal(BaseModel):
    """Deployable per-check fit: temperature + threshold selected on ALL rows.

    Optimistic: the same rows score the metrics below.
    """

    model_config = ConfigDict(extra="forbid")

    check: str
    n: int
    temperature: float
    threshold: float
    metrics: BinaryMetrics
    ece: float


class FinalSection(BaseModel):
    """Final-fit results: per-check artifacts/metrics, macro means, gate verdict."""

    model_config = ConfigDict(extra="forbid")

    checks: dict[str, CheckFinal]
    macro: MacroAverages
    gate: GateSection
    pooled_temperature: float
    label: str


def final_check(data: CheckData, step: float = 0.05, n_bins: int = 10) -> CheckFinal:
    """Fit temperature + threshold on ALL rows of one check; score them."""
    rows = list(range(data.n))
    temperature = fit_temperature(data.probs, data.labels, rows)
    calibrated_probs = [calibrated(vector, temperature)[1] for vector in data.probs]
    sweep = sweep_thresholds(calibrated_probs, data.labels, step=step)
    return CheckFinal(
        check=data.check,
        n=data.n,
        temperature=temperature,
        threshold=sweep.best_threshold,
        metrics=sweep.best,
        ece=ece(calibrated_probs, data.labels, n_bins),
    )


# ------------------------------------------------------------------ assembly


class TuneResult(BaseModel):
    """Everything `--tune` reports: baseline, honest CV, final fit, deltas, verdict."""

    model_config = ConfigDict(extra="forbid")

    meta: dict[str, Any]
    baseline: EvalReport
    cv: CvSection
    final: FinalSection
    deltas: dict[str, dict[str, float]]
    roc_auc: dict[str, float | None]
    honest_assessment: str

    def temperature_map(self) -> TemperatureMap:
        """Deployable temperature map: pooled `default` + per-check noul entries."""
        noul: dict[str, float] = {"default": self.final.pooled_temperature}
        for check, final in self.final.checks.items():
            noul[check] = final.temperature
        return TemperatureMap(noul=noul)

    def tuned_thresholds(self) -> dict[str, float]:
        """Per-check thresholds from the final fit (GateLayaConfig shape)."""
        return {check: final.threshold for check, final in self.final.checks.items()}


def baseline_report(
    datas: Sequence[CheckData],
    thresholds: dict[str, float] | None = None,
    n_bins: int = 10,
) -> EvalReport:
    """Same raw probabilities at (default or given) thresholds, no calibration."""
    results = [
        CheckResult(
            check=data.check,
            probs=[vector[1] for vector in data.probs],
            labels=data.labels,
            threshold=float((thresholds or {}).get(data.check, DEFAULT_THRESHOLDS[data.check])),
        )
        for data in datas
    ]
    return build_report(results, gate=None, n_bins=n_bins)


def final_report(
    datas: Sequence[CheckData], finals: dict[str, CheckFinal], n_bins: int = 10
) -> EvalReport:
    """build_report over per-check calibrated probabilities + tuned thresholds."""
    results = [
        CheckResult(
            check=data.check,
            probs=[calibrated(vector, finals[data.check].temperature)[1] for vector in data.probs],
            labels=data.labels,
            threshold=finals[data.check].threshold,
        )
        for data in datas
    ]
    return build_report(results, gate=None, n_bins=n_bins)


def _cv_macro(cv_checks: dict[str, CheckCV]) -> MacroAverages:
    count = len(cv_checks)
    return MacroAverages(
        accuracy=sum(c.accuracy.mean for c in cv_checks.values()) / count,
        precision=sum(c.precision for c in cv_checks.values()) / count,
        recall=sum(c.recall for c in cv_checks.values()) / count,
        f1=sum(c.f1 for c in cv_checks.values()) / count,
        ece=sum(c.ece.mean for c in cv_checks.values()) / count,
    )


def _honest_assessment(
    baseline: EvalReport,
    cv: CvSection,
    finals: FinalSection,
    aucs: dict[str, float | None],
) -> str:
    """Plain-language verdict: is 0.90/0.15 reachable zero-shot, and what binds."""
    lines: list[str] = []
    fold_count = next(iter(cv.checks.values())).folds

    if cv.gate.passed:
        lines.append(
            f"Achievable zero-shot: honest {fold_count}-fold CV gate PASSES the PRODUCT.md "
            f"gate (macro accuracy {cv.macro.accuracy:.3f} >= {MIN_ACCURACY:.2f}, macro ECE "
            f"{cv.macro.ece:.3f} <= {MAX_ECE:.2f}) with per-check temperature scaling and "
            "train-selected thresholds."
        )
    else:
        lines.append(
            f"NOT achievable zero-shot on this dataset: honest {fold_count}-fold CV gate FAILS "
            f"(macro accuracy {cv.macro.accuracy:.3f} vs {MIN_ACCURACY:.2f}, macro ECE "
            f"{cv.macro.ece:.3f} vs {MAX_ECE:.2f}) with per-check temperature scaling and "
            "train-selected thresholds."
        )

    shortfalls: list[str] = []
    for check, result in cv.checks.items():
        if result.accuracy.mean < MIN_ACCURACY:
            shortfalls.append(f"{check} accuracy {result.accuracy.mean:.3f} < {MIN_ACCURACY:.2f}")
        if result.ece.mean > MAX_ECE:
            shortfalls.append(f"{check} ECE {result.ece.mean:.3f} > {MAX_ECE:.2f}")
    if shortfalls:
        lines.append("Still short after tuning: " + "; ".join(shortfalls) + ".")

    # binding constraint: ranking quality (AUC) vs threshold/calibration placement
    ranked = {check: auc for check, auc in aucs.items() if auc is not None}
    if not ranked:
        lines.append(
            "ROC AUC unavailable (single-class subset) — ranking ceiling not assessable."
        )
    else:
        worst_auc_check = min(ranked, key=lambda check: ranked[check])
        if finals.checks[worst_auc_check].metrics.accuracy < MIN_ACCURACY:
            lines.append(
                f"Binding constraint is ranking quality, not thresholds: {worst_auc_check} "
                f"ROC AUC {ranked[worst_auc_check]:.3f} — even the best single threshold on all "
                f"rows reaches only {finals.checks[worst_auc_check].metrics.accuracy:.3f} "
                "accuracy, so no calibration can close the gap (per-check ROC AUC: "
                + ", ".join(f"{check} {aucs[check]:.3f}" for check in ranked)
                + ")."
            )
        else:
            oracle = max(finals.checks[check].metrics.accuracy for check in finals.checks)
            lines.append(
                f"Ranking is sufficient (best per-check threshold accuracy up to {oracle:.3f}); "
                "the residual gap is threshold/calibration placement, and the CV-vs-baseline "
                "deltas show what tuning recovered."
            )

    lines.append(
        f"Calibration effect (ECE): baseline macro {baseline.macro.ece:.3f} → CV "
        f"{cv.macro.ece:.3f} → final {finals.macro.ece:.3f}. Final-fit numbers are OPTIMISTIC "
        "(temperature and threshold selected on the rows they score); trust the CV estimate."
    )
    return " ".join(lines)


def run_tune(
    datas: Sequence[CheckData],
    *,
    folds: int = 5,
    step: float = 0.05,
    seed: int = 42,
    n_bins: int = 10,
    thresholds: dict[str, float] | None = None,
    meta: dict[str, Any] | None = None,
) -> TuneResult:
    """Run baseline + leakage-free CV + final fit; assemble the full report."""
    if not datas:
        raise ValueError("run_tune needs at least one CheckData")

    baseline = baseline_report(datas, thresholds, n_bins)

    cv_checks = {
        data.check: cv_check(data, folds=folds, step=step, seed=seed, n_bins=n_bins)
        for data in datas
    }
    cv_macro = _cv_macro(cv_checks)
    cv = CvSection(
        checks=cv_checks,
        macro=cv_macro,
        gate=gate_verdict(cv_macro.accuracy, cv_macro.ece),
    )

    finals_by_check = {data.check: final_check(data, step=step, n_bins=n_bins) for data in datas}
    final_full_report = final_report(datas, finals_by_check, n_bins)
    pooled_samples = [
        (vector, label) for data in datas for vector, label in zip(data.probs, data.labels)
    ]
    finals = FinalSection(
        checks=finals_by_check,
        macro=final_full_report.macro,
        gate=final_full_report.gate,
        pooled_temperature=fit(pooled_samples),
        label="optimistic: temperature + threshold selected on the same rows they score",
    )

    baseline_by_check = {check.check: check for check in baseline.checks}
    deltas = {
        check: {
            "accuracy": cv_checks[check].accuracy.mean
            - baseline_by_check[check].metrics.accuracy,
            "ece": cv_checks[check].ece.mean - baseline_by_check[check].ece,
        }
        for check in cv_checks
    }

    aucs: dict[str, float | None] = {}
    for data in datas:
        try:
            aucs[data.check] = roc_auc([vector[1] for vector in data.probs], data.labels)
        except ValueError:
            # single-class subset (e.g. --limit over a positives-first file): AUC undefined
            aucs[data.check] = None

    return TuneResult(
        meta={
            "folds": folds,
            "seed": seed,
            "grid_step": step,
            "n_bins": n_bins,
            "rows": sum(data.n for data in datas),
            "checks": [data.check for data in datas],
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(meta or {}),
        },
        baseline=baseline,
        cv=cv,
        final=finals,
        deltas=deltas,
        roc_auc=aucs,
        honest_assessment=_honest_assessment(baseline, cv, finals, aucs),
    )
