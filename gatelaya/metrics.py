"""Eval metrics: binary scores, calibration error, sweeps, ROC AUC, and gate reports."""

from __future__ import annotations

import math
from typing import Sequence

from pydantic import BaseModel, ConfigDict

# Single source of truth — PRODUCT.md "Success Criteria":
#   "Injection/PII block accuracy >= 0.90 and Mean ECE <= 0.15"
MIN_ACCURACY = 0.90
MAX_ECE = 0.15


class BinaryMetrics(BaseModel):
    """Confusion counts and derived scores for one check at one threshold."""

    model_config = ConfigDict(extra="forbid")

    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int
    n: int


class Bin(BaseModel):
    """One calibration bin: count, mean confidence, positive fraction, |gap|."""

    model_config = ConfigDict(extra="forbid")

    n: int
    mean_conf: float
    frac_positive: float
    gap: float


class CheckResult(BaseModel):
    """Predicted P(true) probabilities, gold labels, and threshold for one check."""

    model_config = ConfigDict(extra="forbid")

    check: str
    probs: list[float]
    labels: list[int]
    threshold: float


class CheckEval(BaseModel):
    """Evaluation of one check: binary metrics, ECE, and calibration bins."""

    model_config = ConfigDict(extra="forbid")

    check: str
    threshold: float
    metrics: BinaryMetrics
    ece: float
    bins: list[Bin]


class MacroAverages(BaseModel):
    """Unweighted mean of each score across the evaluated checks."""

    model_config = ConfigDict(extra="forbid")

    accuracy: float
    precision: float
    recall: float
    f1: float
    ece: float


class ThresholdPoint(BaseModel):
    """Metrics at one threshold of a sweep grid."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    metrics: BinaryMetrics


class SweepResult(BaseModel):
    """Per-threshold metrics over a grid plus the best pick.

    Best pick: highest accuracy, then highest F1, then threshold closest to 0.5,
    then lowest threshold (final deterministic tie-break, favors recall).
    """

    model_config = ConfigDict(extra="forbid")

    step: float
    points: list[ThresholdPoint]
    best_threshold: float
    best: BinaryMetrics


class GateSection(BaseModel):
    """Gate verdict: targets, pass/fail, and human-readable failure lines."""

    model_config = ConfigDict(extra="forbid")

    target: dict[str, float]
    passed: bool
    failures: list[str]


class EvalReport(BaseModel):
    """Full eval output: per-check results, macro averages, and gate verdict."""

    model_config = ConfigDict(extra="forbid")

    checks: list[CheckEval]
    macro: MacroAverages
    gate: GateSection
    n: int


def _validate(probs: Sequence[float], labels: Sequence[int], n_bins: int) -> tuple[list[float], list[int]]:
    """Check shapes/ranges shared by metric functions; returns normalized copies."""
    if len(probs) != len(labels):
        raise ValueError(f"probs length {len(probs)} != labels length {len(labels)}")
    if not probs:
        raise ValueError("probs/labels must be non-empty")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    clean_probs = [float(p) for p in probs]
    clean_labels = [int(l) for l in labels]
    for p in clean_probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of [0, 1]: {p}")
    for label in clean_labels:
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label}")
    return clean_probs, clean_labels


def binary_metrics(
    probs: Sequence[float], labels: Sequence[int], threshold: float
) -> BinaryMetrics:
    """Score `prob >= threshold` against labels; precision/recall are 0.0 when undefined."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    p, l = _validate(probs, labels, n_bins=1)
    tp = fp = tn = fn = 0
    for prob, label in zip(p, l):
        predicted = prob >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    n = len(p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return BinaryMetrics(
        accuracy=(tp + tn) / n,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        n=n,
    )


def threshold_grid(step: float = 0.05) -> list[float]:
    """Interior threshold grid (0, 1): k*step for k = 1 .. round(1/step) - 1.

    Default step 0.05 → [0.05, 0.10, ..., 0.95].
    """
    if not 0.0 < step <= 1.0:
        raise ValueError(f"step must be in (0, 1], got {step}")
    count = int(round(1.0 / step))
    if count < 2:
        raise ValueError(f"step {step} is too coarse: no interior thresholds")
    return [round(k * step, 10) for k in range(1, count)]


def sweep_thresholds(
    probs: Sequence[float], labels: Sequence[int], step: float = 0.05
) -> SweepResult:
    """Evaluate `binary_metrics` at every threshold on the grid; pick the best.

    Best by accuracy → F1 → threshold closest to 0.5 → lowest threshold.
    """
    p, l = _validate(probs, labels, n_bins=1)
    points = [
        ThresholdPoint(threshold=threshold, metrics=binary_metrics(p, l, threshold))
        for threshold in threshold_grid(step)
    ]

    def rank(point: ThresholdPoint) -> tuple[float, float, float, float]:
        return (
            point.metrics.accuracy,
            point.metrics.f1,
            -abs(point.threshold - 0.5),
            -point.threshold,
        )

    best = max(points, key=rank)
    return SweepResult(
        step=step, points=points, best_threshold=best.threshold, best=best.metrics
    )


def roc_auc(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based ROC AUC (Mann-Whitney U with average ranks for ties).

    Perfect separation → 1.0, inverted → 0.0, random scoring → ~0.5.
    Raises ValueError when only one class is present (AUC undefined).
    """
    p, l = _validate(probs, labels, n_bins=1)
    positives = sum(l)
    negatives = len(l) - positives
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"roc_auc needs both classes; got {positives} positive, {negatives} negative"
        )
    order = sorted(range(len(p)), key=lambda i: p[i])
    ranks = [0.0] * len(p)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        # ranks i+1 .. j+1 (1-based) collapse to their average for ties
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    rank_sum_positives = sum(rank for rank, label in zip(ranks, l) if label == 1)
    return (rank_sum_positives - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """(mean, sample standard deviation); a single value has std 0.0."""
    if not values:
        raise ValueError("values must be non-empty")
    vals = [float(v) for v in values]
    count = len(vals)
    mean = sum(vals) / count
    if count == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in vals) / (count - 1)
    return mean, math.sqrt(variance)


def calibration_bins(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> list[Bin]:
    """Summarize equal-width bins over [0,1] (left-closed, last bin includes 1.0)."""
    p, l = _validate(probs, labels, n_bins)
    counts = [0] * n_bins
    conf_sums = [0.0] * n_bins
    pos_sums = [0] * n_bins
    for prob, label in zip(p, l):
        idx = min(int(prob * n_bins), n_bins - 1)
        counts[idx] += 1
        conf_sums[idx] += prob
        pos_sums[idx] += label
    return [
        Bin(
            n=count,
            mean_conf=(conf_sums[i] / count) if count else 0.0,
            frac_positive=(pos_sums[i] / count) if count else 0.0,
            gap=(abs(conf_sums[i] / count - pos_sums[i] / count) if count else 0.0),
        )
        for i, count in enumerate(counts)
    ]


def ece(probs: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo et al. 2017).

    Partition [0,1] into `n_bins` equal-width bins (bin i covers [i/M, (i+1)/M),
    last bin also includes 1.0). Then:

        ECE = sum over bins m of  (|B_m| / n) * |mean_conf(B_m) - frac_positive(B_m)|

    where B_m are the predictions with P(true) in bin m, mean_conf is their mean
    probability, and frac_positive is the fraction whose gold label is 1.
    """
    p, l = _validate(probs, labels, n_bins)
    n = len(p)
    return sum(bin_.n / n * bin_.gap for bin_ in calibration_bins(p, l, n_bins))


def _gate_target(gate: dict[str, float] | None) -> dict[str, float]:
    """Resolve gate targets from overrides; raises on unknown keys."""
    target: dict[str, float] = {"min_accuracy": MIN_ACCURACY, "max_ece": MAX_ECE}
    if gate:
        unknown = set(gate) - set(target)
        if unknown:
            raise ValueError(f"unknown gate keys: {sorted(unknown)}; allowed: {sorted(target)}")
        target.update({key: float(value) for key, value in gate.items()})
    return target


def gate_verdict(
    accuracy: float, ece_value: float, gate: dict[str, float] | None = None
) -> GateSection:
    """PASS/FAIL a (macro accuracy, macro ECE) pair against the gate targets.

    Shared by `build_report` and the tuning CV aggregation so baseline, CV, and
    final verdicts are produced by identical code. `gate` overrides
    `min_accuracy`/`max_ece` (defaults: MIN_ACCURACY/MAX_ECE).
    """
    target = _gate_target(gate)
    failures: list[str] = []
    if accuracy < target["min_accuracy"]:
        failures.append(
            f"macro accuracy {accuracy:.4f} < min_accuracy {target['min_accuracy']:.4f}"
        )
    if ece_value > target["max_ece"]:
        failures.append(f"macro ece {ece_value:.4f} > max_ece {target['max_ece']:.4f}")
    return GateSection(target=target, passed=not failures, failures=failures)


def build_report(
    results: Sequence[CheckResult],
    gate: dict[str, float] | None = None,
    n_bins: int = 10,
) -> EvalReport:
    """Per-check metrics + ECE, macro means, and gate verdict against PRODUCT.md targets.

    `gate` optionally overrides `min_accuracy`/`max_ece` (defaults: MIN_ACCURACY/MAX_ECE).
    """
    if not results:
        raise ValueError("build_report needs at least one CheckResult")
    _gate_target(gate)  # validate gate keys before touching probabilities

    checks: list[CheckEval] = []
    for result in results:
        metrics = binary_metrics(result.probs, result.labels, result.threshold)
        bins = calibration_bins(result.probs, result.labels, n_bins)
        checks.append(
            CheckEval(
                check=result.check,
                threshold=result.threshold,
                metrics=metrics,
                ece=ece(result.probs, result.labels, n_bins),
                bins=bins,
            )
        )

    count = len(checks)
    macro = MacroAverages(
        accuracy=sum(c.metrics.accuracy for c in checks) / count,
        precision=sum(c.metrics.precision for c in checks) / count,
        recall=sum(c.metrics.recall for c in checks) / count,
        f1=sum(c.metrics.f1 for c in checks) / count,
        ece=sum(c.ece for c in checks) / count,
    )

    return EvalReport(
        checks=checks,
        macro=macro,
        gate=gate_verdict(macro.accuracy, macro.ece, gate),
        n=sum(c.metrics.n for c in checks),
    )
