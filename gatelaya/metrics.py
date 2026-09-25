"""Eval metrics: binary scores, calibration error, and PRODUCT.md gate reports."""

from __future__ import annotations

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
    target: dict[str, float] = {"min_accuracy": MIN_ACCURACY, "max_ece": MAX_ECE}
    if gate:
        unknown = set(gate) - set(target)
        if unknown:
            raise ValueError(f"unknown gate keys: {sorted(unknown)}; allowed: {sorted(target)}")
        target.update({key: float(value) for key, value in gate.items()})

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

    failures: list[str] = []
    if macro.accuracy < target["min_accuracy"]:
        failures.append(
            f"macro accuracy {macro.accuracy:.4f} < min_accuracy {target['min_accuracy']:.4f}"
        )
    if macro.ece > target["max_ece"]:
        failures.append(f"macro ece {macro.ece:.4f} > max_ece {target['max_ece']:.4f}")

    return EvalReport(
        checks=checks,
        macro=macro,
        gate=GateSection(target=target, passed=not failures, failures=failures),
        n=sum(c.metrics.n for c in checks),
    )
