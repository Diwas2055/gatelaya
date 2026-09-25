"""Unit tests for gatelaya.metrics: binary scores, ECE, bins, sweeps, AUC, gate reports."""

from __future__ import annotations

import math

import pytest

from gatelaya.metrics import (
    MAX_ECE,
    MIN_ACCURACY,
    CheckResult,
    EvalReport,
    binary_metrics,
    build_report,
    calibration_bins,
    ece,
    gate_verdict,
    mean_std,
    roc_auc,
    sweep_thresholds,
    threshold_grid,
)

# ------------------------------------------------------------- binary_metrics


def test_binary_metrics_hand_computed() -> None:
    """Confusion counts, accuracy, precision, recall, F1 match hand calculation."""
    probs = [0.9, 0.2, 0.6, 0.4, 0.95]
    labels = [1, 0, 1, 1, 0]
    m = binary_metrics(probs, labels, threshold=0.5)
    # preds = [1, 0, 1, 0, 1] -> tp=2 (idx0,2), tn=1 (idx1), fp=1 (idx4), fn=1 (idx3)
    assert (m.tp, m.tn, m.fp, m.fn) == (2, 1, 1, 1)
    assert m.n == 5
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_binary_metrics_threshold_boundary_inclusive() -> None:
    """prob == threshold counts as positive (matches guardrail `cal_p < threshold` rule)."""
    m = binary_metrics([0.5], [1], threshold=0.5)
    assert (m.tp, m.fn) == (1, 0)
    m = binary_metrics([0.5], [0], threshold=0.5)
    assert (m.fp, m.tn) == (1, 0)


def test_binary_metrics_no_predicted_positives() -> None:
    """Undefined precision/recall degrade to 0.0 instead of raising."""
    m = binary_metrics([0.1, 0.2], [1, 0], threshold=0.9)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.accuracy == 0.5


@pytest.mark.parametrize(
    ("probs", "labels", "threshold", "match"),
    [
        ([], [], 0.5, "non-empty"),
        ([0.5, 0.6], [1], 0.5, "length"),
        ([0.5], [2], 0.5, "0 or 1"),
        ([1.5], [1], 0.5, "out of"),
        ([0.5], [1], 1.5, "threshold"),
    ],
)
def test_binary_metrics_invalid_inputs(probs, labels, threshold, match) -> None:
    """Empty, mismatched, or out-of-range inputs raise ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        binary_metrics(probs, labels, threshold)


# ----------------------------------------------------------------------- ece


def test_ece_known_value_fixture() -> None:
    """Hand-computed ECE: 0.25 for the ten-row spread fixture with n_bins=10."""
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # gaps: 0.1+0.2+0.3+0.4+0.5 (weight 1/10 each), 0.4+0.3+0.2 (1/10 each),
    # bin [0.9,1.0] holds 0.9 and 1.0 -> mean 0.95, frac 1.0, gap 0.05 (weight 2/10)
    assert ece(probs, labels, n_bins=10) == pytest.approx(0.1 * 2.4 + 0.2 * 0.05)


def test_ece_perfect_calibration_is_zero() -> None:
    """Well-calibrated predictions (0.5 with 50% positives) give ECE ~ 0."""
    assert ece([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1], n_bins=10) == pytest.approx(0.0)


def test_ece_perfectly_miscalibrated_is_high() -> None:
    """Confident 0.9 predictions that are always wrong give ECE = 0.9."""
    assert ece([0.9] * 4, [0] * 4, n_bins=10) == pytest.approx(0.9)


def test_ece_last_bin_boundary_one() -> None:
    """prob = 1.0 lands in the final bin (inclusive right edge), no index overflow."""
    assert ece([1.0], [1], n_bins=10) == pytest.approx(0.0)
    assert ece([1.0], [0], n_bins=10) == pytest.approx(1.0)


def test_ece_empty_raises() -> None:
    """Empty input is an error, not a silent 0."""
    with pytest.raises(ValueError, match="non-empty"):
        ece([], [])


def test_ece_invalid_n_bins() -> None:
    """n_bins < 1 raises."""
    with pytest.raises(ValueError, match="n_bins"):
        ece([0.5], [1], n_bins=0)


# --------------------------------------------------------- calibration_bins


def test_calibration_bins_shape_and_sums() -> None:
    """Always returns n_bins entries; counts sum to the number of rows."""
    probs = [0.0, 0.05, 0.1, 0.49, 0.5, 0.99, 1.0]
    labels = [0, 0, 1, 0, 1, 1, 1]
    bins = calibration_bins(probs, labels, n_bins=10)
    assert len(bins) == 10
    assert sum(b.n for b in bins) == len(probs)
    # boundary rows: 0.0/0.05 -> bin0, 0.1 -> bin1, 0.49 -> bin4, 0.5 -> bin5, 0.99/1.0 -> bin9
    assert [b.n for b in bins] == [2, 1, 0, 0, 1, 1, 0, 0, 0, 2]
    assert bins[0].mean_conf == pytest.approx(0.025)
    assert bins[0].frac_positive == 0.0
    assert bins[0].gap == pytest.approx(0.025)
    # empty bins report zeros, not NaN
    assert bins[2].n == 0 and bins[2].gap == 0.0


def test_calibration_bins_gap_is_absolute() -> None:
    """gap = |mean_conf - frac_positive| even when the model under-confident."""
    bins = calibration_bins([0.2, 0.2], [1, 1], n_bins=10)
    assert bins[2].gap == pytest.approx(0.8)


# ------------------------------------------------------------- build_report


def _result(name: str, probs: list[float], labels: list[int], threshold: float = 0.5) -> CheckResult:
    return CheckResult(check=name, probs=probs, labels=labels, threshold=threshold)


def test_build_report_macro_averages_and_gate_pass() -> None:
    """Macro averages are means over checks; a passing report marks gate.passed."""
    results = [
        _result("pii", [0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]),          # acc 1.0, ece 0.15
        _result("injection", [0.6, 0.4, 0.7, 0.3], [1, 0, 1, 0]),     # acc 1.0, ece 0.35
    ]
    report = build_report(results)
    assert isinstance(report, EvalReport)
    assert report.n == 8
    assert len(report.checks) == 2
    assert report.checks[0].check == "pii"
    assert report.checks[0].metrics.accuracy == 1.0
    # macro = unweighted mean over the two checks
    assert report.macro.accuracy == pytest.approx(1.0)
    assert report.macro.ece == pytest.approx((0.15 + 0.35) / 2)
    assert report.macro.f1 == pytest.approx(1.0)
    # macro ece 0.25 > MAX_ECE 0.15 -> gate fails with an ece failure line
    assert report.gate.passed is False
    assert report.gate.failures
    assert "macro ece" in report.gate.failures[0]


def test_build_report_gate_pass_with_defaults() -> None:
    """Both targets met -> passed=True, no failures, PRODUCT.md targets recorded."""
    results = [_result("pii", [0.95, 0.05, 0.9, 0.1], [1, 0, 1, 0])]
    report = build_report(results)
    assert report.macro.accuracy == pytest.approx(1.0)
    assert report.macro.ece <= 0.15
    assert report.gate.passed is True
    assert report.gate.failures == []
    assert report.gate.target == {"min_accuracy": MIN_ACCURACY, "max_ece": MAX_ECE}


def test_build_report_gate_accuracy_failure() -> None:
    """Accuracy below target is reported as a failure string."""
    results = [_result("pii", [0.9, 0.9, 0.2, 0.2, 0.2, 0.2], [1, 0, 0, 0, 0, 0])]
    report = build_report(results)
    assert report.macro.accuracy == pytest.approx(5 / 6)
    assert report.gate.passed is False
    assert any("accuracy" in failure for failure in report.gate.failures)


def test_build_report_gate_target_override() -> None:
    """A stricter custom gate fails even when defaults would pass."""
    # acc 0.9 (= MIN_ACCURACY, passes), ece 0.09 (passes) -> override makes it fail
    probs = [0.95, 0.05, 0.9, 0.1, 0.95, 0.05, 0.9, 0.1, 0.95, 0.95]
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    results = [_result("pii", probs, labels)]
    assert build_report(results).gate.passed is True
    report = build_report(results, gate={"min_accuracy": 0.95})
    assert report.gate.passed is False
    assert report.gate.target["min_accuracy"] == 0.95


def test_build_report_rejects_unknown_gate_keys_and_empty_results() -> None:
    """Unknown gate keys and empty result lists are errors."""
    with pytest.raises(ValueError, match="unknown gate keys"):
        build_report([_result("pii", [0.9], [1], 0.5)], gate={"min_f1": 0.9})
    with pytest.raises(ValueError, match="at least one"):
        build_report([])


def test_product_md_gate_constants() -> None:
    """MIN_ACCURACY/MAX_ECE match PRODUCT.md success criteria (0.90 / 0.15)."""
    assert MIN_ACCURACY == 0.90
    assert MAX_ECE == 0.15
    assert math.isclose(MIN_ACCURACY, 0.90) and math.isclose(MAX_ECE, 0.15)


def test_report_json_round_trip() -> None:
    """EvalReport serializes to JSON-compatible dicts (CLI writes it to disk)."""
    report = build_report([_result("pii", [0.9, 0.1], [1, 0])])
    data = report.model_dump(mode="json")
    assert data["gate"]["passed"] in (True, False)
    assert set(data["gate"]["target"]) == {"min_accuracy", "max_ece"}
    assert EvalReport.model_validate(data).n == 2


# ------------------------------------------------------------ threshold_grid


def test_threshold_grid_default_endpoints() -> None:
    """Default grid is 0.05..0.95 step 0.05: 19 interior points, both ends included."""
    grid = threshold_grid()
    assert len(grid) == 19
    assert grid[0] == 0.05 and grid[-1] == 0.95
    assert all(0.0 < t < 1.0 for t in grid)
    assert threshold_grid(0.1) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@pytest.mark.parametrize("step", [0.0, -0.1, 1.5])
def test_threshold_grid_rejects_invalid_step(step: float) -> None:
    """Steps outside (0, 1] are errors, not silent weirdness."""
    with pytest.raises(ValueError, match="step"):
        threshold_grid(step)


def test_threshold_grid_rejects_too_coarse_step() -> None:
    """A step leaving no interior threshold (e.g. 1.0) is an error."""
    with pytest.raises(ValueError, match="coarse"):
        threshold_grid(1.0)


# --------------------------------------------------------- sweep_thresholds


def test_sweep_thresholds_known_best_fixture() -> None:
    """Separable data: every perfect threshold ties, closest to 0.5 wins (0.50)."""
    probs = [0.9, 0.1, 0.9, 0.1]
    labels = [1, 0, 1, 0]
    sweep = sweep_thresholds(probs, labels, step=0.05)
    assert [p.threshold for p in sweep.points] == threshold_grid()
    assert all(p.metrics.accuracy == 1.0 for p in sweep.points if 0.15 <= p.threshold <= 0.90)
    assert sweep.best_threshold == 0.50
    assert sweep.best.accuracy == 1.0
    assert sweep.best.f1 == 1.0


def test_sweep_thresholds_f1_tie_break_precedes_distance() -> None:
    """Equal max accuracy: higher F1 wins even when another grid point is closer to 0.5.

    Hand-computed fixture (4 pos / 4 neg):
      acc 0.75 at t=0.15..0.45 (F1 0.75) and t=0.50, 0.55 (F1 0.667).
      F1 rule keeps {0.15..0.45} → closest-to-0.5 picks 0.45, NOT 0.50.
    """
    probs = [0.60, 0.55, 0.45, 0.10, 0.48, 0.12, 0.10, 0.05]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    sweep = sweep_thresholds(probs, labels, step=0.05)
    assert sweep.best.accuracy == pytest.approx(0.75)
    assert sweep.best.f1 == pytest.approx(0.75)
    assert sweep.best_threshold == pytest.approx(0.45)


def test_sweep_thresholds_invalid_inputs() -> None:
    """sweep reuses binary_metrics validation (bad threshold impossible, bad data raises)."""
    with pytest.raises(ValueError, match="non-empty"):
        sweep_thresholds([], [])
    with pytest.raises(ValueError, match="length"):
        sweep_thresholds([0.5], [1, 0])


# ----------------------------------------------------------------- roc_auc


def test_roc_auc_perfect_and_inverted() -> None:
    """Perfect separation → 1.0; fully inverted scores → 0.0."""
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_roc_auc_hand_computed_with_ties() -> None:
    """Ties get average ranks: probs [0.5, 0.5, 0.9, 0.1], labels [0, 1, 1, 0] → 0.875.

    Sorted: 0.1 (rank 1, label 0), 0.5/0.5 (ranks 2-3 → avg 2.5), 0.9 (rank 4).
    Rank sum of positives = 2.5 + 4 = 6.5; U = 6.5 - 2·3/2 = 3.5; AUC = 3.5 / (2·2).
    """
    assert roc_auc([0.5, 0.5, 0.9, 0.1], [0, 1, 1, 0]) == pytest.approx(0.875)


def test_roc_auc_random_scoring_near_half() -> None:
    """A tiny hand-checkable case lands exactly at chance: [0.4, 0.6] labels [0, 1]."""
    assert roc_auc([0.4, 0.6], [0, 1]) == pytest.approx(1.0)
    assert roc_auc([0.6, 0.4], [0, 1]) == pytest.approx(0.0)


def test_roc_auc_single_class_raises() -> None:
    """AUC is undefined with one class — error, not a fake 0.0/1.0."""
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0.1, 0.2], [0, 0])


def test_roc_auc_invalid_inputs() -> None:
    """Same shared validation as other metrics."""
    with pytest.raises(ValueError, match="length"):
        roc_auc([0.5], [1, 0])
    with pytest.raises(ValueError, match="out of"):
        roc_auc([1.5], [1])


# ---------------------------------------------------------------- mean_std


def test_mean_std_known_values() -> None:
    """Classic fixture [2,4,4,4,5,5,7,9]: mean 5, sample std sqrt(32/7)."""
    mean, std = mean_std([2, 4, 4, 4, 5, 5, 7, 9])
    assert mean == pytest.approx(5.0)
    assert std == pytest.approx(math.sqrt(32 / 7))


def test_mean_std_single_and_empty() -> None:
    """A single value has std 0.0; empty input is an error."""
    assert mean_std([3.5]) == (3.5, 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        mean_std([])


# -------------------------------------------------------------- gate_verdict


def test_gate_verdict_matches_build_report_verdict() -> None:
    """gate_verdict produces the same verdict/failure strings as build_report."""
    report = build_report([_result("pii", [0.9, 0.9, 0.2, 0.2, 0.2, 0.2], [1, 0, 0, 0, 0, 0])])
    verdict = gate_verdict(report.macro.accuracy, report.macro.ece)
    assert verdict.passed == report.gate.passed
    assert verdict.failures == report.gate.failures
    assert verdict.target == report.gate.target
    # passing pair
    assert gate_verdict(0.9, 0.15).passed is True
    assert gate_verdict(0.8999, 0.15).passed is False
    assert gate_verdict(0.9, 0.1501).passed is False
    with pytest.raises(ValueError, match="unknown gate keys"):
        gate_verdict(1.0, 0.0, gate={"min_recall": 0.5})
