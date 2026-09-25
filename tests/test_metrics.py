"""Unit tests for gatelaya.metrics: binary scores, ECE, bins, and gate reports."""

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
