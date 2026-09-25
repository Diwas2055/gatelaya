"""Tests for gatelaya.tune: folds, leakage guard, CV determinism, report + CLI."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import gatelaya.tune as tune_mod
from gatelaya.calibration import TemperatureMap
from gatelaya.config import GateLayaConfig
from gatelaya.metrics import sweep_thresholds
from gatelaya.tune import CheckData, cv_check, make_folds, run_tune

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval.py"


def _load_eval_module() -> Any:
    """Import scripts/eval.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gatelaya_eval_script_tune", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_eval(*args: str) -> subprocess.CompletedProcess[str]:
    """Run scripts/eval.py as a subprocess from the project root."""
    return subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=180,
    )


def _separable_check(name: str, n: int = 40) -> CheckData:
    """Perfectly separable synthetic check: positives get high P(true)."""
    labels = [i % 2 for i in range(n)]
    probs = [[0.08, 0.92] if label else [0.88, 0.12] for label in labels]
    return CheckData(check=name, probs=probs, labels=labels)


def _mixed_check(name: str, n: int = 40) -> CheckData:
    """Partially overlapping scores so thresholds actually matter (deterministic)."""
    labels = [i % 2 for i in range(n)]
    probs: list[list[float]] = []
    for i in range(n):
        p_true = round(0.05 + 0.9 * (((i * 7) % 10) / 10), 4)
        probs.append([round(1 - p_true, 4), p_true])
    return CheckData(check=name, probs=probs, labels=labels)


# ------------------------------------------------------------------ folds


def test_make_folds_is_deterministic_and_partitioned() -> None:
    """Same seed/key → identical folds; folds are disjoint and cover every row."""
    folds = make_folds(40, folds=5, seed=42, key="pii")
    assert folds == make_folds(40, folds=5, seed=42, key="pii")
    flat = [i for fold in folds for i in fold]
    assert sorted(flat) == list(range(40))
    assert len(folds) == 5
    assert all(len(fold) == 8 for fold in folds)


def test_make_folds_seed_and_key_change_the_split() -> None:
    """Different seed or check key shuffles differently (but stays a partition)."""
    base = make_folds(40, folds=5, seed=42, key="pii")
    assert make_folds(40, folds=5, seed=43, key="pii") != base
    assert make_folds(40, folds=5, seed=42, key="toxicity") != base
    other = make_folds(40, folds=5, seed=43, key="pii")
    assert sorted(i for fold in other for i in fold) == list(range(40))


@pytest.mark.parametrize(
    ("n", "folds", "match"),
    [(10, 1, "folds must be"), (4, 5, "need at least")],
)
def test_make_folds_invalid_inputs(n: int, folds: int, match: str) -> None:
    """folds < 2 or more folds than rows is an error."""
    with pytest.raises(ValueError, match=match):
        make_folds(n, folds=folds, seed=42)


# ------------------------------------------------- CV determinism + leakage


def test_cv_check_same_seed_is_identical() -> None:
    """Same seed → byte-identical CV metrics (deterministic folds + fits)."""
    data = _mixed_check("injection")
    first = cv_check(data, folds=5, seed=42)
    second = cv_check(data, folds=5, seed=42)
    assert first == second
    assert len(first.chosen_thresholds) == 5
    assert len(first.temperatures) == 5
    assert first.n == data.n


def test_cv_fit_and_sweep_see_only_train_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """No-leakage guard: temperature fit and threshold sweep receive exactly the
    train fold (complement of the val fold), never the val rows themselves.

    Structural test: `fit` and `calibrated` are stubbed so every call's inputs
    can be compared against the fold split; `sweep_thresholds` records the
    calibrated probabilities it is asked to optimize.
    """
    n = 20
    # unique, strictly increasing P(false) → every row identifiable by its vector
    vectors = []
    for i in range(n):
        p_false = round(0.05 + 0.9 * (i / (n - 1)), 4)
        vectors.append([p_false, round(1 - p_false, 4)])
    labels = [i % 2 for i in range(n)]
    data = CheckData(check="pii", probs=vectors, labels=labels)

    recorded_fits: list[list[tuple[list[float], int]]] = []
    recorded_sweeps: list[tuple[list[float], list[int]]] = []

    def fake_fit(samples: Any, temperatures: Any = None) -> float:
        recorded_fits.append(list(samples))
        return 1.0

    def identity_calibrated(probs: Any, temperature: float) -> list[float]:
        return [float(p) for p in probs]

    def spy_sweep(probs: Any, labels_: Any, step: float = 0.05) -> Any:
        recorded_sweeps.append((list(probs), list(labels_)))
        return sweep_thresholds(probs, labels_, step)

    monkeypatch.setattr(tune_mod, "fit", fake_fit)
    monkeypatch.setattr(tune_mod, "calibrated", identity_calibrated)
    monkeypatch.setattr(tune_mod, "sweep_thresholds", spy_sweep)

    cv_check(data, folds=5, seed=7, step=0.05)

    fold_ids = make_folds(n, folds=5, seed=7, key="pii")
    all_rows = set(range(n))
    assert len(recorded_fits) == 5 and len(recorded_sweeps) == 5
    for fold_id, val_rows in enumerate(fold_ids):
        train_rows = sorted(all_rows - set(val_rows))
        expected_vectors = [vectors[i] for i in train_rows]
        expected_labels = [labels[i] for i in train_rows]
        # temperature fit: train rows only
        fit_vectors = [sample[0] for sample in recorded_fits[fold_id]]
        assert fit_vectors == expected_vectors
        assert [sample[1] for sample in recorded_fits[fold_id]] == expected_labels
        # threshold sweep: calibrated train P(true), train labels only
        sweep_probs, sweep_labels = recorded_sweeps[fold_id]
        assert sweep_probs == [vectors[i][1] for i in train_rows]
        assert sweep_labels == expected_labels
        # no val row leaks into either artifact selection
        assert set(sweep_labels) <= {labels[i] for i in train_rows}


# ------------------------------------------------------------- run_tune


def test_run_tune_report_shape_and_artifacts(tmp_path: Path) -> None:
    """run_tune emits the documented JSON shape; config/calibration are loadable."""
    datas = [_separable_check("pii"), _separable_check("injection")]
    result = run_tune(datas, folds=5, step=0.05, seed=42, meta={"agent": "test"})

    payload = result.model_dump(mode="json")
    assert set(payload) == {
        "meta",
        "baseline",
        "cv",
        "final",
        "deltas",
        "roc_auc",
        "honest_assessment",
    }
    assert set(payload["cv"]) == {"checks", "macro", "gate"}
    assert set(payload["final"]) == {"checks", "macro", "gate", "pooled_temperature", "label"}
    assert set(payload["deltas"]) == {"pii", "injection"}
    assert payload["meta"]["folds"] == 5 and payload["meta"]["seed"] == 42
    assert isinstance(payload["honest_assessment"], str) and payload["honest_assessment"]

    for check, cv in payload["cv"]["checks"].items():
        assert set(cv["accuracy"]) == {"mean", "std"}
        assert set(cv["ece"]) == {"mean", "std"}
        assert len(cv["chosen_thresholds"]) == 5
        assert cv["n"] == 40 and cv["folds"] == 5

    # separable data → ROC AUC 1.0, CV gate should pass
    assert payload["roc_auc"]["pii"] == pytest.approx(1.0)
    assert payload["cv"]["gate"]["passed"] is True
    assert payload["final"]["gate"]["passed"] is True

    # thresholds are deployable (strictly inside (0, 1))
    thresholds = result.tuned_thresholds()
    assert set(thresholds) == {"pii", "injection"}
    assert all(0.0 < t < 1.0 for t in thresholds.values())

    # calibration round-trips through the TemperatureMap file format
    calibration_path = tmp_path / "calibration.json"
    result.temperature_map().save(calibration_path)
    loaded = TemperatureMap.load(calibration_path)
    assert loaded.noul["default"] == pytest.approx(result.final.pooled_temperature)
    for check in thresholds:
        assert loaded.noul[check] == pytest.approx(result.final.checks[check].temperature)
    assert loaded.noul_temperature("pii") == pytest.approx(result.final.checks["pii"].temperature)
    # unknown check falls back to default (backward compatible)
    assert loaded.noul_temperature("unknown") == pytest.approx(result.final.pooled_temperature)

    # config YAML parses as a GateLayaConfig
    config_path = tmp_path / "tuned-config.yaml"
    GateLayaConfig(thresholds=thresholds, calibration_path=calibration_path).to_yaml(config_path)
    reloaded = GateLayaConfig.from_yaml(config_path)
    assert reloaded.thresholds == pytest.approx({**GateLayaConfig().thresholds, **thresholds})


def test_run_tune_is_deterministic_for_same_seed() -> None:
    """Two runs on the same inputs/seed produce identical CV and final sections."""
    datas = [_mixed_check("toxicity"), _mixed_check("pii")]
    first = run_tune(datas, folds=4, seed=11)
    second = run_tune(datas, folds=4, seed=11)
    assert first.cv == second.cv
    assert first.final.checks == second.final.checks
    assert first.roc_auc == second.roc_auc
    assert first.deltas == second.deltas
    assert first.honest_assessment == second.honest_assessment


def test_run_tune_rejects_empty_input() -> None:
    """No checks → error, not a divide-by-zero."""
    with pytest.raises(ValueError, match="at least one"):
        run_tune([])


# ------------------------------------------------------------- CLI plumbing


def test_eval_cli_tune_fake_end_to_end(tmp_path: Path) -> None:
    """--tune --agent fake --limit 20: exit 0, report keys, deployable YAML."""
    report = tmp_path / "tuned.json"
    config = tmp_path / "tuned-config.yaml"
    calibration = tmp_path / "tuned-calibration.json"
    proc = _run_eval(
        "--tune", "--agent", "fake", "--limit", "20",
        "--tune-report", str(report),
        "--tune-config", str(config),
        "--tune-calibration", str(calibration),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SANITY MODE" in proc.stderr
    assert "GATE baseline:" in proc.stdout
    assert "honest" in proc.stdout

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert set(payload) == {
        "meta",
        "baseline",
        "cv",
        "final",
        "deltas",
        "roc_auc",
        "honest_assessment",
    }
    assert payload["meta"]["sanity_mode"] is True
    assert set(payload["baseline"]) == {"checks", "macro", "gate", "n"}
    assert payload["baseline"]["n"] == 80  # 20 rows x 4 checks
    assert set(payload["cv"]["checks"]) == {"pii", "injection", "toxicity", "secret_leak"}
    assert payload["cv"]["gate"]["target"] == {"min_accuracy": 0.90, "max_ece": 0.15}

    reloaded = GateLayaConfig.from_yaml(config)
    assert set(reloaded.thresholds) == {"pii", "injection", "toxicity", "secret_leak"}
    assert all(0.0 < t < 1.0 for t in reloaded.thresholds.values())
    assert reloaded.calibration_path is not None

    temps = TemperatureMap.load(calibration)
    assert temps.noul["default"] > 0
    for check in ("pii", "injection", "toxicity", "secret_leak"):
        assert check in temps.noul
        assert temps.noul_temperature(check) == pytest.approx(temps.noul[check])


def test_eval_cli_tune_gate_uses_cv_metrics(tmp_path: Path) -> None:
    """With --gate --tune, the exit code follows the honest CV verdict."""
    proc = _run_eval(
        "--tune", "--gate", "--agent", "fake", "--limit", "20",
        "--tune-report", str(tmp_path / "t.json"),
        "--tune-config", str(tmp_path / "c.yaml"),
        "--tune-calibration", str(tmp_path / "k.json"),
    )
    # fake agent is perfect → CV gate passes → exit 0
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CV (honest): PASS" in proc.stdout


def test_eval_cli_tune_rejects_bad_folds(tmp_path: Path) -> None:
    """--folds 1 is rejected before any model work."""
    proc = _run_eval("--tune", "--agent", "fake", "--folds", "1")
    assert proc.returncode != 0
    assert "folds must be >= 2" in proc.stderr
