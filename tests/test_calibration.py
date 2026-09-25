"""Tests for temperature calibration: TemperatureMap I/O, scaling, and fit()."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gatelaya.calibration import (
    TemperatureMap,
    calibrated,
    fit,
    load_temperature_map,
    save_temperature_map,
)
from gatelaya.errors import GuardrailConfigurationError

# ------------------------------------------------------------- TemperatureMap


def test_temperature_map_defaults() -> None:
    """Arrange/act: fresh map. Assert: noul default 1.0, no choice buckets."""
    tm = TemperatureMap()
    assert tm.noul == {"default": 1.0}
    assert tm.choice == {}
    assert tm.temperature_for("noul") == 1.0


def test_temperature_for_bucket_lookup() -> None:
    """Arrange: buckets for noul and choice counts. Act: lookups.
    Assert: exact bucket hit, missing count falls back to 1.0."""
    tm = TemperatureMap(choice={"5": 1.2, "3": 0.7}, noul={"default": 0.8})
    assert tm.temperature_for("noul") == pytest.approx(0.8)
    assert tm.temperature_for("choice", 5) == pytest.approx(1.2)
    assert tm.temperature_for("choice", 3) == pytest.approx(0.7)
    assert tm.temperature_for("choice", 7) == pytest.approx(1.0)  # default


def test_temperature_for_rejects_bad_requests() -> None:
    """Arrange: map. Act: choice without option_count; unknown type.
    Assert: GuardrailConfigurationError each time."""
    tm = TemperatureMap()
    with pytest.raises(GuardrailConfigurationError, match="option_count"):
        tm.temperature_for("choice")
    with pytest.raises(GuardrailConfigurationError, match="unknown question type"):
        tm.temperature_for("score")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Arrange: populated map. Act: save -> load. Assert: identical."""
    original = TemperatureMap(choice={"5": 1.25, "20": 0.6}, noul={"default": 1.75})
    path = tmp_path / "calibration" / "temps.json"
    # Act
    save_temperature_map(original, path)
    loaded = TemperatureMap.load(path)
    # Assert
    assert loaded.to_dict() == original.to_dict()
    assert loaded.temperature_for("noul") == pytest.approx(1.75)
    assert loaded.temperature_for("choice", 5) == pytest.approx(1.25)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Arrange/act: missing path. Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="not found"):
        TemperatureMap.load(tmp_path / "nope.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    """Arrange: malformed JSON. Act: load. Assert: config error."""
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GuardrailConfigurationError, match="invalid calibration JSON"):
        TemperatureMap.load(path)


def test_from_dict_rejects_non_mapping_and_bad_shapes() -> None:
    """Arrange/act: bad dict forms. Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="must be a mapping"):
        TemperatureMap.from_dict([1, 2])  # type: ignore[arg-type]
    with pytest.raises(GuardrailConfigurationError, match="must be mappings"):
        TemperatureMap.from_dict({"choice": [], "noul": {}})  # type: ignore[dict-item]


def test_from_dict_coerces_keys_and_values() -> None:
    """Arrange: JSON-parsed dict with int keys/values. Act: from_dict.
    Assert: keys stringified, values floats."""
    tm = TemperatureMap.from_dict({"choice": {5: 2}, "noul": {"default": 1}})
    assert tm.choice == {"5": 2.0}
    assert tm.noul == {"default": 1.0}


def test_save_writes_sorted_json(tmp_path: Path) -> None:
    """Arrange: map. Act: save. Assert: valid JSON on disk with both sections."""
    path = tmp_path / "t.json"
    TemperatureMap(choice={"2": 1.0}, noul={"default": 1.0}).save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"choice", "noul"}


# ---------------------------------------------------------------- calibrated


def test_calibrated_renormalizes_to_probability_vector() -> None:
    """Arrange: arbitrary probs. Act: temperature scale.
    Assert: sums to ~1, values in [0,1], ordering preserved."""
    probs = [0.7, 0.2, 0.1]
    out = calibrated(probs, 2.5)
    assert sum(out) == pytest.approx(1.0)
    assert all(0.0 <= p <= 1.0 for p in out)
    assert out[0] > out[1] > out[2]


def test_calibrated_temperature_one_is_identity() -> None:
    """Arrange: probs. Act: T=1 scaling. Assert: unchanged (within float error)."""
    probs = [0.9, 0.1]
    out = calibrated(probs, 1.0)
    assert out == pytest.approx(probs)


def test_calibrated_higher_temperature_flattens() -> None:
    """Arrange: confident binary probs. Act: raise T.
    Assert: distribution moves toward uniform."""
    low = calibrated([0.99, 0.01], 1.0)
    high = calibrated([0.99, 0.01], 5.0)
    assert high[1] > low[1]
    assert high[1] < 0.5  # pulled toward 0.5


def test_calibrated_handles_zero_probability() -> None:
    """Arrange: a zero entry (would be log(0)). Act: scale.
    Assert: finite result summing to 1."""
    out = calibrated([0.0, 1.0], 2.0)
    assert sum(out) == pytest.approx(1.0)
    assert all(p == p and p not in (float("inf"),) for p in out)


def test_calibrated_rejects_bad_temperature_and_empty_probs() -> None:
    """Arrange/act: T<=0 and empty probs. Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="temperature"):
        calibrated([0.5, 0.5], 0.0)
    with pytest.raises(GuardrailConfigurationError, match="non-empty"):
        calibrated([], 1.0)


# ----------------------------------------------------------------------- fit


def test_fit_on_well_separated_calibrated_data_recovers_unit_temperature() -> None:
    """Arrange: well-separated binary probs whose empirical label frequencies match
    the predicted probabilities (90/10 split at p=[0.9,0.1]) — i.e. perfectly
    calibrated data. Act: fit(). Assert: recovered temperature ≈ 1.0."""
    samples = [([0.9, 0.1], 0)] * 9 + [([0.9, 0.1], 1)] * 1
    # Act
    temperature = fit(samples)
    # Assert
    assert temperature == pytest.approx(1.0, abs=0.05)


def test_fit_on_overconfident_wrong_labels_yields_higher_temperature() -> None:
    """Arrange: overconfident p=0.99 predictions whose true label is the low-prob
    class. Act: fit(). Assert: T > 1 (model must be softened)."""
    samples = [([0.99, 0.01], 1)] * 10
    # Act
    temperature = fit(samples)
    # Assert
    assert temperature > 1.0


def test_fit_returns_unit_temperature_for_empty_samples() -> None:
    """Arrange/act: no samples. Assert: 1.0."""
    assert fit([]) == 1.0


def test_fit_respects_custom_grid() -> None:
    """Arrange: single-sample grid. Act: fit with grid.
    Assert: member of that grid is returned."""
    samples = [([0.6, 0.4], 0)]
    assert fit(samples, temperatures=[0.5]) == 0.5
    assert fit(samples, temperatures=[1.5, 2.5]) in (1.5, 2.5)


def test_fit_default_grid_stays_within_bounds() -> None:
    """Arrange: extreme samples. Act: fit with default grid.
    Assert: result within [0.05, 5.0] (documented grid range)."""
    temperature = fit([([0.99, 0.01], 1)] * 5)
    assert 0.05 <= temperature <= 5.0


def test_fit_rejects_out_of_range_label() -> None:
    """Arrange: label index beyond probs length. Act: fit.
    Assert: GuardrailConfigurationError."""
    with pytest.raises(GuardrailConfigurationError, match="out of range"):
        fit([([0.5, 0.5], 5)])
