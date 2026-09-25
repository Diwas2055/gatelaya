"""Temperature calibration: maps, temperature scaling, and NLL grid fit."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .errors import GuardrailConfigurationError


@dataclass
class TemperatureMap:
    """Per-(question_type, option_count) temperatures: {"choice": {...}, "noul": {...}}."""

    choice: dict[str, float] = field(default_factory=dict)
    noul: dict[str, float] = field(default_factory=lambda: {"default": 1.0})

    def temperature_for(self, question_type: str, option_count: int | None = None) -> float:
        """Return the temperature for a question type (choice keyed by option count)."""
        if question_type == "noul":
            return float(self.noul.get("default", 1.0))
        if question_type == "choice":
            if option_count is None:
                raise GuardrailConfigurationError("option_count required for choice temperature")
            return float(self.choice.get(str(option_count), 1.0))
        raise GuardrailConfigurationError(f"unknown question type: {question_type!r}")

    def noul_temperature(self, check: str | None = None) -> float:
        """Temperature for a noul answer: per-check entry if present, else `default`.

        Tuned calibration files store per-check temperatures as extra noul keys
        (e.g. ``{"default": 1.3, "pii": 1.6}``); runtime looks up by check name
        and falls back to `default`, so plain single-temperature files keep working.
        """
        if check is not None:
            key = str(check)
            if key in self.noul:
                return float(self.noul[key])
        return float(self.noul.get("default", 1.0))

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Return the JSON-serializable map."""
        return {"choice": dict(self.choice), "noul": dict(self.noul)}

    @classmethod
    def from_dict(cls, data: dict) -> TemperatureMap:
        """Build a map from its JSON dict form."""
        if not isinstance(data, dict):
            raise GuardrailConfigurationError("temperature map must be a mapping")
        choice = data.get("choice", {})
        noul = data.get("noul", {"default": 1.0})
        if not isinstance(choice, dict) or not isinstance(noul, dict):
            raise GuardrailConfigurationError("temperature map keys 'choice'/'noul' must be mappings")
        return cls(choice={str(k): float(v) for k, v in choice.items()},
                   noul={str(k): float(v) for k, v in noul.items()})

    @classmethod
    def load(cls, path: str | Path) -> TemperatureMap:
        """Load a temperature map from a JSON file."""
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GuardrailConfigurationError(f"calibration file not found: {p}") from exc
        except json.JSONDecodeError as exc:
            raise GuardrailConfigurationError(f"invalid calibration JSON in {p}: {exc}") from exc
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the temperature map to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_temperature_map(path: str | Path) -> TemperatureMap:
    """Load a temperature map from a JSON file."""
    return TemperatureMap.load(path)


def save_temperature_map(temperatures: TemperatureMap, path: str | Path) -> None:
    """Write a temperature map to a JSON file."""
    temperatures.save(path)


def calibrated(probs: Sequence[float], temperature: float) -> list[float]:
    """Temperature-scale a probability vector (log-space) and renormalize."""
    if temperature <= 0:
        raise GuardrailConfigurationError(f"temperature must be > 0, got {temperature}")
    if not probs:
        raise GuardrailConfigurationError("probs must be non-empty")
    logits = [math.log(max(float(p), 1e-12)) / temperature for p in probs]
    peak = max(logits)
    exps = [math.exp(logit - peak) for logit in logits]
    total = sum(exps)
    return [exp_val / total for exp_val in exps]


def fit(
    samples: Sequence[tuple[Sequence[float], int]],
    temperatures: Sequence[float] | None = None,
) -> float:
    """Grid-search the scalar temperature minimizing NLL over (probs, true_label) pairs."""
    if not samples:
        return 1.0
    grid = list(temperatures) if temperatures is not None else [i / 100.0 for i in range(5, 501)]
    best_temperature = 1.0
    best_nll = math.inf
    for temperature in grid:
        nll = 0.0
        for probs, label in samples:
            if not 0 <= label < len(probs):
                raise GuardrailConfigurationError(
                    f"label {label} out of range for probs of length {len(probs)}"
                )
            p_true = calibrated(probs, temperature)[label]
            nll -= math.log(max(p_true, 1e-12))
        if nll < best_nll:
            best_nll = nll
            best_temperature = temperature
    return best_temperature
