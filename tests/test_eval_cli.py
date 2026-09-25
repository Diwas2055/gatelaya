"""Tests for scripts/eval.py plumbing and evals/data dataset integrity."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval.py"
DATA_DIR = PROJECT_ROOT / "evals" / "data"
DATASET_FILES = ("pii.jsonl", "injection.jsonl", "toxicity.jsonl", "secret_leak.jsonl")
REQUIRED_FIELDS = {"text", "label", "check", "lang", "id"}
ALLOWED_LANGS = {"en", "np", "es", "fr", "de", "hi", "ar"}


def _load_eval_module() -> Any:
    """Import scripts/eval.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("gatelaya_eval_script", EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_eval(*args: str) -> subprocess.CompletedProcess[str]:
    """Run scripts/eval.py as a subprocess from the project root."""
    return subprocess.run(
        [sys.executable, str(EVAL_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=120,
    )


# ------------------------------------------------------------------ CLI plumbing


def test_eval_cli_fake_agent_smoke(tmp_path: Path) -> None:
    """--agent fake --limit 8 --report writes a valid EvalReport and passes the gate."""
    report_path = tmp_path / "report.json"
    proc = _run_eval(
        "--agent", "fake", "--data", "evals/data", "--limit", "8",
        "--report", str(report_path), "--gate",
    )
    assert proc.returncode == 0, proc.stderr
    assert "SANITY MODE" in proc.stderr
    assert "GATE PASS" in proc.stdout

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report) == {"checks", "macro", "gate", "n"}
    assert report["n"] == 32
    assert len(report["checks"]) == 4
    for check in report["checks"]:
        assert check["metrics"]["n"] == 8
        assert check["metrics"]["accuracy"] == pytest.approx(1.0)
        assert 0.0 <= check["ece"] <= 1.0
        assert len(check["bins"]) == 10
        assert sum(b["n"] for b in check["bins"]) == 8
    assert set(report["gate"]) == {"target", "passed", "failures"}
    assert report["gate"]["target"] == {"min_accuracy": 0.90, "max_ece": 0.15}
    assert report["gate"]["passed"] is True
    assert report["gate"]["failures"] == []
    assert set(report["macro"]) == {"accuracy", "precision", "recall", "f1", "ece"}


def test_eval_cli_gate_failure_exits_1(tmp_path: Path) -> None:
    """--gate exits 1 when a stricter threshold pushes accuracy below the target."""
    config_path = tmp_path / "strict.yaml"
    config_path.write_text("thresholds:\n  pii: 0.95\n", encoding="utf-8")
    proc = _run_eval(
        "--agent", "fake", "--data", "evals/data", "--check", "pii", "--limit", "8",
        "--config", str(config_path), "--gate",
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "GATE FAIL" in proc.stdout


def test_eval_cli_lang_and_filters(tmp_path: Path) -> None:
    """--lang + --check filter rows; --limit caps rows per check."""
    report_path = tmp_path / "report.json"
    proc = _run_eval(
        "--agent", "fake", "--data", "evals/data", "--check", "injection",
        "--lang", "np", "--limit", "4", "--report", str(report_path),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [c["check"] for c in report["checks"]] == ["injection"]
    assert report["checks"][0]["metrics"]["n"] == 4


def test_eval_cli_missing_data_exits_nonzero(tmp_path: Path) -> None:
    """A nonexistent --data path is a hard error."""
    proc = _run_eval("--agent", "fake", "--data", str(tmp_path / "nope"))
    assert proc.returncode != 0
    assert "not found" in proc.stderr


def test_load_rows_filters_and_validation(tmp_path: Path) -> None:
    """load_rows groups by check, honors filters/limit, and rejects bad rows."""
    eval_module = _load_eval_module()
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "x", "label": 7, "check": "pii", "lang": "en"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="label must be 0 or 1"):
        eval_module.load_rows(bad, None, None, None)

    ok = tmp_path / "ok.jsonl"
    rows = [
        {"text": "a", "label": 1, "check": "pii", "lang": "en", "id": "pii-001"},
        {"text": "b", "label": 0, "check": "pii", "lang": "np", "id": "pii-002"},
        {"text": "c", "label": 1, "check": "toxicity", "lang": "en", "id": "toxicity-001"},
    ]
    ok.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    grouped = eval_module.load_rows(ok, ["pii"], None, None)
    assert list(grouped) == ["pii"]
    assert len(grouped["pii"]) == 2
    grouped = eval_module.load_rows(ok, None, "np", 1)
    assert {c: len(r) for c, r in grouped.items()} == {"pii": 1}


# ------------------------------------------------------------ dataset integrity


def test_dataset_schema_and_targets() -> None:
    """Every row matches the schema; size/positive/non-English targets hold."""
    for name in DATASET_FILES:
        path = DATA_DIR / name
        assert path.exists(), f"missing dataset file {name}"
        rows = _rows(path)
        check = name.removesuffix(".jsonl")
        ids: set[str] = set()
        for row in rows:
            assert REQUIRED_FIELDS <= set(row), f"{name}: missing fields in {row}"
            assert isinstance(row["text"], str) and row["text"].strip()
            assert row["label"] in (0, 1)
            assert row["check"] == check
            assert row["lang"] in ALLOWED_LANGS
            assert row["id"] not in ids, f"{name}: duplicate id {row['id']}"
            ids.add(row["id"])

        n = len(rows)
        positives = sum(r["label"] for r in rows)
        non_en = sum(1 for r in rows if r["lang"] != "en")
        assert n >= 160, f"{name}: {n} rows < 160"
        assert positives / n >= 0.40, f"{name}: positive ratio {positives / n:.2%} < 40%"
        assert non_en / n >= 0.25, f"{name}: non-en ratio {non_en / n:.2%} < 25%"


def test_dataset_non_english_spread() -> None:
    """Non-English rows are spread across np/es/fr/de/hi/ar in every file."""
    needed = ALLOWED_LANGS - {"en"}
    for name in DATASET_FILES:
        langs = {r["lang"] for r in _rows(DATA_DIR / name)}
        assert needed <= langs, f"{name}: missing langs {sorted(needed - langs)}"


def test_manifest_matches_dataset() -> None:
    """manifest.json counts equal the actual JSONL counts."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == set(DATASET_FILES)
    assert "generation_notes" in manifest and manifest["generation_notes"]

    total_n = total_positives = 0
    total_by_lang: dict[str, int] = {}
    for name, stats in manifest["files"].items():
        rows = _rows(DATA_DIR / name)
        positives = sum(r["label"] for r in rows)
        by_lang: dict[str, int] = {}
        for row in rows:
            by_lang[row["lang"]] = by_lang.get(row["lang"], 0) + 1
            total_by_lang[row["lang"]] = total_by_lang.get(row["lang"], 0) + 1
        assert stats["n"] == len(rows), f"{name}: manifest n mismatch"
        assert stats["positives"] == positives, f"{name}: manifest positives mismatch"
        assert stats["by_lang"] == by_lang, f"{name}: manifest by_lang mismatch"
        total_n += len(rows)
        total_positives += positives

    assert manifest["total"]["n"] == total_n
    assert manifest["total"]["positives"] == total_positives
    assert manifest["total"]["by_lang"] == total_by_lang
