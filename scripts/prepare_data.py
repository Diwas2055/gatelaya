#!/usr/bin/env python3
"""Run two-tier dataset QA and write honest held-out splits.

Outputs (under evals/data/):
  qa_report.json            — per-check coverage + strict-negative-hit report
  splits/train|val|test.jsonl + splits/manifest.json

Exit code 1 if QA fails (strict negative hits or coverage below threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from gatelaya.dataset_qa import (
        CHECKS,
        DEFAULT_SEED,
        MIN_TEST_PER_CHECK,
        SPLIT_RATIOS,
        qa_scan,
        split_dataset,
    )
except ModuleNotFoundError:  # running from a source checkout without install
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gatelaya.dataset_qa import (
        CHECKS,
        DEFAULT_SEED,
        MIN_TEST_PER_CHECK,
        SPLIT_RATIOS,
        qa_scan,
        split_dataset,
    )

DATA_DIR = Path(__file__).resolve().parents[1] / "evals" / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-test-per-check", type=int, default=MIN_TEST_PER_CHECK,
        help="borrow from train so each check has at least this many test rows",
    )
    return parser.parse_args(argv)


def load_rows(data_dir: Path) -> list[dict]:
    """Load all four check JSONL files."""
    rows: list[dict] = []
    for check in CHECKS:
        path = data_dir / f"{check}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing dataset file: {path}")
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows as JSONL (UTF-8, one object per line)."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(
    splits: dict[str, list[dict]], seed: int, min_test_per_check: int, files: list[str]
) -> dict:
    """Split counts by check and label, plus provenance."""
    return {
        "seed": seed,
        "ratios": list(SPLIT_RATIOS),
        "min_test_per_check": min_test_per_check,
        "source_files": files,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "by_split_check": {
            name: dict(Counter(r["check"] for r in rows)) for name, rows in splits.items()
        },
        "by_split_label": {
            name: {str(k): v for k, v in sorted(Counter(int(r["label"]) for r in rows).items())}
            for name, rows in splits.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """QA scan, then write report + splits."""
    args = parse_args(argv)
    data_dir: Path = args.data_dir
    rows = load_rows(data_dir)
    report = qa_scan(rows)

    report_path = data_dir / "qa_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"QA report -> {report_path}")
    for check, entry in report["per_check"].items():
        cov = entry["coverage"]
        cov_txt = "n/a" if cov is None else f"{cov:.3f} (min {entry['threshold']})"
        print(
            f"  {check:<12} pos={entry['positives']:<4} markers={cov_txt:<22} "
            f"strict_neg_hits={entry['strict_hits']}"
        )
        if entry["strict_hit_ids"]:
            print(f"    strict hit ids: {entry['strict_hit_ids']}")
        if entry["misses"]:
            print(f"    coverage misses: {entry['misses']}")

    splits = split_dataset(rows, seed=args.seed, min_test_per_check=args.min_test_per_check)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        write_jsonl(splits_dir / f"{name}.jsonl", split_rows)
    manifest = build_manifest(
        splits, args.seed, args.min_test_per_check, [f"{c}.jsonl" for c in CHECKS]
    )
    (splits_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Splits    -> {splits_dir}/ ({manifest['counts']})")

    if not report["passed"]:
        print("QA FAILED: fix strict negative hits or coverage misses above", file=sys.stderr)
        return 1
    print("QA passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
