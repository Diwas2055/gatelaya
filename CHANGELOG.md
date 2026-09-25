# Changelog

## v0.1.0 (2026-09-25) — first release

### Core
- **GateLayaGuardrail** — LiteLLM `CustomGuardrail` pack: pii / injection / toxicity / secret_leak via Laya typed decisions, pre-call + post-call hooks, block/mask/flag actions, fail-open mode, audit sinks (in-memory / file / SQLite / Postgres)
- **GateLayaRouter** — pre-call model routing: task / complexity / sensitive signals rewrite `data["model"]` (tiers, low-confidence fallback, never blocks)
- **Temperature calibration** — `scripts/calibrate.py`, per-check temperature maps, ECE repair
- **Fine-tuned checkpoint** — RLCD fine-tune on GateLaya eval set; held-out gate **PASS: macro acc 0.908 / ECE 0.086** (targets 0.90 / 0.15); enable via `GATELAYA_MODEL_PATH`

### Eval
- `gatelaya/metrics.py` — accuracy, P/R/F1, ECE (Guo 2017), ROC AUC, sweep, gate report
- `evals/data/` — 654 hand-curated rows (4 checks, 7 languages, 0 mislabels found by QA) + deterministic 70/15/15 splits
- `scripts/eval.py` — `--gate` CI mode, `--tune` (leakage-free CV calibration + threshold search), `--model` override, `--agent fake` sanity mode
- `scripts/finetune.py` — official Laya RLCD recipe on MPS/CUDA/CPU; `scripts/prepare_data.py` QA + splits

### Dashboard (Phase 3)
- FastAPI app (`dashboard/`): stats, decision search, flag review queue with human labels + JSONL export, config editing (thresholds/actions, restart-required semantics), calibration upload, Bearer-token auth
- Single-page Alpine.js + Tailwind UI, 4 tabs, dark theme

### Packaging
- uv-managed: `uv.lock`, PEP 735 dev group, extras `model` / `train` / `redis`
- Docker: `Dockerfile` (python:3.12-slim) + `docker-compose.yml` (litellm-gatelaya, postgres, redis)
- `litellm[proxy]` runtime dep; console scripts `gatelaya-dashboard`
- Apache License 2.0

### Verification
- 318 tests passing (`uv run pytest tests/ -q`)
- Live proxy boot: injection → HTTP 400, clean → pass-through, PII masked (pre-release smoke)
