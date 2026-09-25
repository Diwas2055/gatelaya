# Product

## Name
**GateLaya** — LLM Firewall Gateway

## One-liner
Multilingual guardrail layer that plugs into LiteLLM and blocks PII leaks, prompt injection, and toxic content before and after every LLM call — running entirely on your own hardware.

## Stack
- **Language**: Python 3.12
- **Gateway**: LiteLLM Proxy (CustomGuardrail / CustomLogger hooks)
- **Decision model**: Laya (`laya` PyPI package, Apache 2.0)
  - `convaiinnovations/laya` (English, ModernBERT-large 421M)
  - `convaiinnovations/laya-multilingual` (mmBERT-base 322M, 100+ languages)
  - selected at runtime via Laya `Router`
- **API / services**: FastAPI, async SQLAlchemy
- **Data**: PostgreSQL (decision audit log), Redis (LiteLLM cache — thresholds come from env/YAML, not Redis)
- **Queue**: Celery (low-confidence human-review jobs) — later phases
- **Frontend**: Alpine.js + Tailwind CSS (dashboard) — later phases
- **Packaging**: Docker / docker-compose; install from source `pip install -e .` (PyPI publish planned)

## Problem
LLM guardrails today are either cloud APIs (data leaves the building, English-skewed, per-call cost) or regex (misses meaning, no multilingual). Enterprises deploying LLMs in regulated or multilingual environments have no local, cheap, calibrated option.

## Solution
A drop-in guardrail pack: Laya answers typed questions (`choice`, `score`, `noul`) about every request/response in one 33ms forward pass — PII present, injection attempt, toxic, language — and the LiteLLM hook blocks, masks, flags, or passes. No text generation → nothing to hallucinate. Runs on-prem, $0 per check.

## Users
- **Platform engineers** — wire the pack into an existing LiteLLM proxy via `proxy_config.yaml`.
- **Trust & Safety ops** — tune thresholds, review flagged/blocked traffic in the dashboard.
- **Compliance / security** — export the decision audit log as evidence (why request X was blocked).

## Positioning
The open, local, multilingual alternative to cloud moderation APIs for LLM gateways. Differentiators: typed decisions with calibrated probabilities, 100+ languages in one model, air-gap capable, Apache 2.0, zero per-token cost.

## Core Workflows
1. **Guard (input)**: request hits LiteLLM → `pre_call` hook runs Laya → PII masked / injection blocked / allowed with decision logged.
2. **Guard (output)**: LLM responds → `post_call` hook runs Laya → secrets/PII/toxicity in response blocked or flagged.
3. **Tune**: operator loads calibration set → refit temperatures per (question type, option count) → `calibration.json` regenerated and pointed to via `GATELAYA_CALIBRATION_PATH`.
4. **Review**: flagged request enters review UI → human confirms/rejects → label feeds next calibration.
5. **Audit**: compliance queries the decision log by time range, key, team, language, outcome.

## Functional Capabilities
- LiteLLM integration: `CustomGuardrail` with `pre_call` and `post_call` modes (`during_call` on roadmap)
- Checks: PII detect + type, prompt injection/jailbreak, toxicity, secret leakage, language bucket detection
- Actions: `allow` / `mask` (modify) / `block` / `flag` (audit-only)
- Confidence gating: one threshold per check, process-wide (per-key/team scoping on roadmap)
- Temperature calibration script (required before trusting probabilities)
- Decision audit log: input hash, question, probabilities, action, latency
- OpenAI-compatible passthrough — clients need no code change
- On-prem, Docker one-command deploy

## Non-Goals (v1)
- No text generation, no chat
- No automatic model routing (Phase 2: routing mode)
- No UGC batch pipeline (separate product: PolyGlotGate)
- No training — inference + calibration only

## Constraints
- Laya base checkpoints are near-chance zero-shot for complex decisions → ship with narrow checks (binary/few-option) + calibration; fine-tune documented as extension
- `choice` ≤ ~20 options per question
- No ordinal `score` questions (position bias in multilingual checkpoint) — use ordered `choice`
- Shipped model is uncalibrated → calibration script is a mandatory setup step, not optional
- Checkpoint routing is ASCII/Latin-script heuristic: pure-ASCII → English checkpoint, otherwise multilingual (no language-ID model in v1)

## Success Criteria
- P95 guardrail overhead < 100ms per request (model forward ~33ms + hook overhead) — measurable now
- Injection/PII block accuracy ≥ 0.90 and Mean ECE ≤ 0.15 — **not met (first run)**: labeled eval set (`evals/data/`, 654 rows) + `scripts/eval.py` shipped; real-model run 2026-09-25 (laya 0.3.20, zero-shot, uncalibrated): pii+injection macro accuracy 0.669, ECE 0.213 → **FAIL**. Next: calibrate on labeled traffic, then re-run; fine-tune if zero-shot ceiling holds
- Drop-in: existing OpenAI client works unchanged against guarded LiteLLM endpoint — verified by integration tests
- Full audit coverage: 100% of guarded requests logged with decision + probabilities — covered by unit tests

## Brand
- Name: **GateLaya**
- Tagline: *Every LLM call, decided.*
- Palette: slate/violet/signal-amber
- Tone: precise, security-serious, developer-first
