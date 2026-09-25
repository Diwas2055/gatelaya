# GateLaya module

Multilingual LLM guardrail pack: the Laya decision model plugged into LiteLLM
Proxy as a `CustomGuardrail`. Full product docs live in the repo README; this
file covers the `gatelaya` package itself.

## Install

```bash
pip install -e .              # core (litellm, fastapi, sqlalchemy, pydantic)
pip install -e ".[model]"     # + laya (decision model, downloads ~800MB weights)
pip install -e ".[dev]"       # + pytest, pytest-asyncio, aiosqlite
```

`laya` is imported lazily — the package imports and tests run without it;
the first prediction raises `LayaNotInstalledError` with install instructions.

## Programmatic use

```python
from gatelaya import GateLayaGuardrail, GateLayaConfig
from gatelaya.agent import LayaRouterAgent
from gatelaya.audit import InMemoryAuditSink

guardrail = GateLayaGuardrail(
    config=GateLayaConfig(),                      # thresholds, actions, routing
    agent=LayaRouterAgent(),                      # lazy Laya checkpoints
    audit=InMemoryAuditSink(),                    # or SqlAlchemyAuditSink(url)
    temperatures=None,                            # TemperatureMap from calibration
)
result = await guardrail.async_pre_call_hook(
    user_api_key_dict=None, cache=None,
    data={"messages": [{"role": "user", "content": "hi"}]},
    call_type="acompletion",
)
# result: None (allow/flag) | str (block -> HTTP 400) | dict (masked messages)
```

When LiteLLM constructs the class from `proxy_config.yaml`, `config`/`agent`/
`audit`/`temperatures` are omitted and defaults are built from `GATELAYA_*`
environment variables (see `proxy_config.yaml` header).

## Checks and actions

| check       | question type | default threshold | default action |
|-------------|---------------|-------------------|----------------|
| `pii`       | noul + choice | 0.85              | mask           |
| `injection` | noul          | 0.90              | block          |
| `toxicity`  | noul          | 0.90              | block          |
| `secret_leak` | noul        | 0.85              | block          |

Actions: `allow` (continue), `mask` (pre-call PII redaction), `block`
(pre-call returns a 400 error string; post-call raises `HTTPException(400)`),
`flag` (continue, decision metadata stashed in `data["gatelaya"]`).

Post-call runs `secret_leak` + `toxicity` on the response text.

## Calibration (required before trusting probabilities)

```bash
python scripts/calibrate.py --input labeled.jsonl --check injection \
    --output calibration.json
```

Input JSONL per line: `{"text": ..., "label": ..., "check": "injection"}`;
`label` is `yes`/`no` (or `0`/`1`) for noul questions, an option name for
choice questions (`"question": "pii_type"` on the row). Point the guardrail at
the result with `GATELAYA_CALIBRATION_PATH` or `config.calibration_path`.

## v1 limitations (documented)

- **Masking is regex-based** (email, phone, credit card, SSN). When the model
  answers `pii_type` and the regex found anything, the entire last user
  message collapses to `[REDACTED:pii]`. Regex-blind PII (non-Latin names,
  handles) is *not* redacted — the request is only flagged.
- **Streaming responses pass through unenforced** — the streaming hook is a
  pass-through; streamed responses are not checked or audited in v1.
- **`mask` on response checks degrades to `flag`** (no secret-redaction
  patterns yet).
- **Routing heuristic**: ASCII-only text → `convaiinnovations/laya`,
  anything else → `convaiinnovations/laya-multilingual`.
- **fail_open=True (default)**: if the agent raises, the request is allowed
  and an `agent_error` audit record is written; `fail_open=False` blocks.

## Audit

Every decision (allow/mask/block/flag) is recorded as a `DecisionRecord`:
check, language bucket, raw + calibrated probs, action, input SHA-256,
latency, mode. Sinks: `InMemoryAuditSink` (capped) and `SqlAlchemyAuditSink`
(async, table `guardrail_decisions`, `create_all` on first write; Postgres via
`asyncpg`, SQLite via `aiosqlite`).
