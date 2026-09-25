# Contributing to GateLaya

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/Diwas2055/gatelaya
cd gatelaya
uv sync                    # base deps + dev group (pytest, ruff, httpx)
uv sync --extra model      # optional: adds laya (downloads model weights)
```

## Before you open a PR

```bash
uv run ruff check gatelaya/ dashboard/ scripts/ tests/
uv run pytest tests/ -q
uv run scripts/eval.py --agent fake --limit 8   # eval plumbing sanity
```

CI runs the same three checks on Python 3.12 and 3.13.

## Guidelines

- **Tests required** for behavior changes — follow the existing `tests/` patterns (FakeAgent, no model downloads in pytest).
- **Type hints** and pydantic v2 for public APIs; one-line docstrings.
- **Keep the fail-open contract**: guardrail code must never crash the LiteLLM proxy — errors degrade to `allow` + audit row.
- **Eval changes**: if you touch prediction semantics, run `uv run scripts/eval.py --gate` with the real model (`--extra model`) and report the numbers in your PR.
- Match surrounding style; surgical changes (no drive-by refactors).

## Reporting bugs

Use the bug report template. Include: GateLaya version, litellm version, Python version, config (redact secrets), minimal repro, expected vs actual.

## Security

Do not open public issues for security vulnerabilities — use GitHub Security Advisories on this repo.
