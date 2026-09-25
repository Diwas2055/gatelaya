## What does this PR change?

<!-- Summary + motivation. Link related issues: Fixes #123 -->

## How was it verified?

- [ ] `uv run ruff check gatelaya/ dashboard/ scripts/ tests/`
- [ ] `uv run pytest tests/ -q`
- [ ] Eval impact considered (`uv run scripts/eval.py --gate` with real model, numbers below) — delete if N/A

<!-- paste numbers if run -->

## Risk

- [ ] Fail-open contract preserved (no path can crash the proxy)
- [ ] No secrets / local paths / PII added
- [ ] Public-repo safe
