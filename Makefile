# GateLaya — common dev commands (uv-first)
.PHONY: sync sync-model lint test eval eval-gate run docker-build docker-build-slim docker-up docker-down release

sync:            ## install base + dev deps
	uv sync

sync-model:      ## install with Laya inference extras
	uv sync --extra model

lint:            ## ruff gate (same as CI)
	uv run ruff check gatelaya/ dashboard/ scripts/ tests/

test:            ## full suite
	uv run pytest tests/ -q

eval:            ## eval plumbing sanity (no model)
	uv run scripts/eval.py --agent fake --limit 8

eval-gate:       ## real eval vs PRODUCT.md gate (needs sync-model)
	uv run scripts/eval.py --gate

run:             ## dashboard on :8080
	uv run gatelaya-dashboard

docker-build:    ## full image (default target = model)
	docker build -t gatelaya:dev .

docker-build-slim:
	docker build --target slim -t gatelaya:dev-slim .

docker-up:       ## litellm(:4000) + dashboard(:8080) + postgres + redis
	docker compose up --build -d

docker-down:
	docker compose down

release: lint test eval  ## pre-tag gate
	@echo "run: git tag v<X.Y.Z> && git push origin v<X.Y.Z>"
