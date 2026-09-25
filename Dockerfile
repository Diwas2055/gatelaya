# syntax=docker/dockerfile:1
# Two targets:
#   model (default) — litellm + dashboard + Laya model (torch); works out of the box
#   slim            — litellm + dashboard, no torch; guardrail fail-opens or use GATELAYA_MODEL_PATH

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# deps layer (shared by slim + model)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

FROM builder AS slim
COPY gatelaya/ gatelaya/
COPY custom_guardrail/ custom_guardrail/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY proxy_config.yaml proxy_config.yaml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra postgres
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 4000 8080
CMD ["litellm", "--config", "proxy_config.yaml", "--port", "4000"]

FROM builder AS model
COPY gatelaya/ gatelaya/
COPY custom_guardrail/ custom_guardrail/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY proxy_config.yaml proxy_config.yaml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra postgres --extra model
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 4000 8080
CMD ["litellm", "--config", "proxy_config.yaml", "--port", "4000"]
