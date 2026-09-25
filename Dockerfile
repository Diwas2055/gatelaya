FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY gatelaya/ gatelaya/
COPY custom_guardrail/ custom_guardrail/
COPY scripts/ scripts/
COPY proxy_config.yaml ./

# asyncpg: async Postgres driver for GATELAYA_DATABASE_URL
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir "litellm[proxy]" asyncpg

EXPOSE 4000

CMD ["litellm", "--config", "proxy_config.yaml", "--port", "4000"]
