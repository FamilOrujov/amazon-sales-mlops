FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY models/ ./models/

RUN uv pip install -e . --no-deps

# Production image
FROM python:3.13-slim AS production

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs
COPY --from=builder /app/scripts ./scripts
COPY --from=builder /app/models ./models
COPY --from=builder /app/pyproject.toml ./

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV DOCKER_MODE=true

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "amazon_sales_ml.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

