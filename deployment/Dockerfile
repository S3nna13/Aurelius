FROM python:3.14-slim
WORKDIR /app

# Install curl for HEALTHCHECK and uv for dependency sync
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Sync production dependencies (locked)
RUN uv sync --no-dev --frozen

EXPOSE 8080

# Liveness probe matches the /health endpoint exposed by api_server.py
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["python", "-m", "src.serving"]
