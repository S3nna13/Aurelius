# syntax=docker/dockerfile:1
FROM python:3.12-slim
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy package metadata first for layer caching
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Install dependencies
RUN uv sync --no-dev

# Run as non-root user
RUN useradd -m -r -s /bin/bash aurelius && \
    chown -R aurelius:aurelius /app
USER aurelius

EXPOSE 8080

# Health check against the actual /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["python", "-m", "src.serving.api_server", "--host", "0.0.0.0", "--port", "8080"]
