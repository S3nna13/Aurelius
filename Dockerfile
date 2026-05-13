# syntax=docker/dockerfile:1

# Stage 1: builder — install dependencies and build the wheel
FROM python:3.12-slim-bullseye AS builder

WORKDIR /src

# System deps for PyTorch and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip tooling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Copy only dependency manifests for better caching
COPY pyproject.toml README.md LICENSE* ./

# Install project dependencies (including serve extras)
RUN pip install --no-cache-dir ".[full,serve]"

# Copy the rest of the source tree
COPY src/ ./src/
COPY gateway/ ./gateway/
COPY scripts/ ./scripts/
# tests optional; skip to keep image small
# COPY tests/ ./tests/

# Build a wheel (optional for reproducibility)
# RUN python -m build --wheel

# -----------------------------------------------------------------
# Stage 2: runtime — minimal image containing only runtime deps
FROM python:3.12-slim-bullseye

WORKDIR /app

# Create non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Copy installed packages from builder (site-packages)
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser gateway/ ./gateway/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Expose inference API port (FastAPI) and metrics
EXPOSE 8080

# Environment variables with sensible defaults
ENV AURELIUS_MODEL_PATH=/app/checkpoints/aurelius_1.3b \
    AURELIUS_BACKEND=vllm \
    TENSOR_PARALLEL_SIZE=1 \
    CORS_ORIGINS=*

# Health-check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

# Run the FastAPI production inference server
CMD ["python", "-m", "gateway.aurelius_api", "--host", "0.0.0.0", "--port", "8080"]
