FROM python:3.14-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
COPY src/ src/
RUN uv sync --no-dev
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8080/healthz || exit 1
ENTRYPOINT ["python", "-m", "src.serving"]
