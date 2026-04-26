#!/usr/bin/env bash
set -euo pipefail

# Aurelius local development launcher
# Starts API server and optional frontend dev server.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export AURELIUS_LOG_LEVEL="${AURELIUS_LOG_LEVEL:-debug}"
export AURELIUS_HOST="${AURELIUS_HOST:-127.0.0.1}"
export AURELIUS_PORT="${AURELIUS_PORT:-8080}"

start_api() {
    echo "Starting Aurelius API server on $AURELIUS_HOST:$AURELIUS_PORT..."
    python -m src.serving.api_server --host "$AURELIUS_HOST" --port "$AURELIUS_PORT"
}

start_frontend() {
    echo "Starting frontend dev server..."
    cd frontend
    npm run dev
}

case "${1:-api}" in
    api)
        start_api
        ;;
    frontend)
        start_frontend
        ;;
    all)
        start_api &
        API_PID=$!
        start_frontend &
        FRONTEND_PID=$!
        trap "kill $API_PID $FRONTEND_PID" EXIT
        wait
        ;;
    *)
        echo "Usage: $0 {api|frontend|all}"
        exit 1
        ;;
esac
