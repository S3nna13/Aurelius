#!/usr/bin/env bash
# ============================================================================
# serve_local.sh - One-command local Aurelius serving setup
#
# Installs Ollama if needed, loads the GGUF model, and starts serving.
# Designed for Apple M1 Pro with 16GB unified memory.
#
# Usage:
#   ./scripts/serve_local.sh [--model-path <path>] [--model-name <name>]
# ============================================================================

set -euo pipefail

# --- Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="aurelius"
MODEL_PATH="${PROJECT_ROOT}/models/gguf/aurelius-1.3b-q4_k_m.gguf"
MODELFILE_PATH="${PROJECT_ROOT}/configs/ollama.Modelfile"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            OLLAMA_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model-path <path>] [--model-name <name>] [--port <port>]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --- Color output helpers ---
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
err()   { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# --- Step 1: Check / Install Ollama ---
info "Checking for Ollama installation..."

if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 || echo "unknown")
    ok "Ollama found: ${OLLAMA_VERSION}"
else
    info "Ollama not found. Installing via Homebrew..."

    if ! command -v brew &>/dev/null; then
        err "Homebrew is required to install Ollama. Install from https://brew.sh"
    fi

    brew install ollama
    ok "Ollama installed successfully"
fi

# --- Step 2: Ensure Ollama is running ---
info "Checking if Ollama server is running..."

if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/version" &>/dev/null; then
    ok "Ollama server is already running on port ${OLLAMA_PORT}"
else
    info "Starting Ollama server..."
    OLLAMA_HOST="${OLLAMA_HOST}" ollama serve &
    OLLAMA_PID=$!

    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/version" &>/dev/null; then
            ok "Ollama server started (pid=${OLLAMA_PID})"
            break
        fi
        if [ "$i" -eq 30 ]; then
            err "Ollama server failed to start within 30 seconds"
        fi
        sleep 1
    done
fi

# --- Step 3: Verify model file exists ---
info "Checking for GGUF model at: ${MODEL_PATH}"

if [ ! -f "${MODEL_PATH}" ]; then
    warn "Model file not found at ${MODEL_PATH}"
    warn "Run the conversion pipeline first:"
    warn "  python -m src.serving.convert_to_gguf --hf-model-path <path>"
    err "Cannot proceed without model file"
fi

ok "Model file found ($(du -h "${MODEL_PATH}" | cut -f1) )"

# --- Step 4: Create the Ollama model from Modelfile ---
info "Creating Ollama model '${MODEL_NAME}' from Modelfile..."

# Update the Modelfile to point to the actual model path
TEMP_MODELFILE=$(mktemp)
sed "s|^FROM .*|FROM ${MODEL_PATH}|" "${MODELFILE_PATH}" > "${TEMP_MODELFILE}"

ollama create "${MODEL_NAME}" -f "${TEMP_MODELFILE}"
rm -f "${TEMP_MODELFILE}"

ok "Model '${MODEL_NAME}' created successfully"

# --- Step 5: Verify the model is loaded ---
info "Verifying model is available..."

if ollama list | grep -q "${MODEL_NAME}"; then
    ok "Model '${MODEL_NAME}' is registered with Ollama"
else
    err "Model '${MODEL_NAME}' not found in Ollama registry"
fi

# --- Step 6: Run a quick smoke test ---
info "Running smoke test..."

SMOKE_RESPONSE=$(ollama run "${MODEL_NAME}" "Say hello in one sentence." 2>&1 || true)

if [ -n "${SMOKE_RESPONSE}" ]; then
    ok "Smoke test passed. Response:"
    echo "  ${SMOKE_RESPONSE:0:200}"
else
    warn "Smoke test returned empty response"
fi

# --- Done ---
echo ""
echo "============================================"
echo "  Aurelius is ready for local inference"
echo "============================================"
echo "  Model:    ${MODEL_NAME}"
echo "  Endpoint: http://${OLLAMA_HOST}:${OLLAMA_PORT}"
echo ""
echo "  Quick test:"
echo "    ollama run ${MODEL_NAME} 'What is the meaning of life?'"
echo ""
echo "  Python client:"
echo "    python -m src.serving.chat_client --model ${MODEL_NAME}"
echo "============================================"
