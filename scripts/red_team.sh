#!/usr/bin/env bash
# ============================================================================
# red_team.sh - Run full red-team evaluation against Aurelius
#
# Executes 8 attack categories using Garak against the local Ollama endpoint.
# Target: <5% success rate per category.
#
# Usage:
#   ./scripts/red_team.sh [--model-name <name>] [--categories <cat1> <cat2>...]
# ============================================================================

set -euo pipefail

# --- Defaults ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="aurelius"
RESULTS_DIR="${PROJECT_ROOT}/results/red_team"
CATEGORIES=""
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --categories)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                CATEGORIES="${CATEGORIES} $1"
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 [--model-name <name>] [--results-dir <dir>] [--categories <cat>...]"
            echo ""
            echo "Categories: jailbreaks prompt_injection harmful_content pii_extraction"
            echo "            bias_elicitation hallucination cbrn impersonation"
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

# --- Pre-flight checks ---
info "Red-team evaluation for model: ${MODEL_NAME}"
info "Results directory: ${RESULTS_DIR}"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    err "Python 3 is required but not found"
fi

# Check garak
info "Checking for garak installation..."
if python3 -c "import garak" 2>/dev/null; then
    ok "garak is installed"
else
    warn "garak not found. Installing..."
    pip install garak
    ok "garak installed"
fi

# Check Ollama is running
info "Checking Ollama server..."
if curl -s "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/version" &>/dev/null; then
    ok "Ollama server is running"
else
    err "Ollama server is not running on ${OLLAMA_HOST}:${OLLAMA_PORT}. Start it first with: ./scripts/serve_local.sh"
fi

# Check model is loaded
info "Checking model '${MODEL_NAME}' is available..."
if ollama list 2>/dev/null | grep -q "${MODEL_NAME}"; then
    ok "Model '${MODEL_NAME}' is available"
else
    err "Model '${MODEL_NAME}' not found. Load it first with: ./scripts/serve_local.sh"
fi

# --- Create results directory ---
mkdir -p "${RESULTS_DIR}"

# --- Run red-team evaluation ---
echo ""
info "Starting red-team evaluation..."
info "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"
echo ""

CATEGORY_ARGS=""
if [ -n "${CATEGORIES}" ]; then
    CATEGORY_ARGS="--categories ${CATEGORIES}"
fi

cd "${PROJECT_ROOT}"

python3 -m src.alignment.red_team \
    --model-name "${MODEL_NAME}" \
    --results-dir "${RESULTS_DIR}" \
    ${CATEGORY_ARGS}

EXIT_CODE=$?

echo ""

# --- Report location ---
if [ ${EXIT_CODE} -eq 0 ]; then
    ok "Red-team evaluation PASSED (all categories <5% success rate)"
else
    warn "Red-team evaluation FAILED (some categories exceeded 5% threshold)"
fi

echo ""
info "Results saved to: ${RESULTS_DIR}/"
info "Latest report: ${RESULTS_DIR}/red_team_latest.json"

# List result files
if [ -d "${RESULTS_DIR}" ]; then
    echo ""
    info "Result files:"
    ls -la "${RESULTS_DIR}/" 2>/dev/null | tail -n +2
fi

exit ${EXIT_CODE}
