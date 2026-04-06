#!/usr/bin/env bash
# ===========================================================================
# Aurelius 1.3B -- Full data preparation pipeline
# ===========================================================================
#
# Usage:
#   bash scripts/prepare_data.sh                  # defaults (1B token sample)
#   bash scripts/prepare_data.sh --full           # full FineWeb pipeline
#   bash scripts/prepare_data.sh --sample-only    # download sample only
#
# Prerequisites:
#   - Python 3.12+ with venv or uv
#   - pip install datatrove datasets pyarrow
#
# The pipeline is fully resumable. Re-run after a crash and it picks up
# where it left off (DataTrove marker files).
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- override via environment variables
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs/data_pipeline}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

SAMPLE_TOKENS="${SAMPLE_TOKENS:-1000000000}"       # 1B tokens
PIPELINE_TASKS="${PIPELINE_TASKS:-8}"              # parallel tasks (tune to CPU count)
PIPELINE_WORKERS="${PIPELINE_WORKERS:--1}"         # -1 = auto

# HuggingFace dataset
HF_DATASET="${HF_DATASET:-HuggingFaceFW/fineweb}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

check_python() {
    if ! command -v python3 &>/dev/null; then
        die "python3 not found. Install Python 3.12+."
    fi
    local pyver
    pyver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    log "Using Python ${pyver}"
}

ensure_venv() {
    if [[ ! -d "${VENV_DIR}" ]]; then
        log "Creating virtual environment at ${VENV_DIR}"
        python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
}

install_deps() {
    log "Installing / upgrading dependencies"
    pip install --quiet --upgrade pip
    pip install --quiet \
        "datatrove[all]>=0.3.0" \
        "datasets>=3.0.0" \
        "pyarrow>=17.0.0" \
        "huggingface_hub>=0.25.0"
}

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
download_sample() {
    log "Downloading ~${SAMPLE_TOKENS} token sample from ${HF_DATASET}"
    python3 -m src.data.download_sample \
        --dataset "${HF_DATASET}" \
        --target-tokens "${SAMPLE_TOKENS}" \
        --output-dir "${DATA_DIR}/fineweb/sample" \
        --seed 42
    log "Sample downloaded to ${DATA_DIR}/fineweb/sample/"
}

run_pipeline() {
    local input_path="${1:-${DATA_DIR}/fineweb/sample}"
    local output_path="${2:-${DATA_DIR}/fineweb/clean}"

    log "Running DataTrove pipeline"
    log "  Input:  ${input_path}"
    log "  Output: ${output_path}"
    log "  Tasks:  ${PIPELINE_TASKS}"

    python3 -m src.data.pipeline \
        --input-path "${input_path}" \
        --output-path "${output_path}" \
        --minhash-work-dir "${DATA_DIR}/fineweb/minhash" \
        --input-format parquet \
        --tasks "${PIPELINE_TASKS}" \
        --workers "${PIPELINE_WORKERS}" \
        --logging-dir "${LOGS_DIR}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local mode="${1:-default}"

    cd "${PROJECT_ROOT}"
    check_python

    mkdir -p "${DATA_DIR}" "${LOGS_DIR}"

    ensure_venv
    install_deps

    case "${mode}" in
        --sample-only)
            download_sample
            ;;
        --full)
            log "Running FULL pipeline (FineWeb from HuggingFace)"
            run_pipeline "hf://datasets/${HF_DATASET}" "${DATA_DIR}/fineweb/clean"
            ;;
        --local|default)
            download_sample
            run_pipeline "${DATA_DIR}/fineweb/sample" "${DATA_DIR}/fineweb/clean"
            ;;
        --help|-h)
            sed -n '2,/^# ====/{ /^# ====/d; s/^# \{0,2\}//; p; }' "$0"
            exit 0
            ;;
        *)
            die "Unknown mode: ${mode}. Use --sample-only, --full, --local, or --help."
            ;;
    esac

    log "Data preparation complete!"
    log "Clean data:  ${DATA_DIR}/fineweb/clean/deduped/"
    log "Logs:        ${LOGS_DIR}/"
}

main "$@"
