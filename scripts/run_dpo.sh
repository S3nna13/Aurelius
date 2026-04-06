#!/usr/bin/env bash
# =============================================================================
# Aurelius 1.3B — DPO alignment launch script
#
# Usage:
#   Single GPU:   bash scripts/run_dpo.sh
#   Multi-GPU:    NPROC=4 bash scripts/run_dpo.sh
#   Custom model: MODEL=path/to/sft/model bash scripts/run_dpo.sh
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Configuration (override via env vars) ────────────────────────────────────
MODEL="${MODEL:-checkpoints/aurelius-1.3b-sft/final}"
TOKENIZER="${TOKENIZER:-tokenizers/aurelius-128k}"
REF_MODEL="${REF_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/aurelius-1.3b-dpo}"
NPROC="${NPROC:-1}"

# Hyperparameters
BETA="${BETA:-0.1}"
LR="${LR:-5e-7}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MIN_SCORE_GAP="${MIN_SCORE_GAP:-1.0}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEED="${SEED:-42}"

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="aurelius-1.3b-dpo"

# ── Build CLI args ───────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [[ -n "${REF_MODEL}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --ref-model ${REF_MODEL}"
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-samples ${MAX_SAMPLES}"
fi

echo "======================================================================"
echo "  Aurelius 1.3B — Direct Preference Optimization"
echo "  SFT Model:  ${MODEL}"
echo "  Beta:       ${BETA}"
echo "  LR:         ${LR}, Epochs: ${EPOCHS}"
echo "  Batch:      ${BATCH_SIZE}"
echo "  Score gap:  >= ${MIN_SCORE_GAP}"
echo "  GPUs:       ${NPROC}"
echo "======================================================================"

# ── Launch ───────────────────────────────────────────────────────────────────
if [[ "${NPROC}" -gt 1 ]]; then
    accelerate launch \
        --multi_gpu \
        --num_processes "${NPROC}" \
        --mixed_precision bf16 \
        -m alignment.dpo \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --output-dir "${OUTPUT_DIR}" \
        --beta "${BETA}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --min-score-gap "${MIN_SCORE_GAP}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
else
    python -m alignment.dpo \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --output-dir "${OUTPUT_DIR}" \
        --beta "${BETA}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --min-score-gap "${MIN_SCORE_GAP}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
fi
