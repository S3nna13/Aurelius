#!/usr/bin/env bash
# =============================================================================
# Aurelius 1.3B — SFT alignment launch script
#
# Usage:
#   Single GPU:   bash scripts/run_sft.sh
#   Multi-GPU:    NPROC=4 bash scripts/run_sft.sh
#   Custom model: MODEL=path/to/model bash scripts/run_sft.sh
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Configuration (override via env vars) ────────────────────────────────────
MODEL="${MODEL:-checkpoints/aurelius-1.3b/latest}"
TOKENIZER="${TOKENIZER:-tokenizers/aurelius-128k}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/aurelius-1.3b-sft}"
NPROC="${NPROC:-1}"

# Hyperparameters
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
USE_UNSLOTH="${USE_UNSLOTH:-true}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEED="${SEED:-42}"

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="aurelius-1.3b-sft"

# ── Build CLI args ───────────────────────────────────────────────────────────
EXTRA_ARGS=""
if [[ "${USE_UNSLOTH}" != "true" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no-unsloth"
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-samples ${MAX_SAMPLES}"
fi

echo "======================================================================"
echo "  Aurelius 1.3B — Supervised Fine-Tuning"
echo "  Model:      ${MODEL}"
echo "  LoRA:       r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "  LR:         ${LR}, Epochs: ${EPOCHS}"
echo "  Batch:      ${BATCH_SIZE}, Max seq: ${MAX_SEQ_LEN}"
echo "  Unsloth:    ${USE_UNSLOTH}"
echo "  GPUs:       ${NPROC}"
echo "======================================================================"

# ── Launch ───────────────────────────────────────────────────────────────────
if [[ "${NPROC}" -gt 1 ]]; then
    accelerate launch \
        --multi_gpu \
        --num_processes "${NPROC}" \
        --mixed_precision bf16 \
        -m alignment.sft \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --output-dir "${OUTPUT_DIR}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --lora-r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        --batch-size "${BATCH_SIZE}" \
        --max-seq-length "${MAX_SEQ_LEN}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
else
    python -m alignment.sft \
        --model "${MODEL}" \
        --tokenizer "${TOKENIZER}" \
        --output-dir "${OUTPUT_DIR}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --lora-r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        --batch-size "${BATCH_SIZE}" \
        --max-seq-length "${MAX_SEQ_LEN}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
fi
