#!/usr/bin/env bash
# =============================================================================
# Aurelius 2.7B — Pre-training launch script for M1 Pro with 32GB
#
# Architecture:
#   d_model=2560, n_layers=32, n_heads=20, n_kv_heads=5, d_ff=7168
# Memory: ~17.3GB with gradient checkpointing + Muon+AdamW + bf16 + bs=1
#
# Usage:
#   bash scripts/train_2.7b.sh                    # Fresh training
#   RESUME=checkpoints/aurelius-2.7b/step-0048000 bash scripts/train_2.7b.sh  # Resume
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/train_2.7b.yaml"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "======================================================================"
echo "  Aurelius 2.7B Pre-training (M1 Pro 32GB)"
echo "  Config: ${CONFIG}"
echo "  Device: MPS (Apple Silicon)"
echo "======================================================================"

RESUME_FLAG=""
if [[ -n "${RESUME:-}" ]]; then
    RESUME_FLAG="--resume ${RESUME}"
    echo ">>> Resuming from: ${RESUME}"
fi

python -m training.trainer \
    --config "${CONFIG}" \
    ${RESUME_FLAG}
