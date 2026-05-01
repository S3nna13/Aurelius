#!/usr/bin/env bash
# =============================================================================
# Aurelius 3.0B — Pre-training launch script for M1 Pro with 32GB
#
# Architecture:
#   d_model=3072, n_layers=28, n_heads=24, n_kv_heads=6, d_ff=8192, seq_len=4096
#
# Memory: ~20.5GB with gradient checkpointing + 8-bit Muon + bf16 + bs=1
# Start with seq_len=4096 to keep activations manageable.
# Switch to 8192 via YaRN for final ~100B tokens.
#
# Usage:
#   bash scripts/train_3b.sh                    # Fresh training
#   RESUME=checkpoints/aurelius-3b/step-0048000 bash scripts/train_3b.sh  # Resume
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/train_3b.yaml"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

echo "======================================================================"
echo "  Aurelius 3.0B Pre-training (M1 Pro 32GB)"
echo "  Config: ${CONFIG}"
echo "  Device: MPS (Apple Silicon)"
echo "  Warning: 3B requires 8-bit optimizer or optimizer offloading"
echo "======================================================================"

RESUME_FLAG=""
if [[ -n "${RESUME:-}" ]]; then
    RESUME_FLAG="--resume ${RESUME}"
    echo ">>> Resuming from: ${RESUME}"
fi

python -m training.trainer \
    --config "${CONFIG}" \
    ${RESUME_FLAG}
