#!/usr/bin/env bash
# =============================================================================
# Aurelius 1.3B — Pre-training launch script for H100 cloud cluster
#
# Usage:
#   Single node (8x H100):  bash scripts/run_training.sh
#   Multi-node:              Set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT
#   Resume:                  RESUME_FROM=checkpoints/aurelius-1.3b/step-0004800 bash scripts/run_training.sh
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/train_1b.yaml"
DS_CONFIG="${PROJECT_ROOT}/configs/deepspeed_zero1.json"

# ── Cluster configuration ───────────────────────────────────────────────────
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"        # 8x H100 per node
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── Resume ───────────────────────────────────────────────────────────────────
RESUME_FLAG=""
if [[ -n "${RESUME_FROM:-}" ]]; then
    RESUME_FLAG="--resume ${RESUME_FROM}"
    echo ">>> Resuming from checkpoint: ${RESUME_FROM}"
fi

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Flash Attention 2 settings
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# W&B (set your API key via WANDB_API_KEY env var or wandb login)
export WANDB_PROJECT="aurelius-1.3b"

echo "======================================================================"
echo "  Aurelius 1.3B Pre-training"
echo "  Nodes: ${NNODES}  |  GPUs/node: ${NPROC_PER_NODE}  |  Total GPUs: $((NNODES * NPROC_PER_NODE))"
echo "  Config: ${CONFIG}"
echo "  DeepSpeed: ${DS_CONFIG}"
echo "======================================================================"

# ── Launch ───────────────────────────────────────────────────────────────────
accelerate launch \
    --multi_gpu \
    --num_machines "${NNODES}" \
    --num_processes "$((NNODES * NPROC_PER_NODE))" \
    --machine_rank "${NODE_RANK}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${DS_CONFIG}" \
    -m training.trainer \
    --config "${CONFIG}" \
    ${RESUME_FLAG}
