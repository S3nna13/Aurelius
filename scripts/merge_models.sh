#!/usr/bin/env bash
# Merge SFT + GRPO checkpoints using mergekit
# Usage: ./scripts/merge_models.sh [slerp|ties]
set -euo pipefail
METHOD=${1:-slerp}
mergekit-merge "configs/merge_${METHOD}.yaml" --cuda
