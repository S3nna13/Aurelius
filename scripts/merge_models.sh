#!/usr/bin/env bash
# Merge SFT + GRPO checkpoints using mergekit
# Usage: ./scripts/merge_models.sh [slerp|ties]
set -euo pipefail

METHOD=${1:-slerp}
if [[ "$METHOD" != "slerp" && "$METHOD" != "ties" ]]; then
    echo "Error: method must be 'slerp' or 'ties', got '$METHOD'" >&2
    exit 1
fi

CONFIG="configs/merge_${METHOD}.yaml"
OUT_PATH=$(python3 -c "import yaml; d=yaml.safe_load(open('${CONFIG}')); print(d['output_path'])")

mergekit-merge "${CONFIG}" --out-path "${OUT_PATH}" --cuda
