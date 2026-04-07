#!/usr/bin/env bash
# Convert a HuggingFace checkpoint to GGUF Q4_K_M.
# Usage: convert_to_gguf.sh <hf-model-path> [output-dir]
set -euo pipefail
HF_MODEL="${1:?Usage: $0 <hf-model-path> [output-dir]}"
OUT_DIR="${2:-models/gguf}"
mkdir -p "$OUT_DIR"
python vendor/llama.cpp/convert_hf_to_gguf.py "$HF_MODEL" --outfile "$OUT_DIR/aurelius-1.3b-f16.gguf" --outtype f16
llama-quantize "$OUT_DIR/aurelius-1.3b-f16.gguf" "$OUT_DIR/aurelius-1.3b-q4_k_m.gguf" Q4_K_M
rm "$OUT_DIR/aurelius-1.3b-f16.gguf"
echo "Done: $OUT_DIR/aurelius-1.3b-q4_k_m.gguf"
