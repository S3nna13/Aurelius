#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Train the Aurelius 128K BPE tokenizer on a ~10B-token streaming sample.
#
# Usage:
#   bash scripts/train_tokenizer.sh                  # defaults
#   DATASET=cerebras/SlimPajama-627B bash scripts/train_tokenizer.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────────────

DATASET="${DATASET:-allenai/dolma}"
SPLIT="${SPLIT:-train}"
TEXT_FIELD="${TEXT_FIELD:-text}"

# ~10B tokens ≈ roughly 5–7M documents depending on avg document length.
# Adjust MAX_SAMPLES to hit the desired token budget for your corpus.
MAX_SAMPLES="${MAX_SAMPLES:-6000000}"

VOCAB_SIZE="${VOCAB_SIZE:-128000}"
MIN_FREQUENCY="${MIN_FREQUENCY:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-./tokenizer}"

# ── Preflight checks ────────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Aurelius Tokenizer Training                                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Dataset:       ${DATASET}"
echo "  Split:         ${SPLIT}"
echo "  Text field:    ${TEXT_FIELD}"
echo "  Max samples:   ${MAX_SAMPLES}"
echo "  Vocab size:    ${VOCAB_SIZE}"
echo "  Min frequency: ${MIN_FREQUENCY}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo ""

# Ensure required Python packages are available.
python3 -c "import tokenizers; import datasets" 2>/dev/null || {
    echo "ERROR: Required packages missing.  Install with:"
    echo "  pip install tokenizers datasets"
    exit 1
}

# ── Train ────────────────────────────────────────────────────────────────────

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting tokenizer training..."

python3 -m src.data.tokenizer \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --text-field "${TEXT_FIELD}" \
    --max-samples "${MAX_SAMPLES}" \
    --vocab-size "${VOCAB_SIZE}" \
    --min-frequency "${MIN_FREQUENCY}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training complete."
echo "  Tokenizer saved to: ${OUTPUT_DIR}"

# ── Post-training: copy config ───────────────────────────────────────────────

if [ -f configs/tokenizer_config.json ]; then
    cp configs/tokenizer_config.json "${OUTPUT_DIR}/tokenizer_config.json"
    echo "  Copied tokenizer_config.json to ${OUTPUT_DIR}/"
fi

# ── Quick validation ─────────────────────────────────────────────────────────

echo ""
echo "Running validation..."
python3 -c "
from src.data.tokenizer import AureliusTokenizer

tok = AureliusTokenizer.load('${OUTPUT_DIR}')
print(f'  Vocab size: {tok.vocab_size}')
print(f'  BOS id:     {tok.bos_id}')
print(f'  EOS id:     {tok.eos_id}')
print(f'  PAD id:     {tok.pad_id}')

# Round-trip test
samples = [
    'Hello, world!',
    'def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)',
    'The quick brown fox jumps over the lazy dog.',
    '    four_spaces = True',
]
for s in samples:
    ids = tok.encode(s, add_bos=True, add_eos=True)
    rt = tok.decode(ids, skip_special_tokens=True)
    status = 'OK' if rt.strip() == s.strip() else 'MISMATCH'
    print(f'  [{status}] {len(ids):4d} tokens | {s[:60]!r}')
"

echo ""
echo "Done."
