# ruff: noqa: I001
#!/usr/bin/env python3
"""

# isort: skip_file

Prepare GLM-5 paper + optional Reddit corpus as tokenized .npy shards for Aurelius training.
"""

import argparse
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GLM-5 training data")
    parser.add_argument("--paper-txt", type=Path, required=True, help="Path to the raw .txt paper")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Path to tokenizer JSON")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/pretrain/glm5"), help="Output directory"
    )
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    # Resolve paths
    PAPER_TXT = args.paper_txt.expanduser().resolve()
    TOKENIZER_PATH = args.tokenizer_path.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    SEQ_LEN = args.seq_len
    VAL_RATIO = args.val_ratio

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    # Load and tokenize paper
    with open(PAPER_TXT, encoding="utf-8") as f:
        text = f.read()

    print(f"Paper length: {len(text)} chars")

    encoded = tokenizer.encode(text)
    token_ids = np.array(encoded.ids, dtype=np.uint16)
    print(f"Tokenized length: {len(token_ids)} tokens")

    # Check for overflow (uint16 max is 65535, vocab is ~8192 so safe)
    assert token_ids.max() < 65535, "Token IDs exceed uint16 range"

    # Split train/val
    n_val = int(len(token_ids) * VAL_RATIO)
    train_ids = token_ids[n_val:]
    val_ids = token_ids[:n_val]

    # Save as shards (single shard each for this small corpus)
    train_path = OUTPUT_DIR / "train_shard_000.npy"
    val_path = OUTPUT_DIR / "val_shard_000.npy"
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)

    print(f"Train shard: {train_path} ({len(train_ids):,} tokens)")
    print(f"Val shard:   {val_path} ({len(val_ids):,} tokens)")
    print(
        f"Num train examples (seq_len={SEQ_LEN}): {max(0, (len(train_ids) - SEQ_LEN - 1) // SEQ_LEN + 1)}"
    )
    print(f"Num val examples:   {max(0, (len(val_ids) - SEQ_LEN - 1) // SEQ_LEN + 1)}")
    print("Done.")


if __name__ == "__main__":
    main()
