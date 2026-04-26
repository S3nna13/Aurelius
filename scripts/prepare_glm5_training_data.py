#!/usr/bin/env python3
"""
Prepare GLM-5 paper + optional Reddit corpus as tokenized .npy shards for Aurelius training.
"""

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# Paths
PAPER_TXT = Path("/Users/christienantonio/Desktop/Reference Models/GLM-5.txt")
TOKENIZER_PATH = Path("/Users/christienantonio/Desktop/Reference Models/glm5_tokenizer.json")
OUTPUT_DIR = Path("data/pretrain/glm5")
SEQ_LEN = 512
VAL_RATIO = 0.1


def main():
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

    # Check for overflow (uint16 max is 65535, vocab is ~4418 so safe)
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
        f"Num train examples (seq_len={SEQ_LEN}): {max(0, (len(train_ids) - SEQ_LEN - 1) // SEQ_LEN + 1)}"  # noqa: E501
    )
    print(f"Num val examples:   {max(0, (len(val_ids) - SEQ_LEN - 1) // SEQ_LEN + 1)}")
    print("Done.")


if __name__ == "__main__":
    main()
