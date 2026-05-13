#!/usr/bin/env python3
"""Train BPE tokenizer on jsonl_corpus.txt and shard into train/val .npy files."""

import argparse
import random
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer and encode corpus")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to jsonl_corpus.txt")
    parser.add_argument("--out-dir", type=Path, default=Path("data/pretrain/jsonl"), help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Tokenizer vocabulary size")
    args = parser.parse_args()

    CORPUS = args.corpus.expanduser().resolve()
    OUT_DIR = args.out_dir.expanduser().resolve()
    VOCAB_SIZE = args.vocab_size

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    # ── 1. Train tokenizer ──────────────────────────────────────────────────────
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))  # noqa: S106
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
    )
    tokenizer.train([str(CORPUS)], trainer)
    tokenizer.save(str(OUT_DIR / "tokenizer.json"))
    print(f"Tokenizer saved -> {OUT_DIR / 'tokenizer.json'}  (vocab={VOCAB_SIZE})")

    # ── 2. Encode full corpus ───────────────────────────────────────────────────
    print("Encoding corpus...")
    with open(CORPUS, encoding="utf-8") as f:
        text = f.read()
    encoding = tokenizer.encode(text)
    all_ids = np.array(encoding.ids, dtype=np.uint16)
    print(f"Total tokens: {len(all_ids):,}")

    # ── 3. Shuffle & split ──────────────────────────────────────────────────────
    # Re-encode per-conversation for proper train/val split
    print("Re-encoding per-conversation for proper train/val split...")
    conversations = []
    with open(CORPUS, encoding="utf-8") as f:
        current = []
        for line in f:
            if line.strip() == "---":
                if current:
                    conversations.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            conversations.append("\n".join(current))

    random.shuffle(conversations)
    split = int(0.9 * len(conversations))
    train_convs = conversations[:split]
    val_convs = conversations[split:]

    # Encode each conversation and concatenate
    train_ids_list = []
    for conv in train_convs:
        ids = tokenizer.encode(conv).ids
        train_ids_list.extend(ids)
    val_ids_list = []
    for conv in val_convs:
        ids = tokenizer.encode(conv).ids
        val_ids_list.extend(ids)

    train_arr = np.array(train_ids_list, dtype=np.uint16)
    val_arr = np.array(val_ids_list, dtype=np.uint16)

    np.save(OUT_DIR / "train.npy", train_arr)
    np.save(OUT_DIR / "val.npy", val_arr)

    print(f"Train tokens: {len(train_arr):,} | Val tokens: {len(val_arr):,}")
    print(f"Saved to {OUT_DIR}/{{train.npy, val.npy}}")


if __name__ == "__main__":
    main()
