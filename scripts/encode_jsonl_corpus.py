#!/usr/bin/env python3
"""Encode jsonl_corpus.txt into train/val .npy shards using existing tokenizer."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode JSONL conversations to token ID shards")
    parser.add_argument("--jsonl", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Tokenizer JSON path")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/pretrain/jsonl"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    JSONL_PATH = args.jsonl.expanduser().resolve()
    TOKENIZER_PATH = args.tokenizer.expanduser().resolve()
    OUT_DIR = args.out_dir.expanduser().resolve()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # Read and shuffle conversation lines
    print("Reading JSONL conversations...")
    conversations = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            if not messages:
                continue
            parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    parts.append(f"### {role.capitalize()}\n{content}")
            if parts:
                conversations.append("\n\n".join(parts) + "\n\n")

    print(f"Total conversations: {len(conversations):,}")
    random.shuffle(conversations)

    split = int(0.9 * len(conversations))
    train_convs = conversations[:split]
    val_convs = conversations[split:]

    print(f"Train: {len(train_convs):,}  Val: {len(val_convs):,}")

    # Encode in batches to show progress
    def encode_split(convs, name):
        all_ids = []
        for i, conv in enumerate(convs):
            if (i + 1) % 1000 == 0:
                print(f"  {name}: {i + 1}/{len(convs)}")
            ids = tokenizer.encode(conv).ids
            all_ids.extend(ids)
        return np.array(all_ids, dtype=np.uint16)

    print("Encoding train split...")
    train_arr = encode_split(train_convs, "train")
    print("Encoding val split...")
    val_arr = encode_split(val_convs, "val")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "train.npy", train_arr)
    np.save(OUT_DIR / "val.npy", val_arr)

    print(f"Saved train.npy ({len(train_arr):,} tokens)  val.npy ({len(val_arr):,} tokens)")


if __name__ == "__main__":
    main()
