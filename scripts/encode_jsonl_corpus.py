#!/usr/bin/env python3
"""Encode jsonl_corpus.txt into train/val .npy shards using existing tokenizer."""

import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

JSONL_PATH = Path("/Users/christienantonio/Desktop/Aurelius_Training_Data.jsonl")
TOKENIZER_PATH = Path("/Users/christienantonio/Desktop/Aurelius/data/pretrain/jsonl/tokenizer.json")
OUT_DIR = Path("/Users/christienantonio/Desktop/Aurelius/data/pretrain/jsonl")

random.seed(42)
np.random.seed(42)

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


print("Encoding train...")
train_ids = encode_split(train_convs, "train")
print("Encoding val...")
val_ids = encode_split(val_convs, "val")

print(f"Train tokens: {len(train_ids):,}")
print(f"Val   tokens: {len(val_ids):,}")

np.save(OUT_DIR / "train_0.npy", train_ids)
np.save(OUT_DIR / "val_0.npy", val_ids)
print(f"Saved -> {OUT_DIR}")
