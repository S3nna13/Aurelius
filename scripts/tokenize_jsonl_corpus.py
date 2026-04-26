#!/usr/bin/env python3
"""Train BPE tokenizer on jsonl_corpus.txt and shard into train/val .npy files."""

import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

CORPUS = Path("/Users/christienantonio/Desktop/Aurelius/data/reference_corpus/jsonl_corpus.txt")
OUT_DIR = Path("/Users/christienantonio/Desktop/Aurelius/data/pretrain/jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_SIZE = 8192

random.seed(42)

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
np.random.seed(42)
np.random.shuffle(all_ids)  # shuffle token-level for maximum randomness

# Actually, better to shuffle at conversation level. Let's re-read and encode per-conversation.
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


def encode_conversations(convs):
    all_ids = []
    for conv in convs:
        ids = tokenizer.encode(conv).ids
        all_ids.extend(ids)
    return np.array(all_ids, dtype=np.uint16)


train_ids = encode_conversations(train_convs)
val_ids = encode_conversations(val_convs)

print(f"Train conversations: {len(train_convs):,}  ->  {len(train_ids):,} tokens")
print(f"Val   conversations: {len(val_convs):,}  ->  {len(val_ids):,} tokens")

# ── 4. Save shards ──────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_DIR / "train_0.npy", train_ids)
np.save(OUT_DIR / "val_0.npy", val_ids)
print(f"Saved -> {OUT_DIR}")
