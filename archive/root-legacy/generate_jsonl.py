#!/usr/bin/env python3
"""Generate text from the trained Aurelius JSONL model."""

import torch
from tokenizers import Tokenizer

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

CONFIG = AureliusConfig(
    d_model=768,
    n_layers=10,
    n_heads=12,
    n_kv_heads=4,
    head_dim=64,
    d_ff=2048,
    vocab_size=8192,
    max_seq_len=512,
    rope_theta=500_000.0,
    rms_norm_eps=1.0e-6,
    dropout=0.0,
    tie_embeddings=True,
)


def generate(prompt, max_new=128, temperature=0.8, top_k=40):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = Tokenizer.from_file("data/pretrain/jsonl/tokenizer.json")
    model = AureliusTransformer(CONFIG)
    ckpt = torch.load(
        "checkpoints/aurelius-jsonl/final.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new):
            _, logits, _ = model(input_ids)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if input_ids.size(1) >= CONFIG.max_seq_len:
                input_ids = input_ids[:, -CONFIG.max_seq_len + 1 :]

    output = tokenizer.decode(input_ids[0].tolist())
    return output


if __name__ == "__main__":
    prompts = [
        "### User\nWhat is machine learning?\n\n### Assistant\n",
        "### User\nExplain graph theory.\n\n### Assistant\n",
        "### System\nYou are Aurelius.\n### User\nHello!\n\n### Assistant\n",
    ]
    for p in prompts:
        print("=" * 60)
        print("PROMPT:", repr(p))
        print("-" * 60)
        out = generate(p, max_new=100, temperature=0.7)
        print(out)
        print()
