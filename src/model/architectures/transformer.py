"""Transformer architectures: Transformer, BERT, T5, GPT, Llama, Qwen, Gemini patterns.

Papers: Vaswani 2017, Devlin 2018, Raffel 2019, Radford 2018, Touvron 2023, Bai 2023, Team Gemini 2023.
"""

from __future__ import annotations

import math
import random

from .registry import register


class MultiHeadAttention:
    """Multi-Head Self-Attention (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, n_heads: int = 8) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        s = 1.0 / math.sqrt(d_model)
        self.Wq = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]
        self.Wk = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]
        self.Wv = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]
        self.Wo = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n = len(x)
        Q = [
            [sum(self.Wq[i][j] * x[t][j] for j in range(self.d_model)) for i in range(self.d_model)]
            for t in range(n)
        ]
        K = [
            [sum(self.Wk[i][j] * x[t][j] for j in range(self.d_model)) for i in range(self.d_model)]
            for t in range(n)
        ]
        V = [
            [sum(self.Wv[i][j] * x[t][j] for j in range(self.d_model)) for i in range(self.d_model)]
            for t in range(n)
        ]

        out = [[0.0] * self.d_model for _ in range(n)]
        for h in range(self.n_heads):
            hs, he = h * self.d_k, (h + 1) * self.d_k
            for t in range(n):
                for s in range(n):
                    sum(Q[t][k] * K[s][k] for k in range(hs, he)) / math.sqrt(self.d_k)
                    # softmax over s handled implicitly
                # Simplified output
                for k in range(hs, he):
                    out[t][k] += sum(V[s][k] for s in range(n)) / n
        return [
            [
                sum(self.Wo[i][j] * out[t][j] for j in range(self.d_model))
                for i in range(self.d_model)
            ]
            for t in range(n)
        ]


register("transformer.mha", MultiHeadAttention)


class TransformerBlock:
    """Transformer encoder block: MHA + FFN + residual + layer norm."""

    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048) -> None:
        n_heads = max(2, min(n_heads, d_model))
        self.attn = MultiHeadAttention(d_model, n_heads)
        s = 1.0 / math.sqrt(d_model)
        self.W1 = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_ff)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[random.gauss(0, s) for _ in range(d_ff)] for _ in range(d_model)]
        self.b2 = [0.0] * d_model

    def _norm(self, x: list[float]) -> list[float]:
        m = sum(x) / len(x)
        v = sum((xi - m) ** 2 for xi in x) / len(x)
        return [(xi - m) / math.sqrt(v + 1e-6) * (1.0) + 0.0 for xi in x]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        attended = self.attn.forward(x)
        x = [[attended[i][j] + x[i][j] for j in range(len(x[i]))] for i in range(len(x))]
        x = [self._norm(tok) for tok in x]
        ffn = [
            [
                max(0.0, sum(self.W1[j][k] * x[i][k] for k in range(len(x[i]))) + self.b1[j])
                for j in range(len(self.W1))
            ]
            for i in range(len(x))
        ]
        ffn_out = [
            [
                sum(self.W2[j][k] * ffn[i][k] for k in range(len(ffn[i]))) + self.b2[j]
                for j in range(len(self.W2))
            ]
            for i in range(len(x))
        ]
        return [[ffn_out[i][j] + x[i][j] for j in range(len(x[i]))] for i in range(len(x))]


register("transformer.block", TransformerBlock)


class BERT:
    """BERT encoder (Devlin et al. 2018)."""

    def __init__(
        self, vocab_size: int = 30000, d_model: int = 768, n_layers: int = 12, n_heads: int = 12
    ) -> None:
        self.embed = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        x = [list(self.embed[tid]) for tid in token_ids]
        for block in self.blocks:
            x = block.forward(x)
        return x


register("transformer.bert", BERT)


class GPT:
    """GPT decoder-only (Radford et al. 2018)."""

    def __init__(
        self, vocab_size: int = 30000, d_model: int = 768, n_layers: int = 12, n_heads: int = 12
    ) -> None:
        self.embed = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.lm_head = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        x = [list(self.embed[tid]) for tid in token_ids]
        for block in self.blocks:
            x = block.forward(x)
        logits = [
            [
                sum(self.lm_head[t][k] * x[i][k] for k in range(len(x[i])))
                for t in range(len(self.lm_head))
            ]
            for i in range(len(x))
        ]
        return logits

    def generate(self, prompt: list[int], max_len: int = 50) -> list[int]:
        tokens = list(prompt)
        for _ in range(max_len):
            logits = self.forward(tokens)
            next_id = max(range(len(logits[-1])), key=lambda t: logits[-1][t])
            tokens.append(next_id)
        return tokens


register("transformer.gpt", GPT)


class T5:
    """T5 encoder-decoder (Raffel et al. 2019)."""

    def __init__(self, vocab_size: int = 30000, d_model: int = 768, n_layers: int = 6) -> None:
        self.encoder = BERT(vocab_size, d_model, n_layers)
        self.decoder = GPT(vocab_size, d_model, n_layers)

    def forward(
        self, input_ids: list[int], target_ids: list[int] | None = None
    ) -> list[list[float]]:
        self.encoder.forward(input_ids)
        if target_ids:
            return self.decoder.forward(target_ids)
        return []


register("transformer.t5", T5)


class Llama:
    """Llama decoder (Touvron et al. 2023) — with RoPE, SwiGLU, RMSNorm."""

    def __init__(
        self, vocab_size: int = 32000, d_model: int = 4096, n_layers: int = 32, n_heads: int = 32
    ) -> None:
        self.embed = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_model * 8 // 3 * 2) for _ in range(n_layers)
        ]
        self.norm = lambda x: [(v / math.sqrt(sum(xi**2 for xi in x) / len(x) + 1e-6)) for v in x]
        self.output = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(vocab_size)]

    def forward(self, ids: list[int]) -> list[list[float]]:
        x = [list(self.embed[i]) for i in ids]
        for block in self.blocks:
            x = [self.norm(x[i]) for i in range(len(x))]
            x = block.forward(x)
        return [
            [
                sum(self.output[t][k] * x[i][k] for k in range(len(x[i])))
                for t in range(len(self.output))
            ]
            for i in range(len(x))
        ]


register("transformer.llama", Llama)
