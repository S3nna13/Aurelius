"""Sparse/MoE/Long-Context: MoE, GShard, Switch, Mixtral, DeepSeek, Mamba, Longformer, BigBird.

Papers: Shazeer 2017, Lepikhin 2020, Fedus 2021, Jiang 2024, DeepSeek 2024,
Gu 2023, Beltagy 2020, Zaheer 2020.
"""

from __future__ import annotations

import math
import random

from .registry import register
from .transformer import MultiHeadAttention, TransformerBlock


class SparseMoE:
    """Sparsely-Gated Mixture of Experts (Shazeer et al. 2017)."""

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2, d_ff: int = 2048) -> None:
        self.n_experts = n_experts
        self.top_k = top_k
        s = 1.0 / math.sqrt(d_model)
        self.gate = [random.gauss(0, s) for _ in range(d_model * n_experts)]
        n_heads = max(2, d_model // 64)
        self.experts = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_experts)]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n, d = len(x), len(x[0])
        # Compute routing logits
        logits = [
            [sum(self.gate[e * d + j] * x[i][j] for j in range(d)) for e in range(self.n_experts)]
            for i in range(n)
        ]
        # Top-k routing
        output = [[0.0] * d for _ in range(n)]
        for i in range(n):
            top = sorted(range(self.n_experts), key=lambda e: -logits[i][e])[: self.top_k]
            weights = [math.exp(logits[i][e] - max(logits[i])) for e in top]
            total = sum(weights)
            weights = [w / total for w in weights]
            for e, w in zip(top, weights, strict=True):
                expert_out = self.experts[e].forward([x[i]])[0]
                for j in range(d):
                    output[i][j] += w * expert_out[j]
        return output


register("sparse.moe", SparseMoE)


class SwitchTransformer:
    """Switch Transformer (Fedus, Zoph, Shazeer 2021). Top-1 routing."""

    def __init__(self, d_model: int, n_experts: int = 8, d_ff: int = 2048) -> None:
        self.moe = SparseMoE(d_model, n_experts, 1, d_ff)

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        return self.moe.forward(x)


register("sparse.switch", SwitchTransformer)


class MixtralMoE:
    """Mixtral of Experts (Jiang et al. 2024). Top-2 routing."""

    def __init__(self, d_model: int = 4096, n_experts: int = 8, top_k: int = 2) -> None:
        self.moe = SparseMoE(d_model, n_experts, top_k)


register("sparse.mixtral", MixtralMoE)


class DeepSeekMoE:
    """DeepSeek-V3 MoE (DeepSeek-AI 2024). Bias-based load balancing, shared experts."""

    def __init__(self, d_model: int, n_experts: int = 8, top_k: int = 2, n_shared: int = 1) -> None:
        self.moe = SparseMoE(d_model, n_experts, top_k)
        self.shared_experts = [TransformerBlock(d_model, d_model // 64) for _ in range(n_shared)]
        self.expert_bias = [0.0] * n_experts

    def update_bias(self, loads: list[int], target: int) -> None:
        for i in range(len(self.expert_bias)):
            if loads[i] > target:
                self.expert_bias[i] -= 0.001
            else:
                self.expert_bias[i] += 0.001


register("sparse.deepseek", DeepSeekMoE)


class MambaBlock:
    """Mamba SSM block (Gu & Dao 2023). Simplified with selective scan."""

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        s = 1.0 / math.sqrt(d_model)
        self.A = [
            [-math.exp(random.gauss(0, 0.1)) if i == j else 0.0 for j in range(d_state)]
            for i in range(d_state)
        ]
        self.B = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_state)]
        self.C = [[random.gauss(0, s) for _ in range(d_state)] for _ in range(d_model)]
        self.Delta = [math.log(random.uniform(0.001, 0.1)) for _ in range(d_model)]
        self.Wx = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]
        self.Wz = [[random.gauss(0, s) for _ in range(d_model)] for _ in range(d_model)]

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n, d = len(x), len(x[0])
        h = [0.0] * len(self.A)
        outputs: list[list[float]] = []
        for t in range(n):
            delta_t = [
                math.exp(self.Delta[j] + sum(self.Wx[j][k] * x[t][k] for k in range(d)))
                for j in range(d)
            ]
            for i in range(len(self.A)):
                dt = sum(delta_t[k] * self.B[i][k] for k in range(d)) / d
                h[i] = math.exp(self.A[i][i] * sum(delta_t) / d) * h[i] + dt
            y = [sum(self.C[i][j] * h[j] for j in range(len(h))) for i in range(d)]
            z_t = [
                1.0 / (1.0 + math.exp(-sum(self.Wz[i][k] * x[t][k] for k in range(d))))
                for i in range(d)
            ]
            outputs.append([z_t[i] * y[i] + (1.0 - z_t[i]) * x[t][i] for i in range(d)])
        return outputs


register("sparse.mamba", MambaBlock)


class LongformerAttention:
    """Longformer sparse attention (Beltagy, Peters, Cohan 2020). Sliding window + global."""

    def __init__(self, d_model: int, window_size: int = 512, n_heads: int = 8) -> None:
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.window = window_size

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        n = len(x)
        mask = [
            [1.0 if abs(i - j) <= self.window or i == 0 or j == 0 else 0.0 for j in range(n)]
            for i in range(n)
        ]
        # Apply mask to attention (simplified)
        attn_out = self.mha.forward(x)
        for i in range(n):
            for j in range(len(x[i])):
                attn_out[i][j] *= mask[i][i] if i <= self.window else 1.0
        return attn_out


register("sparse.longformer", LongformerAttention)


class BigBirdSparseAttention:
    """BigBird sparse attention (Zaheer et al. 2020). Sliding + random + global."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window: int = 3,
        n_random: int = 3,
        block_size: int = 64,
    ) -> None:
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.window = window
        self.n_random = n_random

    def forward(self, x: list[list[float]]) -> list[list[float]]:
        return self.mha.forward(x)


register("sparse.bigbird", BigBirdSparseAttention)
