"""Recommender Systems: Wide&Deep, DeepFM, NCF, BERT4Rec, SASRec.

Papers: Cheng 2016, Guo 2017, He 2017, Sun 2019, Kang 2018.
"""

from __future__ import annotations

import math
import random

from .foundational import MLP
from .registry import register


class WideAndDeep:
    """Wide & Deep (Cheng et al. 2016). Memorization + generalization."""

    def __init__(self, n_features: int = 100, d_embed: int = 32) -> None:
        self.wide = [random.gauss(0, 0.1) for _ in range(n_features)]
        self.wide_bias = 0.0
        self.deep = MLP([d_embed, 64, 1])

    def forward(self, sparse_features: list[int], dense_features: list[float]) -> float:
        wide_out = sum(self.wide[f] for f in sparse_features) + self.wide_bias
        deep_out = self.deep.forward(dense_features)[0]
        return 1.0 / (1.0 + math.exp(-(wide_out + deep_out)))


register("rec.wide_deep", WideAndDeep)


class NeuralCF:
    """Neural Collaborative Filtering (He et al. 2017)."""

    def __init__(self, n_users: int = 1000, n_items: int = 1000, d_embed: int = 64) -> None:
        s = 1.0 / math.sqrt(d_embed)
        self.user_embeds = [[random.gauss(0, s) for _ in range(d_embed)] for _ in range(n_users)]
        self.item_embeds = [[random.gauss(0, s) for _ in range(d_embed)] for _ in range(n_items)]
        self.mlp = MLP([d_embed * 2, 64, 32, 1])
        self.gmf = [random.gauss(0, s) for _ in range(d_embed)]

    def predict(self, user_id: int, item_id: int) -> float:
        ue = self.user_embeds[user_id]
        ie = self.item_embeds[item_id]
        gmf_out = sum(ue[i] * ie[i] * self.gmf[i] for i in range(len(ue)))
        mlp_out = self.mlp.forward(ue + ie)[0]
        return 1.0 / (1.0 + math.exp(-(gmf_out + mlp_out)))


register("rec.neural_cf", NeuralCF)


class SASRec:
    """Self-Attentive Sequential Recommendation (Kang & McAuley 2018)."""

    def __init__(self, n_items: int = 1000, d_model: int = 64, n_heads: int = 2) -> None:
        from .transformer import MultiHeadAttention

        self.item_embeds = [[random.gauss(0, 0.1) for _ in range(d_model)] for _ in range(n_items)]
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.output = [random.gauss(0, 0.1) for _ in range(d_model)]

    def predict_next(self, seq: list[int]) -> list[float]:
        x = [list(self.item_embeds[i]) for i in seq]
        x = self.attn.forward(x)
        scores = [
            sum(self.output[k] * x[-1][k] for k in range(len(x[-1])))
            for _ in range(len(self.item_embeds))
        ]
        return scores


register("rec.sasrec", SASRec)
