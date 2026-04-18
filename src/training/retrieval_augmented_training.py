"""Retrieval-Augmented Training (RAT) for language models.

Implements a vector memory bank, cross-attention retriever, RAT layer,
a self-contained RATModel (embedding + TransformerBlocks + RATLayers + lm_head),
and a RATTrainer that encodes a prefix into the memory bank before computing
LM loss on the suffix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RATConfig:
    """Configuration for RAT model and trainer."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_heads: int = 2
    d_mem: int = 32
    k_retrieved: int = 3
    capacity: int = 128


# ---------------------------------------------------------------------------
# VectorMemoryBank
# ---------------------------------------------------------------------------


class VectorMemoryBank:
    """Fixed-capacity circular buffer of (key, value) vector pairs.

    Keys are used for cosine-similarity search; values are the content
    that the retriever blends back into the model.
    """

    def __init__(self, capacity: int, d_key: int, d_val: int) -> None:
        self.capacity = capacity
        self.d_key = d_key
        self.d_val = d_val

        self.keys: Tensor = torch.zeros(capacity, d_key)
        self.values: Tensor = torch.zeros(capacity, d_val)
        self.size: int = 0           # number of valid entries  (≤ capacity)
        self._write_ptr: int = 0     # next write position in the circular buffer

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, keys: Tensor, values: Tensor) -> None:
        """Insert key–value pairs into the circular buffer.

        Args:
            keys:   (N, d_key)
            values: (N, d_val)
        """
        n = keys.size(0)
        if n == 0:
            return

        device = self.keys.device
        keys = keys.detach().to(device)
        values = values.detach().to(device)

        # Write in chunks that may wrap around the end of the buffer.
        written = 0
        while written < n:
            space_until_end = self.capacity - self._write_ptr
            chunk = min(n - written, space_until_end)
            self.keys[self._write_ptr: self._write_ptr + chunk] = keys[written: written + chunk]
            self.values[self._write_ptr: self._write_ptr + chunk] = values[written: written + chunk]
            self._write_ptr = (self._write_ptr + chunk) % self.capacity
            written += chunk

        self.size = min(self.size + n, self.capacity)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def query(self, q: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Top-k cosine similarity search against stored keys.

        Args:
            q: (B, d_key) query vectors.
            k: number of neighbours to retrieve.

        Returns:
            indices: (B, k) int64 — indices into the bank (valid slice).
            scores:  (B, k) float — cosine similarities, descending.
        """
        if self.size == 0:
            B = q.size(0)
            k_eff = min(k, self.capacity)
            indices = torch.zeros(B, k_eff, dtype=torch.long, device=q.device)
            scores = torch.zeros(B, k_eff, device=q.device)
            return indices, scores

        # Work only on the valid portion of the buffer.
        k_eff = min(k, self.size)
        valid_keys = self.keys[: self.size].to(q.device)          # (size, d_key)
        q_norm = F.normalize(q, dim=-1)                            # (B, d_key)
        k_norm = F.normalize(valid_keys, dim=-1)                   # (size, d_key)
        sims = q_norm @ k_norm.T                                   # (B, size)
        scores, indices = sims.topk(k_eff, dim=-1, sorted=True)   # (B, k_eff)
        return indices, scores

    def retrieve(self, q: Tensor, k: int) -> Tensor:
        """Return the top-k values for each query.

        Args:
            q: (B, d_key)
            k: number of neighbours.

        Returns:
            (B, k, d_val) — zero tensor when the bank is empty.
        """
        B = q.size(0)
        device = q.device

        # Graceful fallback when the bank has no entries yet.
        if self.size == 0:
            k_eff = min(k, self.capacity)
            return torch.zeros(B, k_eff, self.d_val, device=device)

        indices, _ = self.query(q, k)                       # (B, k_eff)
        _, k_eff = indices.shape
        valid_values = self.values[: self.size].to(device)  # (size, d_val)

        # Gather: flat index into valid_values
        flat_idx = indices.reshape(-1)                      # (B*k_eff,)
        retrieved = valid_values[flat_idx]                  # (B*k_eff, d_val)
        return retrieved.reshape(B, k_eff, self.d_val)      # (B, k_eff, d_val)


# ---------------------------------------------------------------------------
# CrossAttentionRetriever
# ---------------------------------------------------------------------------


class CrossAttentionRetriever(nn.Module):
    """Projects retrieved memory values into model space via multi-head cross-attention.

    The *query* comes from the model's hidden states; the *keys* and *values*
    are derived from the retrieved memory documents.
    """

    def __init__(self, d_model: int, d_mem: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Projections for query (from model), key/value (from memory)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_mem, d_model, bias=False)
        self.v_proj = nn.Linear(d_mem, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self._scale = math.sqrt(self.head_dim)

    def forward(self, query: Tensor, retrieved: Tensor) -> Tensor:
        """Cross-attend from query tokens over retrieved memory values.

        Args:
            query:     (B, T, d_model) — model hidden states.
            retrieved: (B, k, d_mem)  — retrieved memory values.

        Returns:
            (B, T, d_model) — context-enriched representations.
        """
        B, T, _ = query.shape
        k = retrieved.size(1)
        H = self.n_heads
        hd = self.head_dim

        Q = self.q_proj(query)        # (B, T, d_model)
        K = self.k_proj(retrieved)    # (B, k, d_model)
        V = self.v_proj(retrieved)    # (B, k, d_model)

        # Reshape to (B, H, T, hd) / (B, H, k, hd)
        Q = Q.reshape(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        K = K.reshape(B, k, H, hd).transpose(1, 2)   # (B, H, k, hd)
        V = V.reshape(B, k, H, hd).transpose(1, 2)   # (B, H, k, hd)

        attn_weights = (Q @ K.transpose(-2, -1)) / self._scale  # (B, H, T, k)
        attn_weights = F.softmax(attn_weights, dim=-1)

        out = attn_weights @ V                                   # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)   # (B, T, d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# RATLayer
# ---------------------------------------------------------------------------


class RATLayer(nn.Module):
    """Retrieval-Augmented Training layer.

    Queries the memory bank with a mean-pooled representation of the current
    hidden states, retrieves k documents, cross-attends over them, and blends
    the result back into the hidden states via a learnable scalar gate.
    """

    def __init__(self, d_model: int, d_mem: int, n_heads: int, k_retrieved: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.k_retrieved = k_retrieved

        self.retriever = CrossAttentionRetriever(d_model, d_mem, n_heads)
        # Learnable scalar gate — stored as an unconstrained parameter; sigmoid gives [0,1]
        self.gate_logit = nn.Parameter(torch.zeros(1))

    @property
    def gate(self) -> Tensor:
        """Sigmoid-gated blending scalar in [0, 1]."""
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: Tensor, memory_bank: VectorMemoryBank) -> Tensor:
        """Augment hidden states with retrieved memory.

        Args:
            x:           (B, T, d_model)
            memory_bank: VectorMemoryBank to query.

        Returns:
            (B, T, d_model) — augmented hidden states.
        """
        # Query key: mean-pool over sequence
        q = x.mean(dim=1)                           # (B, d_model)

        # Retrieve k documents from the memory bank
        retrieved = memory_bank.retrieve(q, self.k_retrieved)   # (B, k, d_mem)
        retrieved = retrieved.to(x.device)

        # Cross-attend
        ctx = self.retriever(x, retrieved)           # (B, T, d_model)

        # Blend via sigmoid gate
        return x + self.gate * ctx


# ---------------------------------------------------------------------------
# Inline TransformerBlock (no external deps)
# ---------------------------------------------------------------------------


class _TransformerBlock(nn.Module):
    """Minimal pre-norm Transformer block: self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, 4 * d_model, bias=True)
        self.ff2 = nn.Linear(4 * d_model, d_model, bias=True)

        self._scale = math.sqrt(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:  # (B, T, d_model)
        B, T, _ = x.shape
        H = self.n_heads
        hd = self.head_dim

        # --- Self-attention ---
        h = self.norm1(x)
        qkv = self.qkv(h)                             # (B, T, 3*d_model)
        Q, K, V = qkv.split(self.d_model, dim=-1)
        Q = Q.reshape(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        K = K.reshape(B, T, H, hd).transpose(1, 2)
        V = V.reshape(B, T, H, hd).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / self._scale  # (B, H, T, T)
        # causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ V                                # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        out = self.o_proj(out)
        x = x + out

        # --- FFN ---
        h = self.norm2(x)
        h = F.gelu(self.ff1(h))
        h = self.ff2(h)
        return x + h


# ---------------------------------------------------------------------------
# RATModel
# ---------------------------------------------------------------------------


class RATModel(nn.Module):
    """Language model with interleaved Transformer + RAT layers.

    Architecture (per layer pair):
        TransformerBlock → RATLayer

    A shared VectorMemoryBank is updated by encode_to_memory() and queried
    by each RATLayer during forward().
    """

    def __init__(self, config: RATConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(512, config.d_model)  # learnable positional

        self.transformer_layers = nn.ModuleList(
            [_TransformerBlock(config.d_model, config.n_heads) for _ in range(config.n_layers)]
        )
        self.rat_layers = nn.ModuleList(
            [
                RATLayer(config.d_model, config.d_mem, config.n_heads, config.k_retrieved)
                for _ in range(config.n_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Shared memory bank (not an nn.Module — lives as a plain attribute)
        self.memory_bank = VectorMemoryBank(config.capacity, config.d_model, config.d_mem)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    # ------------------------------------------------------------------

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute LM logits.

        Args:
            input_ids: (B, T) int64 token ids.

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)
        x = self.embedding(input_ids) + self.pos_embedding(positions)       # (B, T, d_model)

        for tf_block, rat_layer in zip(self.transformer_layers, self.rat_layers):
            x = tf_block(x)
            x = rat_layer(x, self.memory_bank)

        x = self.norm(x)
        return self.lm_head(x)   # (B, T, vocab_size)

    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_to_memory(self, input_ids: Tensor) -> None:
        """Encode token sequences and store their representations in the memory bank.

        Uses the mean-pooled hidden states after the last TransformerBlock
        as both keys and values (keys and values share the same space here
        since d_model == d_mem in the default config).

        Args:
            input_ids: (B, T) int64 token ids.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        for tf_block in self.transformer_layers:
            x = tf_block(x)

        # Mean-pool over the sequence dimension → (B, d_model)
        pooled = x.mean(dim=1)   # (B, d_model)

        # Store pooled vector as both key and value
        self.memory_bank.add(pooled, pooled)


# ---------------------------------------------------------------------------
# RATTrainer
# ---------------------------------------------------------------------------


class RATTrainer:
    """Trainer for RATModel.

    Each training step:
    1. Splits input_ids into prefix (first half) and suffix (second half).
    2. Encodes the prefix into the memory bank.
    3. Computes cross-entropy LM loss on the suffix using the full model forward pass.
    """

    def __init__(self, model: RATModel, lr: float = 1e-3, k_retrieved: int = 3) -> None:
        self.model = model
        self.k_retrieved = k_retrieved
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------

    def train_step(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Run one optimisation step.

        Encodes the first half of input_ids as prefix memory, then computes
        LM loss on input_ids shifted against labels.

        Args:
            input_ids: (B, T) int64.
            labels:    (B, T) int64 — same shape; positions set to -100 are ignored.

        Returns:
            Scalar loss tensor (detached value accessible via .item()).
        """
        self.model.train()
        self.optimizer.zero_grad()

        T = input_ids.size(1)
        prefix_len = max(1, T // 2)
        prefix = input_ids[:, :prefix_len]

        # Encode prefix into memory (no_grad inside encode_to_memory)
        self.model.encode_to_memory(prefix)

        # Full forward pass
        logits = self.model(input_ids)   # (B, T, vocab_size)

        # Cross-entropy loss (flat)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        loss.backward()
        self.optimizer.step()

        return loss

    # ------------------------------------------------------------------

    def populate_memory(self, corpus: list[Tensor]) -> None:
        """Encode all tensors in *corpus* and add them to the memory bank.

        Args:
            corpus: List of (B, T) or (T,) integer token tensors.
        """
        for item in corpus:
            if item.dim() == 1:
                item = item.unsqueeze(0)
            self.model.encode_to_memory(item)
