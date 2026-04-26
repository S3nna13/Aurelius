"""
In-Context Learning with Retrieval (ICL Retrieval)

Dense retrieval of relevant demonstrations for in-context learning,
with methods for demonstration selection, ordering, and quality filtering.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ICLRetrievalConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    capacity: int = 100
    k_demos: int = 4
    max_demo_tokens: int = 32
    mmr_lambda: float = 0.5


# ---------------------------------------------------------------------------
# DemonstrationStore
# ---------------------------------------------------------------------------


class DemonstrationStore:
    """Fixed-capacity key-value store for ICL demonstrations."""

    def __init__(self, d_model: int, capacity: int = 1000) -> None:
        self.d_model = d_model
        self.capacity = capacity
        # Pre-allocate key matrix; fill with zeros until populated
        self.keys: torch.Tensor = torch.zeros(capacity, d_model)
        self.values: list[dict] = []
        self.size: int = 0

    # ------------------------------------------------------------------
    def add(self, key_embedding: torch.Tensor, value: dict) -> None:
        """Add a single demonstration.  Wraps around when at capacity."""
        idx = self.size % self.capacity
        self.keys[idx] = key_embedding.detach().float()
        if idx < len(self.values):
            self.values[idx] = value
        else:
            self.values.append(value)
        self.size += 1

    def add_batch(self, embeddings: torch.Tensor, values: list[dict]) -> None:
        """Add a batch of demonstrations."""
        assert embeddings.shape[0] == len(values), "embeddings and values must have the same length"  # noqa: S101
        for emb, val in zip(embeddings, values):
            self.add(emb, val)

    # ------------------------------------------------------------------
    def search(self, query: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine-similarity top-k search.

        Returns
        -------
        scores  : [k]  — cosine similarities
        indices : [k]  — positions in the store
        """
        n = min(self.size, self.capacity)
        if n == 0:
            return torch.empty(0), torch.empty(0, dtype=torch.long)

        k = min(k, n)
        q = query.detach().float()
        stored = self.keys[:n]

        # Normalise for cosine similarity
        q_norm = F.normalize(q.unsqueeze(0), dim=-1)  # [1, d]
        k_norm = F.normalize(stored, dim=-1)  # [n, d]
        sims = (k_norm @ q_norm.T).squeeze(-1)  # [n]

        scores, indices = torch.topk(sims, k)
        return scores, indices

    def retrieve(self, query: torch.Tensor, k: int) -> list[dict]:
        """Return top-k value dicts for a query."""
        _, indices = self.search(query, k)
        return [self.values[i.item()] for i in indices]


# ---------------------------------------------------------------------------
# DemonstrationEncoder
# ---------------------------------------------------------------------------


class DemonstrationEncoder(nn.Module):
    """Lightweight transformer-style encoder that maps token ids → [d_model]."""

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=max(1, d_model // 8),
                    dim_feedforward=d_model * 4,
                    dropout=0.0,
                    batch_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : [B, T]

        Returns
        -------
        embeddings : [B, d_model]   (mean pooling over T)
        """
        x = self.embed(input_ids)  # [B, T, d]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        # Mean pool over the sequence dimension
        embeddings = x.mean(dim=1)  # [B, d]
        return embeddings


# ---------------------------------------------------------------------------
# DemonstrationSelector
# ---------------------------------------------------------------------------


class DemonstrationSelector:
    """Selects demonstrations from a DemonstrationStore for ICL."""

    def __init__(self, store: DemonstrationStore, encoder: DemonstrationEncoder) -> None:
        self.store = store
        self.encoder = encoder

    # ------------------------------------------------------------------
    def select_random(self, k: int) -> list[dict]:
        n = min(self.store.size, self.store.capacity)
        k = min(k, n)
        indices = random.sample(range(n), k)
        return [self.store.values[i] for i in indices]

    # ------------------------------------------------------------------
    def select_by_similarity(self, query_ids: torch.Tensor, k: int) -> list[dict]:
        """Encode query and retrieve top-k by cosine similarity."""
        with torch.no_grad():
            q_emb = self.encoder(query_ids.unsqueeze(0)).squeeze(0)  # [d]
        return self.store.retrieve(q_emb, k)

    # ------------------------------------------------------------------
    def select_diverse(
        self, query_ids: torch.Tensor, k: int, mmr_lambda: float = 0.5
    ) -> list[dict]:
        """Maximal Marginal Relevance (MMR) selection.

        At each step picks the demo that maximises:
            λ * sim(d, query) - (1-λ) * max_{j in selected} sim(d, d_j)
        """
        n = min(self.store.size, self.store.capacity)
        k = min(k, n)

        with torch.no_grad():
            q_emb = self.encoder(query_ids.unsqueeze(0)).squeeze(0)  # [d]

        stored = self.store.keys[:n]  # [n, d]
        q_norm = F.normalize(q_emb.unsqueeze(0), dim=-1)  # [1, d]
        s_norm = F.normalize(stored, dim=-1)  # [n, d]
        relevance = (s_norm @ q_norm.T).squeeze(-1)  # [n]
        sim_matrix = s_norm @ s_norm.T  # [n, n]

        selected_indices: list[int] = []
        candidate_mask = list(range(n))

        for _ in range(k):
            if not candidate_mask:
                break
            best_idx: int | None = None
            best_score = float("-inf")
            for ci in candidate_mask:
                rel = mmr_lambda * relevance[ci].item()
                if selected_indices:
                    max_sim_to_selected = max(sim_matrix[ci, si].item() for si in selected_indices)
                    div = (1.0 - mmr_lambda) * max_sim_to_selected
                else:
                    div = 0.0
                score = rel - div
                if score > best_score:
                    best_score = score
                    best_idx = ci
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            candidate_mask.remove(best_idx)

        return [self.store.values[i] for i in selected_indices]

    # ------------------------------------------------------------------
    def select_by_coverage(self, query_ids: torch.Tensor, k: int, n_gram: int = 2) -> list[dict]:
        """Select demonstrations that cover diverse n-grams present in the query."""
        n = min(self.store.size, self.store.capacity)
        k = min(k, n)

        query_list = query_ids.tolist()
        query_ngrams: set = set()
        for i in range(len(query_list) - n_gram + 1):
            query_ngrams.add(tuple(query_list[i : i + n_gram]))

        covered: set = set()
        selected: list[dict] = []
        remaining = list(range(n))

        while len(selected) < k and remaining:
            best_idx: int | None = None
            best_new = -1
            for ri in remaining:
                demo = self.store.values[ri]
                demo_tokens = demo.get("input", []) + demo.get("output", [])
                demo_ngrams = set()
                for i in range(len(demo_tokens) - n_gram + 1):
                    demo_ngrams.add(tuple(demo_tokens[i : i + n_gram]))
                new_coverage = len((demo_ngrams & query_ngrams) - covered)
                if new_coverage > best_new:
                    best_new = new_coverage
                    best_idx = ri
            if best_idx is None:
                break
            selected.append(self.store.values[best_idx])
            demo_tokens = self.store.values[best_idx].get("input", []) + self.store.values[
                best_idx
            ].get("output", [])
            for i in range(len(demo_tokens) - n_gram + 1):
                covered.add(tuple(demo_tokens[i : i + n_gram]))
            remaining.remove(best_idx)

        return selected


# ---------------------------------------------------------------------------
# ICLPromptAssembler
# ---------------------------------------------------------------------------


class ICLPromptAssembler:
    """Assembles in-context learning prompts from demonstrations + query."""

    def __init__(self, max_demo_tokens: int = 64, separator_id: int = 0) -> None:
        self.max_demo_tokens = max_demo_tokens
        self.separator_id = separator_id

    # ------------------------------------------------------------------
    def assemble(self, demos: list[dict], query_ids: torch.Tensor) -> torch.Tensor:
        """Concatenate demo_input + demo_output + sep + ... + query_ids.

        Returns a 1-D LongTensor.
        """
        parts: list[torch.Tensor] = []
        for demo in demos:
            inp = torch.tensor(demo.get("input", []), dtype=torch.long)
            out = torch.tensor(demo.get("output", []), dtype=torch.long)
            sep = torch.tensor([self.separator_id], dtype=torch.long)
            demo_seq = torch.cat([inp, out, sep], dim=0)
            # Clip each demo to budget
            demo_seq = demo_seq[: self.max_demo_tokens]
            parts.append(demo_seq)
        parts.append(query_ids.long())
        return torch.cat(parts, dim=0)

    # ------------------------------------------------------------------
    def reorder_by_similarity(
        self,
        demos: list[dict],
        query_ids: torch.Tensor,
        encoder: DemonstrationEncoder,
    ) -> list[dict]:
        """Sort demos by cosine similarity to query (most similar last).

        Placing the most relevant demo closest to the query is a common
        ICL ordering heuristic.
        """
        if not demos:
            return demos

        with torch.no_grad():
            q_emb = encoder(query_ids.unsqueeze(0)).squeeze(0)  # [d]
            q_emb_n = F.normalize(q_emb.unsqueeze(0), dim=-1)

            scores = []
            for demo in demos:
                inp = torch.tensor(
                    demo.get("input", []) + demo.get("output", []),
                    dtype=torch.long,
                )
                if inp.numel() == 0:
                    scores.append(0.0)
                    continue
                d_emb = encoder(inp.unsqueeze(0)).squeeze(0)
                d_emb_n = F.normalize(d_emb.unsqueeze(0), dim=-1)
                sim = (d_emb_n @ q_emb_n.T).item()
                scores.append(sim)

        # Sort ascending so most similar is last
        sorted_pairs = sorted(zip(scores, demos), key=lambda x: x[0])
        return [d for _, d in sorted_pairs]

    # ------------------------------------------------------------------
    def truncate_to_budget(self, demos: list[dict], budget: int) -> list[dict]:
        """Drop demonstrations (from front) until total demo tokens <= budget."""
        result = list(demos)
        while result:
            total = sum(len(d.get("input", [])) + len(d.get("output", [])) for d in result)
            if total <= budget:
                break
            result.pop(0)
        return result


# ---------------------------------------------------------------------------
# ICLRetrievalTrainer
# ---------------------------------------------------------------------------


class ICLRetrievalTrainer:
    """Trains the demonstration encoder with InfoNCE contrastive loss."""

    def __init__(
        self,
        lm: nn.Module,
        encoder: DemonstrationEncoder,
        lr: float = 1e-3,
    ) -> None:
        self.lm = lm
        self.encoder = encoder
        self.optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

    # ------------------------------------------------------------------
    def contrastive_loss(
        self,
        query_emb: torch.Tensor,  # [B, d]
        pos_emb: torch.Tensor,  # [B, d]
        neg_emb: torch.Tensor,  # [B, d]
    ) -> torch.Tensor:
        """InfoNCE loss: pull queries toward positive, push away negative.

        For each query i the positives are the same-index pos embeddings
        and all other-index positives + all negatives are treated as
        negatives in the contrastive denominator.
        """
        B = query_emb.shape[0]
        q = F.normalize(query_emb, dim=-1)  # [B, d]
        p = F.normalize(pos_emb, dim=-1)  # [B, d]
        n = F.normalize(neg_emb, dim=-1)  # [B, d]

        # Logits: query vs {all positives, all negatives}
        # [B, 2B]
        candidates = torch.cat([p, n], dim=0)  # [2B, d]
        logits = q @ candidates.T  # [B, 2B]

        # Targets: for query i the positive is at index i
        targets = torch.arange(B, device=query_emb.device)
        loss = F.cross_entropy(logits, targets)
        return loss

    # ------------------------------------------------------------------
    def train_retriever_step(
        self,
        query_ids: torch.Tensor,  # [B, T]
        pos_ids: torch.Tensor,  # [B, T]
        neg_ids: torch.Tensor,  # [B, T]
    ) -> torch.Tensor:
        """Single gradient step on the encoder."""
        self.optimizer.zero_grad()
        q_emb = self.encoder(query_ids)
        p_emb = self.encoder(pos_ids)
        n_emb = self.encoder(neg_ids)
        loss = self.contrastive_loss(q_emb, p_emb, n_emb)
        loss.backward()
        self.optimizer.step()
        return loss.detach()
