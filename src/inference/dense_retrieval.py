"""Dense passage retrieval: bi-encoder embeddings, flat vector index, and cross-encoder re-ranking."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class RetrieverConfig:
    """Configuration for dense passage retrieval."""

    d_model: int = 512
    index_type: str = "flat"          # "flat" | "ivf" (ivf is a placeholder)
    top_k: int = 5
    use_reranker: bool = False
    normalize_embeddings: bool = True
    batch_size: int = 32


class BiEncoder(nn.Module):
    """Bi-encoder for dense retrieval: shared backbone encodes queries and passages.

    The backbone's logits are mean-pooled over the token dimension and projected to
    a compact embedding space via a learned linear layer.
    """

    def __init__(self, backbone: nn.Module, d_model: int, proj_dim: int) -> None:
        """
        Args:
            backbone: Shared transformer backbone (AureliusTransformer or compatible).
            d_model: Vocabulary size of the backbone (size of the logits dimension).
            proj_dim: Output embedding dimension after projection.
        """
        super().__init__()
        self.backbone = backbone
        # Project from vocab logits → compact embedding
        self.proj = nn.Linear(d_model, proj_dim, bias=False)
        self._normalize = True  # controlled via RetrieverConfig at encode-time

    def encode(self, input_ids: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode a batch of token sequences into dense embeddings.

        Args:
            input_ids: (B, T) integer token ids.
            normalize: If True, L2-normalize the output embeddings.

        Returns:
            (B, proj_dim) embedding tensor.
        """
        with torch.no_grad():
            # Backbone returns (loss, logits, past_key_values)
            _, logits, _ = self.backbone(input_ids)  # logits: (B, T, V)

        # Mean-pool over the sequence dimension
        pooled = logits.mean(dim=1)  # (B, V)

        # Project to embedding space
        emb = self.proj(pooled)  # (B, proj_dim)

        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    def forward(
        self,
        query_ids: torch.Tensor,
        passage_ids: torch.Tensor,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of query-passage pairs.

        Args:
            query_ids:   (B, T) query token ids.
            passage_ids: (B, T) passage token ids.
            normalize:   L2-normalize embeddings.

        Returns:
            Tuple of (query_emb, passage_emb), each (B, proj_dim).
        """
        query_emb = self.encode(query_ids, normalize=normalize)
        passage_emb = self.encode(passage_ids, normalize=normalize)
        return query_emb, passage_emb


class FlatIndex:
    """Brute-force flat vector index using dot-product (or cosine) similarity.

    No external dependencies — uses pure PyTorch matrix multiply for search.
    """

    def __init__(self, dim: int) -> None:
        """
        Args:
            dim: Embedding dimension.
        """
        self.dim = dim
        self._embeddings: torch.Tensor | None = None  # (N, dim) accumulated store
        self._ids: list[int] = []

    def add(self, embeddings: torch.Tensor, ids: list[int] | None = None) -> None:
        """Append embeddings to the index.

        Args:
            embeddings: (N, dim) float tensor.
            ids:        List of integer ids (defaults to sequential from current size).
        """
        embeddings = embeddings.detach().float().cpu()
        n = embeddings.shape[0]

        if ids is None:
            start = len(self._ids)
            ids = list(range(start, start + n))

        if self._embeddings is None:
            self._embeddings = embeddings
        else:
            self._embeddings = torch.cat([self._embeddings, embeddings], dim=0)

        self._ids.extend(ids)

    def search(
        self, query: torch.Tensor, top_k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the top-k most similar entries for each query vector.

        Args:
            query:  (d,) or (B, d) float tensor.
            top_k:  Number of results to return per query.

        Returns:
            Tuple of (scores, ids):
                scores: (B, top_k) similarity scores.
                ids:    (B, top_k) integer ids (as LongTensor).
        """
        if self._embeddings is None or len(self._ids) == 0:
            raise RuntimeError("Index is empty — call add() first.")

        query = query.detach().float().cpu()
        if query.dim() == 1:
            query = query.unsqueeze(0)  # (1, d)

        # Dot-product similarity — (B, N)
        scores = query @ self._embeddings.T

        k = min(top_k, self._embeddings.shape[0])
        top_scores, top_local_indices = torch.topk(scores, k, dim=1)  # (B, k)

        # Map local indices → stored ids
        id_tensor = torch.tensor(self._ids, dtype=torch.long)
        top_ids = id_tensor[top_local_indices]  # (B, k)

        return top_scores, top_ids

    def size(self) -> int:
        """Return the number of embeddings stored in the index."""
        return len(self._ids)

    def reset(self) -> None:
        """Clear all stored embeddings and ids."""
        self._embeddings = None
        self._ids = []


class CrossEncoderReranker(nn.Module):
    """Cross-encoder reranker that scores concatenated query-passage pairs.

    Uses the backbone's last-token logits projected to a scalar relevance score.
    """

    def __init__(self, backbone: nn.Module) -> None:
        """
        Args:
            backbone: Shared transformer backbone.
        """
        super().__init__()
        self.backbone = backbone

        # Infer vocab_size from lm_head or embedding
        if hasattr(backbone, "lm_head"):
            vocab_size = backbone.lm_head.weight.shape[0]
        elif hasattr(backbone, "embed"):
            vocab_size = backbone.embed.weight.shape[0]
        else:
            raise ValueError("Cannot infer vocab_size from backbone.")

        self.score_head = nn.Linear(vocab_size, 1, bias=True)

    def score(self, query_passage_ids: torch.Tensor) -> torch.Tensor:
        """Score a batch of concatenated query+passage token sequences.

        Args:
            query_passage_ids: (B, T) integer token ids.

        Returns:
            (B,) relevance scores.
        """
        with torch.no_grad():
            _, logits, _ = self.backbone(query_passage_ids)  # (B, T, V)

        # Use the last token's logits
        last_logits = logits[:, -1, :]  # (B, V)

        scores = self.score_head(last_logits).squeeze(-1)  # (B,)
        return scores


def retrieve_and_rerank(
    query_ids: torch.Tensor,
    index: FlatIndex,
    bi_encoder: BiEncoder,
    reranker: CrossEncoderReranker | None,
    passage_ids_list: list[torch.Tensor],
    config: RetrieverConfig,
) -> list[int]:
    """Retrieve top-k passages and optionally rerank with a cross-encoder.

    Args:
        query_ids:        (1, T) or (T,) query token ids.
        index:            Populated FlatIndex of passage embeddings.
        bi_encoder:       BiEncoder for query encoding.
        reranker:         Optional CrossEncoderReranker for reranking.
        passage_ids_list: List of passage token id tensors (one per passage in index).
        config:           RetrieverConfig controlling top_k and reranker usage.

    Returns:
        List of passage indices (integer ids from the index), sorted by score descending.
    """
    # Ensure query_ids is (1, T)
    if query_ids.dim() == 1:
        query_ids = query_ids.unsqueeze(0)

    # Encode query
    query_emb = bi_encoder.encode(query_ids, normalize=config.normalize_embeddings)  # (1, proj_dim)

    # Search index for top_k candidates
    scores, candidate_ids = index.search(query_emb, top_k=config.top_k)
    # scores: (1, k), candidate_ids: (1, k)
    candidate_ids_list: list[int] = candidate_ids[0].tolist()

    if reranker is not None and config.use_reranker:
        # Concatenate query + passage tokens for cross-encoder scoring
        query_seq = query_ids[0]  # (T_q,)
        pairs: list[torch.Tensor] = []
        for cid in candidate_ids_list:
            passage_seq = passage_ids_list[cid]
            if passage_seq.dim() > 1:
                passage_seq = passage_seq[0]
            pair = torch.cat([query_seq, passage_seq], dim=0)  # (T_q + T_p,)
            pairs.append(pair)

        # Pad to same length
        max_len = max(p.shape[0] for p in pairs)
        padded = torch.zeros(len(pairs), max_len, dtype=torch.long)
        for i, p in enumerate(pairs):
            padded[i, : p.shape[0]] = p

        pair_scores = reranker.score(padded)  # (k,)

        # Sort by reranker score descending
        sorted_order = torch.argsort(pair_scores, descending=True)
        candidate_ids_list = [candidate_ids_list[i] for i in sorted_order.tolist()]
    else:
        # Already sorted by bi-encoder score (topk returns sorted)
        pass

    return candidate_ids_list


def build_index(
    passages_ids: list[torch.Tensor],
    bi_encoder: BiEncoder,
    config: RetrieverConfig,
) -> FlatIndex:
    """Encode all passages in batches and populate a FlatIndex.

    Args:
        passages_ids: List of passage token id tensors.  Each may be (T,) or (1, T).
        bi_encoder:   BiEncoder for passage encoding.
        config:       RetrieverConfig controlling batch_size and normalize_embeddings.

    Returns:
        FlatIndex populated with embeddings for all passages.
    """
    # Infer proj_dim from bi_encoder projection layer
    proj_dim = bi_encoder.proj.out_features

    index = FlatIndex(dim=proj_dim)

    n = len(passages_ids)
    for start in range(0, n, config.batch_size):
        batch = passages_ids[start : start + config.batch_size]

        # Normalize each tensor to (1, T) then stack into (B, T)
        tensors: list[torch.Tensor] = []
        for p in batch:
            if p.dim() == 1:
                tensors.append(p.unsqueeze(0))
            else:
                tensors.append(p[:1])  # take first row if batched

        # Pad to same length in this batch
        max_len = max(t.shape[1] for t in tensors)
        padded = torch.zeros(len(tensors), max_len, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, : t.shape[1]] = t[0]

        embs = bi_encoder.encode(padded, normalize=config.normalize_embeddings)  # (B, proj_dim)

        batch_ids = list(range(start, start + len(batch)))
        index.add(embs, ids=batch_ids)

    return index
