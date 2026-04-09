"""Cross-encoder re-ranker for improving retrieval quality.

Complements rag.py and rag_pipeline.py by scoring query-document pairs jointly
using the Aurelius model through logit, perplexity, or embedding similarity methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RerankerConfig:
    max_seq_len: int = 256
    score_method: str = "logit"        # "logit" | "embedding" | "perplexity"
    batch_size: int = 8
    normalize_scores: bool = True


@dataclass
class ScoredDocument:
    text: str
    original_rank: int
    rerank_score: float
    original_score: float | None


def format_query_document(query: str, document: str) -> str:
    """Format query and document into a relevance prompt.

    Returns:
        'Query: {query}\\nDocument: {document}\\nRelevant:'
    """
    return f"Query: {query}\nDocument: {document}\nRelevant:"


def score_query_document_logit(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    query: str,
    document: str,
    max_seq_len: int = 256,
    yes_token_id: int = 121,    # ord('y')
    no_token_id: int = 110,     # ord('n')
) -> float:
    """Score relevance via yes/no logit at the last token position.

    Formats the prompt, encodes it, runs the model, then computes:
        log(p_yes / (p_yes + p_no))

    Returns:
        Float relevance score; higher = more relevant.
    """
    prompt = format_query_document(query, document)
    token_ids = encode_fn(prompt)[:max_seq_len]
    if not token_ids:
        return 0.0

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        loss, logits, _pkv = model(input_ids)

    # logits: (1, S, vocab_size) — pick last position
    last_logits = logits[0, -1, :]   # (vocab_size,)
    log_probs = F.log_softmax(last_logits, dim=-1)
    log_p_yes = log_probs[yes_token_id].item()
    log_p_no = log_probs[no_token_id].item()

    # log(p_yes / (p_yes + p_no)) = log_p_yes - log(p_yes + p_no)
    # Use logsumexp for numerical stability
    log_denom = torch.logsumexp(
        torch.tensor([log_p_yes, log_p_no]), dim=0
    ).item()
    return log_p_yes - log_denom


def score_query_document_perplexity(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    query: str,
    document: str,
    max_seq_len: int = 256,
) -> float:
    """Score relevance as negative perplexity of document given query context.

    Lower perplexity → higher relevance → returns -perplexity (higher = better).
    """
    query_ids = encode_fn(query)
    doc_ids = encode_fn(document)

    combined_ids = (query_ids + doc_ids)[:max_seq_len]
    if len(combined_ids) < 2:
        return 0.0

    input_ids = torch.tensor([combined_ids], dtype=torch.long)
    labels = torch.tensor([combined_ids], dtype=torch.long)

    with torch.no_grad():
        loss, logits, _pkv = model(input_ids, labels=labels)

    if loss is None:
        # Compute loss manually if model returned None
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        perplexity = math.exp(ce.item())
    else:
        perplexity = math.exp(loss.item())

    return -perplexity


def score_query_document_embedding(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    query: str,
    document: str,
    max_seq_len: int = 256,
) -> float:
    """Score as cosine similarity between query and document last-layer embeddings.

    Uses a forward hook on model.layers[-1] to capture hidden states, then
    mean-pools and computes cosine similarity.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(module: nn.Module, inputs: tuple, output: tuple) -> None:
        # TransformerBlock returns (hidden_state, kv); capture hidden state
        if isinstance(output, tuple):
            captured["hidden"] = output[0].detach()
        else:
            captured["hidden"] = output.detach()

    handle = model.layers[-1].register_forward_hook(hook_fn)

    def _encode_and_pool(text: str) -> torch.Tensor:
        token_ids = encode_fn(text)[:max_seq_len]
        if not token_ids:
            token_ids = [0]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            model(input_ids)
        hidden = captured.get("hidden")  # (1, S, d_model)
        if hidden is None:
            raise RuntimeError("Hook did not capture hidden state")
        return hidden[0].mean(dim=0)  # (d_model,)

    try:
        q_emb = _encode_and_pool(query)
        d_emb = _encode_and_pool(document)
    finally:
        handle.remove()

    q_norm = F.normalize(q_emb.float(), dim=-1)
    d_norm = F.normalize(d_emb.float(), dim=-1)
    return (q_norm @ d_norm).item()


def batch_score(
    model: nn.Module,
    encode_fn: Callable[[str], list[int]],
    query: str,
    documents: list[str],
    cfg: RerankerConfig,
) -> list[float]:
    """Score all documents for a query using cfg.score_method.

    Returns:
        List of floats, one score per document.
    """
    method = cfg.score_method
    scores: list[float] = []

    for doc in documents:
        if method == "logit":
            s = score_query_document_logit(
                model, encode_fn, query, doc, max_seq_len=cfg.max_seq_len
            )
        elif method == "perplexity":
            s = score_query_document_perplexity(
                model, encode_fn, query, doc, max_seq_len=cfg.max_seq_len
            )
        elif method == "embedding":
            s = score_query_document_embedding(
                model, encode_fn, query, doc, max_seq_len=cfg.max_seq_len
            )
        else:
            raise ValueError(f"Unknown score_method: {method!r}")
        scores.append(s)

    return scores


def rerank(
    query: str,
    documents: list[str],
    scores: list[float],
    original_scores: list[float] | None = None,
) -> list[ScoredDocument]:
    """Create ScoredDocuments and sort by rerank_score descending.

    Args:
        query:           the query string (unused here but kept for API symmetry)
        documents:       document texts
        scores:          reranking scores, one per document
        original_scores: optional original retrieval scores

    Returns:
        List of ScoredDocument sorted by rerank_score descending.
    """
    scored = [
        ScoredDocument(
            text=doc,
            original_rank=i,
            rerank_score=s,
            original_score=original_scores[i] if original_scores is not None else None,
        )
        for i, (doc, s) in enumerate(zip(documents, scores))
    ]
    scored.sort(key=lambda x: x.rerank_score, reverse=True)
    return scored


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[int]:
    """RRF fusion of multiple ranked lists.

    score[doc] = sum over rankings of 1 / (k + rank)
    where rank is 1-indexed.

    Returns:
        Document indices sorted by descending RRF score.
    """
    rrf_scores: dict[int, float] = {}
    for ranking in rankings:
        for rank_idx, doc_id in enumerate(ranking):
            rank = rank_idx + 1  # 1-indexed
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)


class CrossEncoderReranker:
    """Full re-ranking pipeline using an Aurelius model as cross-encoder."""

    def __init__(
        self,
        model: nn.Module,
        encode_fn: Callable[[str], list[int]],
        cfg: RerankerConfig,
    ) -> None:
        self.model = model
        self.encode_fn = encode_fn
        self.cfg = cfg

    def rerank(
        self,
        query: str,
        documents: list[str],
        original_scores: list[float] | None = None,
    ) -> list[ScoredDocument]:
        """Score and sort documents. Returns all ScoredDocuments sorted descending."""
        raw_scores = batch_score(
            self.model, self.encode_fn, query, documents, self.cfg
        )

        if self.cfg.normalize_scores and raw_scores:
            min_s = min(raw_scores)
            max_s = max(raw_scores)
            denom = max_s - min_s + 1e-8
            norm_scores = [(s - min_s) / denom for s in raw_scores]
        else:
            norm_scores = raw_scores

        return rerank(query, documents, norm_scores, original_scores)

    def evaluate_ndcg(
        self,
        query: str,
        documents: list[str],
        relevance_labels: list[int],    # 0=irrelevant, 1=relevant, 2=highly relevant
        top_k: int = 5,
    ) -> float:
        """Compute NDCG@k after reranking.

        DCG  = sum_i relevance[i] / log2(i + 2)   (i is 0-indexed)
        NDCG = DCG / IDCG (ideal DCG)
        """
        scored_docs = self.rerank(query, documents)

        # Build a map from document text to relevance label (handle duplicates by first)
        label_map: dict[str, int] = {}
        for doc, lbl in zip(documents, relevance_labels):
            if doc not in label_map:
                label_map[doc] = lbl

        # DCG after reranking
        dcg = 0.0
        k = min(top_k, len(scored_docs))
        for i, sd in enumerate(scored_docs[:k]):
            rel = label_map.get(sd.text, 0)
            dcg += rel / math.log2(i + 2)

        # Ideal DCG: sorted ground truth descending
        sorted_labels = sorted(relevance_labels, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(sorted_labels[:k]):
            idcg += rel / math.log2(i + 2)

        if idcg == 0.0:
            return 1.0 if dcg == 0.0 else 0.0

        return min(dcg / idcg, 1.0)

    def fusion_rerank(
        self,
        query: str,
        candidate_lists: list[list[str]],
    ) -> list[str]:
        """Fuse multiple candidate lists via RRF, then rerank top candidates.

        Returns document texts in reranked order.
        """
        # Build a global index of all unique documents
        all_docs: list[str] = []
        seen: set[str] = set()
        for lst in candidate_lists:
            for doc in lst:
                if doc not in seen:
                    all_docs.append(doc)
                    seen.add(doc)

        doc_to_idx = {doc: i for i, doc in enumerate(all_docs)}

        # Convert each candidate list to index rankings
        rankings: list[list[int]] = []
        for lst in candidate_lists:
            rankings.append([doc_to_idx[doc] for doc in lst if doc in doc_to_idx])

        # RRF fusion to get ordering of indices
        fused_indices = reciprocal_rank_fusion(rankings)

        # Gather documents in fused order
        fused_docs = [all_docs[i] for i in fused_indices]

        # Rerank the fused candidates
        scored = self.rerank(query, fused_docs)
        return [sd.text for sd in scored]
