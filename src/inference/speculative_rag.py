"""Speculative RAG: retrieval-augmented generation with speculative decoding.

Combines retrieval-augmented generation (RAG) with speculative decoding:
  1. Retrieve top-k documents for the query via TF-IDF-style word-overlap scoring.
  2. Draft: a small/fast model generates a candidate answer using retrieved context.
  3. Verify: the large target model re-scores the draft using the same context.
  4. Accept/reject tokens using standard speculative acceptance sampling.
  5. Optional: re-rank retrieved docs based on draft quality.

Public API
----------
SpeculativeRAGConfig  — configuration dataclass
Document              — retrieved document with text, id, and score
DocumentRetriever     — word-overlap TF-IDF-style retriever over a document store
ContextBuilder        — concatenates query + retrieved docs into a token sequence
SpeculativeRAGDecoder — full RAG speculative generation loop
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeRAGConfig:
    """Configuration for Speculative RAG decoding."""

    n_draft_tokens: int = 8  # tokens to draft per round
    top_k_docs: int = 3  # docs to retrieve
    draft_temperature: float = 1.0
    verify_temperature: float = 1.0
    rerank: bool = False  # re-rank docs after draft


# ---------------------------------------------------------------------------
# Document and Retriever
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A single document in the retrieval corpus."""

    text: str
    doc_id: str
    score: float = 0.0


def _word_overlap_score(query: str, doc_text: str) -> float:
    """Count shared words between query and doc (case-insensitive).

    Returns the number of overlapping unique words divided by the total
    unique words in the query. Returns 0.0 when either string is empty.
    """
    query_words = set(query.lower().split())
    doc_words = set(doc_text.lower().split())
    if not query_words or not doc_words:
        return 0.0
    intersection = query_words & doc_words
    return len(intersection) / len(query_words)


class DocumentRetriever:
    """Simple TF-IDF-style retriever over an in-memory document store.

    Scoring is based on word-overlap between the query and each document.
    No external dependencies (no sklearn, no scipy).

    Parameters
    ----------
    documents:
        Initial list of :class:`Document` objects to index.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._corpus: list[Document] = list(documents)

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """Return the top-k documents scored by word overlap with *query*.

        Parameters
        ----------
        query:
            The query string.
        top_k:
            Number of documents to return.

        Returns
        -------
        List of :class:`Document` objects sorted by score descending,
        length min(top_k, corpus_size).
        """
        if not self._corpus:
            return []

        scored: list[Document] = []
        for doc in self._corpus:
            score = _word_overlap_score(query, doc.text)
            scored.append(Document(text=doc.text, doc_id=doc.doc_id, score=score))

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """Append new documents to the corpus.

        Parameters
        ----------
        docs:
            Documents to add.
        """
        self._corpus.extend(docs)

    def __len__(self) -> int:
        return len(self._corpus)


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Concatenates query + retrieved docs into a single token sequence.

    Produces a flat integer tensor suitable for passing to a language model.
    Documents are appended in order; the sequence is truncated to *max_len*
    tokens when the concatenated length would exceed it.
    """

    def build(
        self,
        query_ids: Tensor,  # (T_q,)
        doc_ids: list[Tensor],  # list of (T_d,) token sequences
        max_len: int = 512,
    ) -> Tensor:  # (max_len,) or shorter
        """Build a context tensor from query and document token sequences.

        Parameters
        ----------
        query_ids:
            1-D int64 token sequence for the query, shape ``(T_q,)``.
        doc_ids:
            List of 1-D int64 token sequences for each retrieved document.
        max_len:
            Hard cap on the output length.

        Returns
        -------
        1-D int64 tensor of length ``<= max_len``.
        """
        parts: list[Tensor] = [query_ids.reshape(-1)]
        for d in doc_ids:
            parts.append(d.reshape(-1))

        context = torch.cat(parts, dim=0)  # (sum_T,)
        return context[:max_len]


# ---------------------------------------------------------------------------
# Mock models (self-contained, no imports from src/model)
# ---------------------------------------------------------------------------


class MockDraftModel(nn.Module):
    """Tiny random draft model for testing.

    Takes ``input_ids (B, T)`` → ``logits (B, T, vocab_size)``.
    The projection weights are fixed at construction time so tests are
    reproducible when a manual seed is set before calling forward.
    """

    def __init__(self, vocab_size: int = 256, hidden: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            ``(B, T)`` int64 token ids.

        Returns
        -------
        ``(B, T, vocab_size)`` float logits.
        """
        x = self.embed(input_ids)  # (B, T, hidden)
        return self.proj(x)  # (B, T, vocab_size)


class MockTargetModel(nn.Module):
    """Tiny random target model for testing.

    Slightly larger hidden dimension than the draft model to simulate a
    more capable verifier. Takes ``input_ids (B, T)`` → ``logits (B, T, vocab_size)``.
    """

    def __init__(self, vocab_size: int = 256, hidden: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        input_ids:
            ``(B, T)`` int64 token ids.

        Returns
        -------
        ``(B, T, vocab_size)`` float logits.
        """
        x = self.embed(input_ids)  # (B, T, hidden)
        return self.proj(x)  # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Speculative RAG Decoder
# ---------------------------------------------------------------------------


class SpeculativeRAGDecoder:
    """Full Speculative RAG generation loop.

    Retrieves context documents, drafts tokens with the small model using the
    retrieved context, then verifies with the target model and applies
    speculative acceptance/rejection sampling.

    Parameters
    ----------
    draft_model:
        Small/fast model; ``nn.Module`` taking ``(B, T)`` → ``(B, T, V)``.
    target_model:
        Large/accurate model; same signature.
    retriever:
        :class:`DocumentRetriever` instance over the document corpus.
    config:
        :class:`SpeculativeRAGConfig` controlling hyperparameters.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        retriever: DocumentRetriever,
        config: SpeculativeRAGConfig,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.retriever = retriever
        self.config = config
        self._context_builder = ContextBuilder()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tokenize_doc(self, doc: Document, vocab_size: int = 256) -> Tensor:
        """Convert doc text to a token sequence via simple ASCII hashing.

        This is a lightweight tokenizer suitable for testing: each character
        maps to ``ord(char) % vocab_size``.

        Returns 1-D int64 tensor.
        """
        ids = [ord(c) % vocab_size for c in doc.text]
        return torch.tensor(ids, dtype=torch.long)

    def _query_to_text(self, query_ids: Tensor) -> str:
        """Convert query token ids back to a pseudo-text for retrieval scoring."""
        return " ".join(str(t.item()) for t in query_ids.reshape(-1))

    # ------------------------------------------------------------------
    # Draft phase
    # ------------------------------------------------------------------

    def draft_with_context(
        self,
        query_ids: Tensor,
        context_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Draft *n_draft_tokens* tokens autoregressively using the draft model.

        Parameters
        ----------
        query_ids:
            ``(T_q,)`` query token ids (unused directly — context already built).
        context_ids:
            ``(T_c,)`` flattened context including query + docs.

        Returns
        -------
        draft_ids : ``(1, n_draft_tokens)`` int64 draft token ids.
        draft_logits : ``(1, n_draft_tokens, V)`` logits at each draft step.
        """
        n = self.config.n_draft_tokens
        temp = max(float(self.config.draft_temperature), 1e-8)

        # Add batch dimension
        current = context_ids.reshape(1, -1)  # (1, T_c)

        all_ids: list[Tensor] = []
        all_logits: list[Tensor] = []

        with torch.no_grad():
            for _ in range(n):
                logits = self.draft_model(current)  # (1, T, V)
                last_logit = logits[:, -1:, :]  # (1, 1, V)
                last_logit_2d = logits[:, -1, :]  # (1, V)

                probs = F.softmax(last_logit_2d / temp, dim=-1)  # (1, V)
                next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)

                all_ids.append(next_tok)  # (1, 1)
                all_logits.append(last_logit)  # (1, 1, V)

                current = torch.cat([current, next_tok], dim=1)  # (1, T+1)

        draft_ids = torch.cat(all_ids, dim=1)  # (1, n)
        draft_logits = torch.cat(all_logits, dim=1)  # (1, n, V)

        return draft_ids, draft_logits

    # ------------------------------------------------------------------
    # Verification phase
    # ------------------------------------------------------------------

    def verify_with_context(
        self,
        context_ids: Tensor,
        draft_ids: Tensor,
    ) -> Tensor:
        """Run the target model on ``[context_ids | draft_ids]`` in one forward pass.

        Parameters
        ----------
        context_ids:
            ``(T_c,)`` or ``(1, T_c)`` context token ids.
        draft_ids:
            ``(1, n_draft)`` draft token ids.

        Returns
        -------
        target_logits : ``(1, T_c + n_draft, V)`` logits from the target model.
        """
        ctx = context_ids.reshape(1, -1)  # (1, T_c)
        full_ids = torch.cat([ctx, draft_ids], dim=1)  # (1, T_c + n_draft)

        with torch.no_grad():
            target_logits = self.target_model(full_ids)  # (1, T_c + n_draft, V)

        return target_logits

    # ------------------------------------------------------------------
    # Acceptance / rejection sampling
    # ------------------------------------------------------------------

    def _accept_reject(
        self,
        context_len: int,
        draft_ids: Tensor,  # (1, n_draft)
        draft_logits: Tensor,  # (1, n_draft, V)
        target_logits: Tensor,  # (1, T_c + n_draft, V)
    ) -> tuple[Tensor, int]:
        """Apply speculative acceptance sampling for each draft token.

        For each draft position *i*:
          p_draft = softmax(draft_logits[:, i, :])
          p_target = softmax(target_logits[:, context_len-1+i, :])
          accept_prob = min(1, p_target[token] / p_draft[token])
          u ~ Uniform(0, 1) → accept if u < accept_prob

        On rejection, sample from corrected distribution
        ``max(0, p_target - p_draft)`` and stop.
        If all drafts accepted, sample one bonus token from target.

        Returns
        -------
        accepted_ids : ``(1, k)`` int64 — accepted tokens (k >= 0).
        n_accepted   : int — count of originally drafted tokens accepted.
        """
        d_temp = max(float(self.config.draft_temperature), 1e-8)
        v_temp = max(float(self.config.verify_temperature), 1e-8)
        n_draft = draft_ids.shape[1]

        accepted: list[Tensor] = []
        n_accepted = 0

        for i in range(n_draft):
            # Draft distribution at step i
            d_logit = draft_logits[:, i, :]  # (1, V)
            d_probs = F.softmax(d_logit / d_temp, dim=-1)  # (1, V)

            # Target distribution: position (context_len - 1 + i) predicts
            # the token at context_len + i, which is draft_ids[:, i].
            t_pos = context_len - 1 + i
            t_logit = target_logits[:, t_pos, :]  # (1, V)
            t_probs = F.softmax(t_logit / v_temp, dim=-1)  # (1, V)

            draft_tok = draft_ids[:, i]  # (1,)
            p_draft = d_probs[0, draft_tok[0].item()]  # scalar
            p_target = t_probs[0, draft_tok[0].item()]  # scalar

            accept_prob = min(1.0, (p_target / (p_draft + 1e-10)).item())
            u = torch.rand(1).item()

            if u < accept_prob:
                accepted.append(draft_tok.unsqueeze(0))  # (1, 1)
                n_accepted += 1
            else:
                # Corrected distribution: max(0, p_target - p_draft)
                adj = (t_probs - d_probs).clamp(min=0.0)  # (1, V)
                adj_sum = adj.sum(dim=-1, keepdim=True)
                adj = torch.where(
                    adj_sum < 1e-10,
                    t_probs,
                    adj / (adj_sum + 1e-10),
                )
                fallback = torch.multinomial(adj, num_samples=1)  # (1, 1)
                accepted.append(fallback)
                break  # stop after first rejection

        # Bonus token when all drafts accepted
        if n_accepted == n_draft:
            bonus_pos = context_len + n_draft - 1
            bonus_logit = target_logits[:, bonus_pos, :]  # (1, V)
            bonus_probs = F.softmax(bonus_logit / v_temp, dim=-1)
            bonus_tok = torch.multinomial(bonus_probs, num_samples=1)  # (1, 1)
            accepted.append(bonus_tok)

        if accepted:
            accepted_ids = torch.cat(accepted, dim=1)  # (1, k)
        else:
            accepted_ids = torch.zeros(1, 0, dtype=torch.long)

        return accepted_ids, n_accepted

    # ------------------------------------------------------------------
    # Optional re-ranking
    # ------------------------------------------------------------------

    def _rerank_docs(
        self,
        docs: list[Document],
        draft_ids: Tensor,  # (1, n_draft)
    ) -> list[Document]:
        """Re-rank retrieved docs based on word overlap with draft tokens.

        Uses the draft token ids as a surrogate query (converted to a
        space-separated string of ids) and re-scores each doc.
        """
        draft_query = " ".join(str(t.item()) for t in draft_ids.reshape(-1))
        reranked: list[Document] = []
        for doc in docs:
            new_score = _word_overlap_score(draft_query, doc.text)
            reranked.append(Document(text=doc.text, doc_id=doc.doc_id, score=new_score))
        reranked.sort(key=lambda d: d.score, reverse=True)
        return reranked

    # ------------------------------------------------------------------
    # Main generate loop
    # ------------------------------------------------------------------

    def generate(
        self,
        query_ids: Tensor,  # (T_q,) or (1, T_q)
        max_new_tokens: int = 32,
    ) -> tuple[Tensor, dict]:
        """Full RAG speculative generation.

        Steps per round:
          1. Retrieve top-k documents using word-overlap scoring on query.
          2. Build context = [query_ids | doc1_ids | doc2_ids | ...].
          3. Draft *n_draft_tokens* with the draft model.
          4. Optionally re-rank docs based on draft quality.
          5. Verify with target model using same context.
          6. Accept/reject with speculative sampling.
          7. Append accepted tokens to the running output.

        Parameters
        ----------
        query_ids:
            ``(T_q,)`` or ``(1, T_q)`` query token ids.
        max_new_tokens:
            Maximum number of new tokens to generate (total across all rounds).

        Returns
        -------
        output_ids : ``(1, T_q + n_generated)`` int64 generated token ids
                     (excludes context documents from the returned sequence).
        stats      : dict with keys ``n_accepted``, ``n_rounds``,
                     ``acceptance_rate``, and ``n_docs_retrieved``.
        """
        cfg = self.config
        query_flat = query_ids.reshape(-1)  # (T_q,)

        # Retrieve documents
        query_text = self._query_to_text(query_flat)
        docs = self.retriever.retrieve(query_text, top_k=cfg.top_k_docs)

        # Build document token sequences
        vocab_size: int = 256
        # Infer vocab_size from draft model if possible
        if hasattr(self.draft_model, "vocab_size"):
            vocab_size = self.draft_model.vocab_size

        doc_token_seqs: list[Tensor] = [
            self._tokenize_doc(doc, vocab_size=vocab_size) for doc in docs
        ]

        # Build the context (query + docs), cap at 512 tokens
        max_ctx = 512
        context_ids = self._context_builder.build(query_flat, doc_token_seqs, max_len=max_ctx)

        # Running output (just the generated portion, not the context docs)
        output_ids: Tensor = query_flat.clone()  # (T_q,) initially

        total_accepted = 0
        n_rounds = 0
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            n_draft = min(cfg.n_draft_tokens, remaining)

            # Temporarily override n_draft_tokens
            original_n = cfg.n_draft_tokens
            cfg.n_draft_tokens = n_draft

            # Draft
            draft_ids, draft_logits = self.draft_with_context(
                query_flat, context_ids
            )  # (1, n_draft), (1, n_draft, V)

            # Optional re-rank
            if cfg.rerank and docs:
                docs = self._rerank_docs(docs, draft_ids)
                doc_token_seqs = [self._tokenize_doc(d, vocab_size=vocab_size) for d in docs]
                context_ids = self._context_builder.build(
                    query_flat, doc_token_seqs, max_len=max_ctx
                )

            cfg.n_draft_tokens = original_n

            # Verify
            target_logits = self.verify_with_context(context_ids, draft_ids)
            # (1, T_c + n_draft, V)

            context_len = context_ids.shape[0]

            # Accept / reject
            accepted_ids, n_acc = self._accept_reject(
                context_len, draft_ids, draft_logits, target_logits
            )

            total_accepted += n_acc
            n_rounds += 1

            # Append accepted tokens to context and output
            if accepted_ids.shape[1] > 0:
                n_to_add = min(accepted_ids.shape[1], remaining)
                new_toks = accepted_ids[:, :n_to_add]  # (1, k)

                # Extend context
                context_ids = torch.cat([context_ids.reshape(-1), new_toks.reshape(-1)], dim=0)
                # Truncate context to keep it manageable
                if context_ids.shape[0] > max_ctx:
                    context_ids = context_ids[-max_ctx:]

                # Extend output_ids
                output_ids = torch.cat([output_ids, new_toks.reshape(-1)], dim=0)
                tokens_generated += n_to_add
            else:
                # No token accepted: fallback — sample one token from target
                with torch.no_grad():
                    fallback_input = context_ids.reshape(1, -1)
                    fb_logits = self.target_model(fallback_input)  # (1, T, V)
                    last_v_temp = max(float(cfg.verify_temperature), 1e-8)
                    fb_probs = F.softmax(fb_logits[:, -1, :] / last_v_temp, dim=-1)
                    fb_tok = torch.multinomial(fb_probs, num_samples=1)  # (1, 1)

                context_ids = torch.cat([context_ids, fb_tok.reshape(-1)], dim=0)
                if context_ids.shape[0] > max_ctx:
                    context_ids = context_ids[-max_ctx:]

                output_ids = torch.cat([output_ids, fb_tok.reshape(-1)], dim=0)
                tokens_generated += 1

        # Final stats
        total_draft_tokens = cfg.n_draft_tokens * n_rounds
        acceptance_rate = total_accepted / total_draft_tokens if total_draft_tokens > 0 else 0.0

        stats: dict = {
            "n_accepted": total_accepted,
            "n_rounds": n_rounds,
            "acceptance_rate": acceptance_rate,
            "n_docs_retrieved": len(docs),
        }

        return output_ids.reshape(1, -1), stats
