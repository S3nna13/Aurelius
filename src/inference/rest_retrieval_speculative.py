"""REST: Retrieval-Based Speculative Decoding (He et al., 2023 — arXiv:2311.08252).

REST replaces the neural draft model in speculative decoding with a retrieval
datastore built from training documents.  Candidates are n-gram continuations
rather than autoregressive samples, so draft generation has zero model cost.

Paper variable notation is preserved throughout:
  n       — n-gram order used as the retrieval key
  γ (gamma) — number of draft tokens proposed per step (max speculation length)
  c       — current context (token id list)
  d_t     — draft token at position t
  p       — target LM probability distribution

Public API
----------
RESTDatastore     — builds and queries the n-gram → continuations lookup table
RESTDecoder       — draft (Grow) + verify (Check) using exact-match acceptance
RESTAccelerator   — high-level single-step wrapper
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Datastore — Section 3.1 of the paper
# ---------------------------------------------------------------------------

class RESTDatastore:
    """N-gram retrieval datastore for REST draft generation.

    The datastore maps every n-gram tuple observed in training documents to the
    list of tokens that followed that n-gram, enabling frequency-based ranking.

    Parameters
    ----------
    n:
        N-gram order used as the retrieval key (default 3 matches paper).
    """

    def __init__(self, n: int = 3) -> None:
        if n < 1:
            raise ValueError(f"n-gram order must be >= 1, got {n}")
        self.n: int = n
        # Internal store: ngram_tuple -> Counter of next-token frequencies
        self._store: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add_document(self, token_ids: List[int]) -> None:
        """Index all n-grams in *token_ids* into the datastore.

        For each position i where a full n-gram [i..i+n) exists and a next
        token token_ids[i+n] exists, we record that continuation.

        Parameters
        ----------
        token_ids:
            Sequence of integer token ids from one training document.
        """
        if len(token_ids) < self.n + 1:
            # Not enough tokens to form even one (ngram, continuation) pair.
            return
        for i in range(len(token_ids) - self.n):
            ngram: Tuple[int, ...] = tuple(token_ids[i : i + self.n])
            next_tok: int = token_ids[i + self.n]
            self._store[ngram][next_tok] += 1

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(self, context_ids: List[int], top_k: int = 5) -> List[int]:
        """Return the top-*k* next tokens for the n-gram suffix of *context_ids*.

        Uses the last ``n`` tokens of *context_ids* as the query key and returns
        candidate tokens sorted by descending frequency.  Returns an empty list
        when the n-gram is not in the datastore (no crash).

        Parameters
        ----------
        context_ids:
            Current context as a list of integer token ids.
        top_k:
            Maximum number of candidate tokens to return.

        Returns
        -------
        List of up to *top_k* integer token ids, ordered by frequency (desc).
        """
        if len(context_ids) < self.n:
            return []
        query: Tuple[int, ...] = tuple(context_ids[-self.n :])
        counter = self._store.get(query)
        if not counter:
            return []
        return [tok for tok, _ in counter.most_common(top_k)]


# ---------------------------------------------------------------------------
# Decoder — Section 3.2 (Grow) + Section 3.3 (Check)
# ---------------------------------------------------------------------------

class RESTDecoder:
    """REST draft-and-verify decoder.

    Parameters
    ----------
    datastore:
        Pre-built :class:`RESTDatastore` instance.
    gamma:
        Maximum number of draft tokens γ to propose per step (paper default 4).
    """

    def __init__(self, datastore: RESTDatastore, gamma: int = 4) -> None:
        if gamma < 1:
            raise ValueError(f"gamma must be >= 1, got {gamma}")
        self.datastore: RESTDatastore = datastore
        self.gamma: int = gamma

    # ------------------------------------------------------------------
    # Grow step (Section 3.2)
    # ------------------------------------------------------------------

    def draft(self, context_ids: List[int], top_k: int = 5) -> List[int]:
        """Produce up to γ draft tokens via greedy n-gram retrieval.

        At each step the most frequent continuation is appended to the running
        context and used to query the next n-gram.  The loop stops after γ
        tokens or when the datastore returns no candidates (empty draft = safe
        fallback when context is too short or unseen).

        Parameters
        ----------
        context_ids:
            Current context (list of int token ids).  Not mutated.
        top_k:
            Number of candidates retrieved at each step; only the top-1 is used
            for greedy chaining but the parameter is forwarded to the datastore.

        Returns
        -------
        List of up to γ draft token ids.  May be empty.
        """
        draft_tokens: List[int] = []
        running_ctx: List[int] = list(context_ids)

        for _ in range(self.gamma):
            candidates = self.datastore.retrieve(running_ctx, top_k=top_k)
            if not candidates:
                break
            # Greedy: take the most frequent continuation (candidates[0])
            d_t: int = candidates[0]
            draft_tokens.append(d_t)
            running_ctx.append(d_t)

        return draft_tokens

    # ------------------------------------------------------------------
    # Check step — exact-match acceptance (Section 3.3)
    # ------------------------------------------------------------------

    def verify(
        self,
        context_ids: List[int],
        draft_tokens: List[int],
        target_probs_fn: Callable[[List[int]], Tensor],
    ) -> Tuple[List[int], int]:
        """Verify draft tokens against the target LM using greedy acceptance.

        The target LM is run *once* on the concatenated sequence
        ``context_ids + draft_tokens``.  Each draft token d_t is accepted iff
        it equals the argmax of the target distribution at that position
        (exact-match greedy criterion, Section 3.3).  On first rejection all
        subsequent draft tokens are discarded.  A bonus token — the argmax at
        the last accepted position + 1 — is always appended so every call
        produces at least one new token.

        Parameters
        ----------
        context_ids:
            Current context (list of int token ids).
        draft_tokens:
            Draft sequence produced by :meth:`draft`.
        target_probs_fn:
            Callable ``f(token_ids: List[int]) -> Tensor`` that returns a
            float tensor of shape ``(len(token_ids), V)`` where each row is the
            probability distribution p(· | prefix) at that position.

        Returns
        -------
        accepted : List[int]
            Accepted tokens including the bonus token at the end.
        n_accepted : int
            Number of *draft* tokens that were accepted (bonus token excluded).
        """
        if not draft_tokens:
            # No draft — just sample a single token from target at context.
            probs = target_probs_fn(context_ids)   # (T, V)
            bonus_tok: int = int(torch.argmax(probs[-1]).item())
            return [bonus_tok], 0

        full_seq: List[int] = list(context_ids) + list(draft_tokens)
        # Shape: (len(full_seq), V)
        probs: Tensor = target_probs_fn(full_seq)

        ctx_len: int = len(context_ids)
        n_accepted: int = 0

        for t, d_t in enumerate(draft_tokens):
            # Position in probs that predicts token at (ctx_len + t)
            # i.e., row ctx_len - 1 + t  (0-indexed, predicts next token)
            pred_pos: int = ctx_len - 1 + t
            target_greedy: int = int(torch.argmax(probs[pred_pos]).item())
            if target_greedy == d_t:
                n_accepted += 1
            else:
                # Stop at first mismatch
                break

        # Bonus token: argmax at the position after the last accepted draft token
        bonus_pos: int = ctx_len - 1 + n_accepted
        bonus_tok = int(torch.argmax(probs[bonus_pos]).item())

        accepted: List[int] = list(draft_tokens[:n_accepted]) + [bonus_tok]
        return accepted, n_accepted


# ---------------------------------------------------------------------------
# High-level accelerator
# ---------------------------------------------------------------------------

class RESTAccelerator:
    """High-level wrapper that performs one complete REST step.

    Parameters
    ----------
    datastore:
        Pre-built :class:`RESTDatastore` instance.
    gamma:
        Maximum speculation length γ (draft tokens per step).
    top_k:
        Number of datastore candidates to retrieve per draft position.
    """

    def __init__(
        self,
        datastore: RESTDatastore,
        gamma: int = 4,
        top_k: int = 5,
    ) -> None:
        self.decoder = RESTDecoder(datastore=datastore, gamma=gamma)
        self.top_k: int = top_k

    def step(
        self,
        context_ids: List[int],
        target_probs_fn: Callable[[List[int]], Tensor],
    ) -> Tuple[List[int], int]:
        """Run one draft + verify step.

        Parameters
        ----------
        context_ids:
            Current context (list of int token ids).
        target_probs_fn:
            Callable as described in :meth:`RESTDecoder.verify`.

        Returns
        -------
        new_tokens : List[int]
            Tokens to append to the context (≥ 1).
        n_accepted : int
            Number of draft tokens accepted (excludes bonus token).
        """
        draft_toks = self.decoder.draft(context_ids, top_k=self.top_k)
        new_tokens, n_accepted = self.decoder.verify(
            context_ids, draft_toks, target_probs_fn
        )
        return new_tokens, n_accepted
