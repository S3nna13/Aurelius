"""Token Recycling for speculative decoding (Tang et al., 2024).

Reuses previously generated tokens as draft candidates without any auxiliary
model. A recycling buffer maintains (token → next_token) transition counts
observed during generation. When generating token t, previously seen sequences
ending with the last k tokens are used as draft candidates — entirely free,
no model forward pass required. Draft tokens are then verified in parallel.

Public API
----------
TokenRecyclingConfig    — configuration dataclass
RecyclingBuffer         — sliding-window buffer with bigram/unigram statistics
TokenRecycler           — main class: update() from generated tokens, draft()
build_recycling_draft_tree — build a tree of draft candidates from a buffer
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# TokenRecyclingConfig
# ---------------------------------------------------------------------------


@dataclass
class TokenRecyclingConfig:
    """Configuration for token recycling speculative decoding."""

    max_draft_tokens: int = 5
    buffer_size: int = 512
    use_bigram: bool = True
    min_count: int = 1


# ---------------------------------------------------------------------------
# RecyclingBuffer
# ---------------------------------------------------------------------------


class RecyclingBuffer:
    """Sliding-window buffer that tracks unigram and bigram transition counts.

    The buffer holds the last *max_size* token ids seen during generation.
    Transition statistics are derived from consecutive pairs in the buffer.
    """

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size
        # Store raw token stream (capped)
        self._tokens: deque[int] = deque(maxlen=max_size)
        # adjacency[token] -> Counter of tokens that followed it
        self.adjacency: dict[int, Counter] = {}
        # bigram_table[(t1, t2)] -> Counter of tokens that followed bigram
        self.bigram_table: dict[tuple[int, int], Counter] = {}

    def add(self, token_id: int) -> None:
        """Add a single token to the buffer, updating transition tables."""
        if self._tokens:
            prev = self._tokens[-1]
            # Update unigram adjacency
            if prev not in self.adjacency:
                self.adjacency[prev] = Counter()
            self.adjacency[prev][token_id] += 1

            # Update bigram table if we have at least 2 prior tokens
            if len(self._tokens) >= 2:
                prev_prev = list(self._tokens)[-2]  # second to last
                bigram = (prev_prev, prev)
                if bigram not in self.bigram_table:
                    self.bigram_table[bigram] = Counter()
                self.bigram_table[bigram][token_id] += 1

        self._tokens.append(token_id)

    def get_recent(self, n: int) -> list[int]:
        """Return the last *n* tokens from the buffer (oldest first)."""
        tokens_list = list(self._tokens)
        return tokens_list[-n:] if n <= len(tokens_list) else tokens_list[:]

    def get_bigram_candidates(self, t1: int, t2: int, top_k: int = 5) -> list[tuple[int, int]]:
        """Return top_k (next_token, count) pairs following bigram (t1, t2).

        Results are sorted descending by count.
        """
        bigram = (t1, t2)
        counter = self.bigram_table.get(bigram)
        if counter is None:
            return []
        return counter.most_common(top_k)

    def get_unigram_candidates(self, token: int, top_k: int = 5) -> list[tuple[int, int]]:
        """Return top_k (next_token, count) pairs following *token*.

        Results are sorted descending by count.
        """
        counter = self.adjacency.get(token)
        if counter is None:
            return []
        return counter.most_common(top_k)

    def __len__(self) -> int:
        return len(self._tokens)


# ---------------------------------------------------------------------------
# TokenRecycler
# ---------------------------------------------------------------------------


class TokenRecycler:
    """Token Recycling speculative decoding draft generator.

    Maintains an internal :class:`RecyclingBuffer` and exposes:

    * ``update(token_ids)`` — ingest newly generated tokens.
    * ``draft(context_ids, n_tokens)`` — produce a single draft sequence.
    * ``batch_draft(context_ids, n_candidates, n_tokens)`` — produce multiple
      diverse draft sequences for tree-based verification.
    """

    def __init__(
        self,
        vocab_size: int,
        max_draft_tokens: int = 5,
        buffer_size: int = 512,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_draft_tokens = max_draft_tokens
        self.buffer_size = buffer_size

        # Public statistics tables (also stored in buffer but mirrored here
        # for direct access as specified in the API)
        self.adjacency: dict[int, Counter] = {}
        self.bigram_table: dict[tuple[int, int], Counter] = {}
        self.buffer: deque = deque(maxlen=buffer_size)

        # Internal RecyclingBuffer (keeps consistent state)
        self._rbuf: RecyclingBuffer = RecyclingBuffer(max_size=buffer_size)

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, token_ids: Tensor) -> None:
        """Ingest new tokens and update internal transition tables.

        Parameters
        ----------
        token_ids:
            1-D int tensor of newly generated token ids.
        """
        ids: list[int] = token_ids.view(-1).tolist()
        for tok in ids:
            self._rbuf.add(tok)
            self.buffer.append(tok)

        # Keep public mirrors in sync with RecyclingBuffer internals
        self.adjacency = self._rbuf.adjacency
        self.bigram_table = self._rbuf.bigram_table

    # ------------------------------------------------------------------
    # draft
    # ------------------------------------------------------------------

    def draft(self, context_ids: Tensor, n_tokens: int) -> Tensor | None:
        """Generate a single draft sequence of length *n_tokens*.

        Strategy
        --------
        1. Try bigram lookup with the last two context tokens.
        2. Fall back to unigram lookup with the last context token.
        3. Return ``None`` if the buffer is empty or no candidates exist.

        Parameters
        ----------
        context_ids:
            1-D int tensor of the current context (recent tokens).
        n_tokens:
            Number of draft tokens to produce.

        Returns
        -------
        Tensor of shape ``(n_tokens,)`` or ``None``.
        """
        if len(self._rbuf) == 0:
            return None

        ctx: list[int] = context_ids.view(-1).tolist()
        draft_ids: list[int] = []

        # Seed from context; we'll extend greedily using the recycling buffer
        # The "current" last two tokens come from context + already drafted
        def _last_two(base: list[int], extra: list[int]) -> tuple[int | None, int | None]:
            combined = base + extra
            if len(combined) >= 2:
                return combined[-2], combined[-1]
            elif len(combined) == 1:
                return None, combined[-1]
            return None, None

        for _ in range(n_tokens):
            t_prev, t_last = _last_two(ctx, draft_ids)

            next_tok: int | None = None

            # Bigram lookup
            if t_prev is not None and t_last is not None:
                candidates = self._rbuf.get_bigram_candidates(t_prev, t_last, top_k=1)
                if candidates:
                    next_tok = candidates[0][0]

            # Unigram fallback
            if next_tok is None and t_last is not None:
                candidates = self._rbuf.get_unigram_candidates(t_last, top_k=1)
                if candidates:
                    next_tok = candidates[0][0]

            if next_tok is None:
                # Cannot extend further — break and zero-pad
                break

            draft_ids.append(next_tok)

        if not draft_ids:
            return None

        # Pad / truncate to exactly n_tokens
        if len(draft_ids) < n_tokens:
            draft_ids += [0] * (n_tokens - len(draft_ids))

        return torch.tensor(draft_ids[:n_tokens], dtype=torch.long)

    # ------------------------------------------------------------------
    # batch_draft
    # ------------------------------------------------------------------

    def batch_draft(
        self,
        context_ids: Tensor,
        n_candidates: int = 4,
        n_tokens: int = 5,
    ) -> Tensor:
        """Generate *n_candidates* diverse draft sequences of length *n_tokens*.

        Each candidate explores a different branch of the recycling trie by
        iterating over the top-k next-token options at the first draft position.

        Returns
        -------
        Tensor of shape ``(n_candidates, n_tokens)``.
        """
        ctx: list[int] = context_ids.view(-1).tolist()

        # Collect multiple options for the first draft token
        t_prev: int | None = ctx[-2] if len(ctx) >= 2 else None
        t_last: int | None = ctx[-1] if len(ctx) >= 1 else None

        first_candidates: list[int] = []
        if t_prev is not None and t_last is not None:
            bigram_cands = self._rbuf.get_bigram_candidates(t_prev, t_last, top_k=n_candidates)
            first_candidates = [tok for tok, _ in bigram_cands]

        if len(first_candidates) < n_candidates and t_last is not None:
            uni_cands = self._rbuf.get_unigram_candidates(t_last, top_k=n_candidates)
            seen = set(first_candidates)
            for tok, _ in uni_cands:
                if tok not in seen:
                    first_candidates.append(tok)
                if len(first_candidates) >= n_candidates:
                    break

        # Pad first_candidates with 0 if still not enough
        while len(first_candidates) < n_candidates:
            first_candidates.append(0)

        rows: list[list[int]] = []
        for first_tok in first_candidates[:n_candidates]:
            # Build rest of the sequence greedily from this branching point
            seq: list[int] = [first_tok]
            extended_ctx = ctx + seq

            for _ in range(n_tokens - 1):
                t2 = extended_ctx[-1]
                t1 = extended_ctx[-2] if len(extended_ctx) >= 2 else None

                next_tok: int | None = None
                if t1 is not None:
                    bc = self._rbuf.get_bigram_candidates(t1, t2, top_k=1)
                    if bc:
                        next_tok = bc[0][0]
                if next_tok is None:
                    uc = self._rbuf.get_unigram_candidates(t2, top_k=1)
                    if uc:
                        next_tok = uc[0][0]
                if next_tok is None:
                    next_tok = 0

                seq.append(next_tok)
                extended_ctx.append(next_tok)

            rows.append(seq[:n_tokens])

        return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------------
# build_recycling_draft_tree
# ---------------------------------------------------------------------------


def build_recycling_draft_tree(
    buffer: RecyclingBuffer,
    context: list[int],
    depth: int = 3,
) -> dict:
    """Build a tree of draft token candidates from *buffer*.

    Starting from the tokens in *context*, greedily expand the most likely
    continuations up to *depth* levels deep.

    Parameters
    ----------
    buffer:
        A :class:`RecyclingBuffer` with transition statistics.
    context:
        List of recent token ids (the current context).
    depth:
        Maximum depth of the draft tree.

    Returns
    -------
    Nested dict ``{token_id: {token_id: {...}}}`` representing the draft tree.
    Each key is a candidate next token; the value is a sub-tree of the same
    form (empty dict at leaf nodes / max depth).
    """
    if depth <= 0 or not context:
        return {}

    t_last = context[-1]
    t_prev = context[-2] if len(context) >= 2 else None

    # Collect candidate next tokens
    candidates: list[int] = []
    if t_prev is not None:
        bigram_cands = buffer.get_bigram_candidates(t_prev, t_last, top_k=5)
        candidates = [tok for tok, _ in bigram_cands]

    if not candidates:
        uni_cands = buffer.get_unigram_candidates(t_last, top_k=5)
        candidates = [tok for tok, _ in uni_cands]

    tree: dict = {}
    for tok in candidates:
        # Recurse: extend context with this candidate
        subtree = build_recycling_draft_tree(buffer, context + [tok], depth - 1)
        tree[tok] = subtree

    return tree
