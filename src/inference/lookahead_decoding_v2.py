"""Lookahead Decoding v2: parallel n-gram speculation without a draft model.

Based on Fu et al. 2024, "Break the Sequential Dependency of LLM Inference
Using Lookahead Decoding". Maintains a lookahead window of parallel Jacobi
branches and an n-gram pool of candidates verified against the base model.

Key idea:
  1. Run W parallel Jacobi iterations to generate speculative tokens.
  2. Collect produced n-grams into a recency-ordered pool.
  3. At each LLM call, query the pool for prefix-matching candidates, verify
     them against greedy autoregressive continuations, and accept the longest
     matching prefix — reducing average per-token LLM calls below 1.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import LongTensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LookaheadConfig:
    """Configuration for LookaheadDecoder.

    Attributes:
        window_size:    W — number of parallel lookahead branches.
        ngram_size:     N — n-gram length stored in / queried from the pool.
        pool_size:      Maximum number of n-grams kept in the pool (LRU eviction).
        guess_set_size: Top-K candidates from the pool verified per decoding step.
    """

    window_size: int = 5
    ngram_size: int = 3
    pool_size: int = 64
    guess_set_size: int = 5


# ---------------------------------------------------------------------------
# NGramPool
# ---------------------------------------------------------------------------


class NGramPool:
    """Recency-ordered pool of n-grams used as speculative candidates.

    Internally uses an :class:`collections.OrderedDict` keyed by the tuple
    representation of each n-gram.  The most-recently-added entry is at the
    *end* of the dict so that ``query`` can iterate in reverse to return
    candidates by recency without sorting.

    LRU eviction: when the pool exceeds ``pool_size``, the *oldest* entry
    (front of the OrderedDict) is removed.
    """

    def __init__(self, ngram_size: int, pool_size: int) -> None:
        if ngram_size < 2:
            raise ValueError(
                "ngram_size must be >= 2 (need at least a 1-token prefix + 1 continuation)"
            )
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self.ngram_size = ngram_size
        self.pool_size = pool_size
        # OrderedDict preserves insertion order; oldest → newest
        self._store: OrderedDict[tuple[int, ...], None] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, tokens: list[int]) -> None:
        """Extract every n-gram from *tokens* and add them to the pool.

        Duplicate n-grams are moved to the end (most-recent position).
        When the pool is full the oldest entry is evicted.

        Args:
            tokens: A token sequence of length >= ngram_size.  If shorter,
                    no n-grams are added.
        """
        n = self.ngram_size
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i : i + n])
            # Move to end if already present (update recency)
            if gram in self._store:
                self._store.move_to_end(gram)
            else:
                self._store[gram] = None
                # Evict oldest when over capacity
                if len(self._store) > self.pool_size:
                    self._store.popitem(last=False)

    def query(self, context: list[int], k: int = 5) -> list[list[int]]:
        """Return up to *k* n-grams whose (n-1)-prefix matches context's tail.

        The prefix match is against the last ``ngram_size - 1`` tokens of
        *context*.  Candidates are returned most-recently-added first.

        Args:
            context: Current token context.  Must have at least ``ngram_size - 1``
                     tokens; otherwise returns an empty list.
            k:       Maximum number of candidates to return.

        Returns:
            A list of at most *k* token lists, each of length ``ngram_size``.
        """
        prefix_len = self.ngram_size - 1
        if len(context) < prefix_len:
            return []
        prefix = tuple(context[-prefix_len:]) if prefix_len > 0 else ()
        results: list[list[int]] = []
        # Iterate newest → oldest
        for gram in reversed(self._store):
            if gram[:prefix_len] == prefix:
                results.append(list(gram))
                if len(results) >= k:
                    break
        return results

    def size(self) -> int:
        """Return the current number of n-grams in the pool."""
        return len(self._store)

    def clear(self) -> None:
        """Remove all n-grams from the pool."""
        self._store.clear()


# ---------------------------------------------------------------------------
# LookaheadVerifier
# ---------------------------------------------------------------------------


class LookaheadVerifier:
    """Verifies speculative n-gram candidates against ground-truth tokens.

    The verifier is stateless; it only depends on ``vocab_size`` for a
    lightweight sanity guard (tokens outside [0, vocab_size) are treated as
    mismatches).
    """

    def __init__(self, vocab_size: int) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab_size = vocab_size

    def verify_ngram(
        self,
        candidate_ngram: list[int],
        ground_truth_tokens: list[int],
    ) -> int:
        """Return the length of the longest matching prefix.

        Comparison stops at the first mismatch or when either sequence is
        exhausted.

        Examples::

            verify_ngram([A, B, C], [A, B, C])  → 3
            verify_ngram([A, B, C], [A, B, X])  → 2
            verify_ngram([A, B, C], [X, B, C])  → 0

        Args:
            candidate_ngram:    Speculative token sequence.
            ground_truth_tokens: Greedy AR tokens to compare against.

        Returns:
            Number of leading tokens that match (0 … min(len(candidate), len(truth))).
        """
        n_match = 0
        for c, g in zip(candidate_ngram, ground_truth_tokens):
            if c == g:
                n_match += 1
            else:
                break
        return n_match

    def select_best_candidate(
        self,
        candidates: list[list[int]],
        ground_truth: list[int],
    ) -> tuple[list[int], int]:
        """Verify all candidates; return the one with the most accepted tokens.

        Ties are broken by taking the first candidate in *candidates* with the
        maximum acceptance length (preserves recency ordering from the pool).

        Args:
            candidates:   List of speculative n-gram token sequences.
            ground_truth: Ground-truth continuation produced by greedy AR.

        Returns:
            A tuple ``(best_ngram, n_accepted)`` where *best_ngram* is the
            winning candidate and *n_accepted* is how many tokens to accept.
            If *candidates* is empty, returns ``([], 0)``.
        """
        if not candidates:
            return [], 0

        best_ngram: list[int] = []
        best_n: int = 0
        for cand in candidates:
            n = self.verify_ngram(cand, ground_truth)
            if n > best_n:
                best_n = n
                best_ngram = cand
        return best_ngram, best_n


# ---------------------------------------------------------------------------
# LookaheadDecoder
# ---------------------------------------------------------------------------


class LookaheadDecoder:
    """Lookahead decoder that uses an NGramPool to speculate continuations.

    ``model_fn`` must have the signature::

        model_fn(input_ids: LongTensor[1, T]) -> LongTensor[1, T, V]

    where V is the vocabulary size.

    The decoder does **not** maintain any draft model; speculation comes
    entirely from the n-gram pool built up during generation.

    Args:
        model_fn:   Callable mapping ``(1, T)`` ids to ``(1, T, V)`` logits.
        vocab_size: Vocabulary size (must match the model's output dimension).
        config:     :class:`LookaheadConfig` controlling window/pool parameters.
    """

    def __init__(
        self,
        model_fn: Callable[[LongTensor], LongTensor],
        vocab_size: int,
        config: LookaheadConfig,
    ) -> None:
        self.model_fn = model_fn
        self.vocab_size = vocab_size
        self.config = config
        self.verifier = LookaheadVerifier(vocab_size)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def generate_step(
        self,
        context_ids: LongTensor,
        pool: NGramPool,
    ) -> tuple[list[int], int]:
        """Perform one lookahead decoding step.

        Algorithm:
          1. Query the pool for up to ``config.guess_set_size`` candidates
             whose (N-1)-prefix matches the tail of *context_ids*.
          2. Run the model on *context_ids* to obtain next-token logits →
             greedy argmax → base token.
          3. Build a ground-truth sequence of length ``config.ngram_size`` by
             greedily decoding ``ngram_size - 1`` additional tokens (each
             conditioned on the growing context).  This simulates what the
             model *would* produce without speculation.
          4. Verify candidates.  If the best match accepts ≥ 1 token, return
             those tokens; otherwise fall back to the single base token.
          5. Add the accepted token sequence (plus one lookahead token beyond
             the accepted prefix when a candidate matched) to the pool.

        Args:
            context_ids: ``(1, T)`` long tensor — current context.
            pool:        The shared :class:`NGramPool` to query and update.

        Returns:
            ``(accepted_tokens, n_accepted)`` where *accepted_tokens* is a list
            of new token ids (length ≥ 1) and *n_accepted* is the count.
        """
        cfg = self.config
        context_list: list[int] = context_ids[0].tolist()

        # --- Step 1: query pool for candidate n-grams ---
        candidates = pool.query(context_list, k=cfg.guess_set_size)

        # --- Step 2: run model, get base token ---
        with torch.no_grad():
            logits = self.model_fn(context_ids)  # (1, T, V)
        next_token_logits = logits[0, -1, :]  # (V,)
        base_token = int(next_token_logits.argmax().item())

        # --- Step 3: build greedy ground-truth of length ngram_size ---
        ground_truth: list[int] = [base_token]
        cur_ids = torch.cat(
            [context_ids, torch.tensor([[base_token]], dtype=torch.long)],
            dim=1,
        )
        for _ in range(cfg.ngram_size - 1):
            with torch.no_grad():
                gt_logits = self.model_fn(cur_ids)
            gt_next = int(gt_logits[0, -1, :].argmax().item())
            ground_truth.append(gt_next)
            cur_ids = torch.cat(
                [cur_ids, torch.tensor([[gt_next]], dtype=torch.long)],
                dim=1,
            )

        # --- Step 4: verify candidates ---
        best_ngram, n_accepted = self.verifier.select_best_candidate(candidates, ground_truth)

        # --- Step 5: choose accepted tokens & update pool ---
        if n_accepted >= 1:
            accepted_tokens = best_ngram[:n_accepted]
        else:
            accepted_tokens = [base_token]
            n_accepted = 1

        # Add ground-truth n-gram to pool for future reuse
        pool.add(ground_truth)

        return accepted_tokens, n_accepted

    # ------------------------------------------------------------------
    # Full generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: LongTensor,
        max_new_tokens: int,
    ) -> LongTensor:
        """Generate up to *max_new_tokens* new tokens via lookahead decoding.

        Args:
            prompt_ids:     ``(1, T)`` long tensor — prompt token ids.
            max_new_tokens: Maximum number of new tokens to append.

        Returns:
            ``(1, T + n_new)`` long tensor with prompt + generated tokens.
        """
        pool = NGramPool(
            ngram_size=self.config.ngram_size,
            pool_size=self.config.pool_size,
        )

        # Seed pool with prompt n-grams so the first query can find candidates
        prompt_list: list[int] = prompt_ids[0].tolist()
        if len(prompt_list) >= self.config.ngram_size:
            pool.add(prompt_list)

        context = prompt_ids  # (1, T)
        n_generated = 0

        while n_generated < max_new_tokens:
            remaining = max_new_tokens - n_generated
            accepted_tokens, n_accepted = self.generate_step(context, pool)

            # Clip to remaining budget
            accepted_tokens = accepted_tokens[:remaining]
            n_accepted = len(accepted_tokens)

            new_ids = torch.tensor([accepted_tokens], dtype=torch.long)
            context = torch.cat([context, new_ids], dim=1)
            n_generated += n_accepted

        return context
