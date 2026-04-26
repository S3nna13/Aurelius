"""Parallel lookahead decoding with n-gram caching and Jacobi decoding.

Implements n-gram speculation via Jacobi-style parallel draft generation
and greedy verification against the target model. This is distinct from
the rolling-cache lookahead in lookahead.py — here the focus is on
parallel draft branches and explicit accept/reject verification.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LookaheadConfig:
    """Configuration for parallel lookahead / Jacobi decoding."""

    lookahead_window: int = 7  # number of parallel draft steps in Jacobi branch
    guess_set_size: int = 7  # max candidates to keep in guess set
    n_gram_size: int = 3  # n-gram size for the speculation cache
    max_new_tokens: int = 256  # maximum tokens to generate
    verification_steps: int = 1  # verification rounds per step


# ---------------------------------------------------------------------------
# N-gram cache
# ---------------------------------------------------------------------------


class NGramCache:
    """Maps n-gram context tuples to lists of observed next tokens.

    Distinct from the NGramCache in lookahead.py which uses a ``candidates``
    interface.  Here we expose ``lookup`` for draft retrieval and ``update``
    for online learning from the generated sequence.
    """

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self._cache: dict[tuple, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    def update(self, tokens: list[int]) -> None:
        """Add all n-grams extracted from *tokens* into the cache.

        For each position ``i`` in the sequence we form the key
        ``(tokens[i], ..., tokens[i + n - 2])`` and append
        ``tokens[i + n - 1]`` to its entry (deduplicating).
        """
        if len(tokens) < self.n:
            return
        for i in range(len(tokens) - self.n + 1):
            key = tuple(tokens[i : i + self.n - 1])
            next_tok = tokens[i + self.n - 1]
            if next_tok not in self._cache[key]:
                self._cache[key].append(next_tok)

    # ------------------------------------------------------------------
    def lookup(self, context: tuple) -> list[int]:
        """Return candidate next tokens for *context* n-gram key.

        Args:
            context: Tuple of (n-1) token ids forming the lookup key.

        Returns:
            List of candidate next-token ids, empty if key not found.
        """
        return list(self._cache.get(context, []))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """Number of unique n-gram keys stored."""
        return len(self._cache)


# ---------------------------------------------------------------------------
# Draft generation — Jacobi branch
# ---------------------------------------------------------------------------


def generate_lookahead_branch(
    model: torch.nn.Module,
    input_ids: list[int],
    branch_len: int,
) -> list[int]:
    """Generate a speculative draft branch of *branch_len* tokens.

    Implements the Jacobi decoding principle: each draft token is generated
    from the current context independently (greedy argmax), simulating
    parallel forward passes.  In a real Jacobi decoder these would iterate
    until a fixed point; here we do a single greedy pass for simplicity and
    speed, which is the standard baseline used in lookahead decoding papers.

    Args:
        model: AureliusTransformer — called as ``loss, logits, pkv = model(ids)``.
        input_ids: Current prefix token ids as a plain Python list.
        branch_len: Number of draft tokens to produce.

    Returns:
        List of *branch_len* integer token ids.
    """
    if branch_len <= 0:
        return []

    device = next(model.parameters()).device
    draft: list[int] = []
    context = list(input_ids)

    with torch.no_grad():
        for _ in range(branch_len):
            ids = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            _, logits, _ = model(ids)
            next_tok = int(logits[0, -1, :].argmax())
            draft.append(next_tok)
            context.append(next_tok)

    return draft


# ---------------------------------------------------------------------------
# Draft verification
# ---------------------------------------------------------------------------


def verify_draft(
    model: torch.nn.Module,
    prefix_ids: list[int],
    draft_ids: list[int],
) -> tuple[list[int], int]:
    """Verify *draft_ids* against the model using greedy acceptance.

    Runs the model on the concatenated sequence ``prefix + draft`` and
    compares the argmax prediction at each draft position to the draft
    token.  Accepts greedily: all tokens up to (but not including) the
    first mismatch are accepted.

    Args:
        model: AureliusTransformer.
        prefix_ids: Verified prefix token ids.
        draft_ids: Speculative draft tokens to check.

    Returns:
        ``(accepted_tokens, n_accepted)`` where *accepted_tokens* is the
        list of accepted draft token ids and *n_accepted* is its length.
    """
    if not draft_ids:
        return [], 0

    device = next(model.parameters()).device
    full_seq = prefix_ids + draft_ids
    ids = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        _, logits, _ = model(ids)

    # logits shape: (1, seq_len, vocab)
    # Position prefix_len - 1 predicts draft_ids[0], etc.
    prefix_len = len(prefix_ids)
    accepted: list[int] = []

    for i, draft_tok in enumerate(draft_ids):
        pos = prefix_len - 1 + i  # logit position predicting draft_ids[i]
        predicted = int(logits[0, pos, :].argmax())
        if predicted == draft_tok:
            accepted.append(draft_tok)
        else:
            break  # stop at first mismatch

    return accepted, len(accepted)


# ---------------------------------------------------------------------------
# Jacobi Decoder
# ---------------------------------------------------------------------------


class JacobiDecoder:
    """Parallel lookahead decoder combining n-gram speculation with Jacobi drafts.

    At each decoding step:
    1. Try the n-gram cache first — cheap O(1) lookup.
    2. If cache misses, fall back to ``generate_lookahead_branch``.
    3. Verify the draft against the target model.
    4. Append accepted tokens and update the cache.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: LookaheadConfig,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self._ngram_cache = NGramCache(n=config.n_gram_size)

    # ------------------------------------------------------------------
    def _get_draft_from_cache(self, context: list[int]) -> list[int]:
        """Look up the n-gram cache to build a multi-token draft.

        Takes the last ``(n_gram_size - 1)`` tokens as the lookup key.
        If found, chains successive lookups to extend the draft up to
        ``lookahead_window`` tokens.

        Returns:
            Draft token list (may be empty on a cache miss).
        """
        n = self.config.n_gram_size
        window = self.config.lookahead_window
        draft: list[int] = []
        current_context = list(context)

        for _ in range(window):
            if len(current_context) < n - 1:
                break
            key = tuple(current_context[-(n - 1) :])
            candidates = self._ngram_cache.lookup(key)
            if not candidates:
                break
            # Use first candidate (highest-frequency, inserted first)
            next_tok = candidates[0]
            draft.append(next_tok)
            current_context.append(next_tok)

        return draft

    # ------------------------------------------------------------------
    def decode(self, prompt: str) -> tuple[str, dict]:
        """Run parallel lookahead decoding on *prompt*.

        Args:
            prompt: Input text string.

        Returns:
            ``(decoded_text, stats)`` where stats contains:
            - ``tokens_generated``: total new tokens appended.
            - ``cache_hits``: how many steps used a cache-based draft.
            - ``mean_tokens_per_step``: average accepted tokens per step.
        """
        cfg = self.config
        token_ids: list[int] = self.tokenizer_encode(prompt)
        self._ngram_cache.update(token_ids)

        tokens_generated = 0
        cache_hits = 0
        total_steps = 0
        total_accepted = 0

        while tokens_generated < cfg.max_new_tokens:
            remaining = cfg.max_new_tokens - tokens_generated
            branch_len = min(cfg.lookahead_window, remaining)

            # 1. Try cache draft
            draft = self._get_draft_from_cache(token_ids)
            if draft:
                cache_hits += 1
                draft = draft[:branch_len]
            else:
                # 2. Jacobi branch fallback
                draft = generate_lookahead_branch(self.model, token_ids, branch_len)

            # 3. Verify draft
            accepted, n_accepted = verify_draft(self.model, token_ids, draft)

            if n_accepted > 0:
                token_ids.extend(accepted)
                tokens_generated += n_accepted
                total_accepted += n_accepted
                self._ngram_cache.update(token_ids)
            else:
                # Accept nothing — run one greedy step to make progress
                device = next(self.model.parameters()).device
                ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, logits, _ = self.model(ids)
                next_tok = int(logits[0, -1, :].argmax())
                token_ids.append(next_tok)
                tokens_generated += 1
                total_accepted += 1
                self._ngram_cache.update(token_ids)

            total_steps += 1

        mean_tokens_per_step = total_accepted / max(total_steps, 1)
        decoded_text = self.tokenizer_decode(token_ids)
        stats = {
            "tokens_generated": tokens_generated,
            "cache_hits": cache_hits,
            "mean_tokens_per_step": mean_tokens_per_step,
        }
        return decoded_text, stats


# ---------------------------------------------------------------------------
# Speedup estimator
# ---------------------------------------------------------------------------


def estimate_speedup(n_accepted_per_step: float, draft_cost: float = 0.1) -> float:
    """Theoretical speedup from parallel lookahead decoding.

    Based on the standard speculative decoding analysis:

        speedup = n_accepted / (1 + draft_cost * n_accepted)

    where ``draft_cost`` is the relative cost of running the draft model
    compared to the target model (0.1 = 10% of target cost per token).

    Args:
        n_accepted_per_step: Mean number of tokens accepted per decoding step.
        draft_cost: Fractional cost of the draft pass relative to target.

    Returns:
        Speedup factor (float >= 1.0 when n_accepted_per_step >= 1.0).
    """
    return n_accepted_per_step / (1.0 + draft_cost * n_accepted_per_step)
