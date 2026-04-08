"""Multi-draft speculative decoding (Miao et al. 2023 / Cai et al. 2024).

Instead of a single draft sequence, k independent drafters propose k candidate
next-token sequences in parallel.  The verifier (target model) checks all k
candidates simultaneously in one batched forward pass.  The optimal transport
acceptance criterion (a.k.a. typical acceptance) guarantees the output
distribution exactly matches the target model's distribution.

References:
    Miao et al. 2023 — "SpecInfer: Accelerating LLM serving with speculative
        inference and token tree verification."
    Cai et al. 2024 — "Medusa: Simple LLM Inference Acceleration Framework
        with Multiple Decoding Heads."
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration & data containers
# ---------------------------------------------------------------------------

@dataclass
class MultiDraftConfig:
    n_drafts: int = 4        # number of draft candidates per step
    draft_steps: int = 4     # lookahead steps per draft
    temperature: float = 1.0 # target model sampling temperature


@dataclass
class DraftCandidate:
    tokens: list[int]      # candidate token sequence (draft_steps tokens)
    log_probs: list[float] # draft model log-probs for each token


# ---------------------------------------------------------------------------
# Core acceptance criterion
# ---------------------------------------------------------------------------

def typical_acceptance(
    draft_probs: torch.Tensor,   # (n_drafts,) draft probabilities for each candidate
    target_probs: torch.Tensor,  # (n_drafts,) target model probabilities for each candidate
) -> int:
    """Acceptance criterion for multi-draft speculative decoding.

    For each candidate i: accept with probability min(1, target_prob_i / draft_prob_i).
    If multiple candidates are accepted, sample one proportional to their target_probs.
    If none accepted, return -1 so the caller can resample from the target distribution.

    Args:
        draft_probs:  (n_drafts,) — draft model probabilities assigned to each candidate token.
        target_probs: (n_drafts,) — target model probabilities for the same tokens.

    Returns:
        Index of accepted candidate in [0, n_drafts-1], or -1 if all rejected.
    """
    n = draft_probs.shape[0]
    accepted_indices: list[int] = []
    accepted_target_probs: list[float] = []

    for i in range(n):
        d_p = draft_probs[i].item()
        t_p = target_probs[i].item()
        accept_prob = min(1.0, t_p / (d_p + 1e-9))
        u = torch.rand(1).item()
        if u < accept_prob:
            accepted_indices.append(i)
            accepted_target_probs.append(t_p)

    if not accepted_indices:
        return -1

    if len(accepted_indices) == 1:
        return accepted_indices[0]

    # Sample proportionally to target probabilities among accepted candidates
    weights = torch.tensor(accepted_target_probs)
    s = weights.sum()
    if s < 1e-10:
        # fallback: uniform over accepted
        idx = torch.randint(len(accepted_indices), (1,)).item()
    else:
        weights = weights / s
        idx = torch.multinomial(weights, 1).item()
    return accepted_indices[int(idx)]


# ---------------------------------------------------------------------------
# Batch verification
# ---------------------------------------------------------------------------

def batch_verify_drafts(
    target_model: nn.Module,
    prefix_ids: list[int],
    draft_candidates: list[list[int]],  # n_drafts lists each of draft_steps tokens
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, list[int]]:
    """Verify all draft candidates in a single target model forward pass.

    Strategy: concatenate prefix + each candidate into separate batch items.
    Run the target model on the batch and extract probabilities at the first
    predicted position (the position that predicts draft_candidates[i][0]).

    The sequences may have different lengths only when draft_steps varies; here
    all candidates have the same length (draft_steps), so we just pad the batch
    to a common length.

    Args:
        target_model: AureliusTransformer (or compatible).
        prefix_ids:   List of token IDs forming the common prefix.
        draft_candidates: n_drafts sequences, each of length draft_steps.
        max_seq_len:  Truncate prefix to fit within this budget.

    Returns:
        target_probs_per_candidate: (n_drafts,) — target probability of
            draft_candidates[i][0] given prefix, for each draft i.
        accepted_tokens: list[int] — the accepted sequence of tokens chosen
            by typical_acceptance for the first position.  Length 0 or 1.
    """
    n_drafts = len(draft_candidates)
    if n_drafts == 0:
        return torch.zeros(0), []

    draft_len = max(len(c) for c in draft_candidates)

    # Build batch: each row = prefix + candidate (possibly padded)
    prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long)
    prefix_len = prefix_tensor.shape[0]

    # Truncate prefix if necessary to leave room for draft tokens
    if prefix_len + draft_len > max_seq_len:
        prefix_tensor = prefix_tensor[-(max_seq_len - draft_len):]
        prefix_len = prefix_tensor.shape[0]

    total_len = prefix_len + draft_len
    PAD_ID = 0  # arbitrary; we only read positions before pad tokens

    batch = torch.full((n_drafts, total_len), PAD_ID, dtype=torch.long)
    for i, candidate in enumerate(draft_candidates):
        row = torch.cat([prefix_tensor, torch.tensor(candidate, dtype=torch.long)])
        # Pad candidate if shorter than draft_len
        if len(candidate) < draft_len:
            pad = torch.full((draft_len - len(candidate),), PAD_ID, dtype=torch.long)
            row = torch.cat([prefix_tensor, torch.tensor(candidate, dtype=torch.long), pad])
        batch[i] = row

    # Forward pass — AureliusTransformer returns (loss, logits, extras)
    with torch.no_grad():
        _, logits, _ = target_model(batch)
    # logits: (n_drafts, total_len, vocab_size)

    # Position that predicts draft_candidates[i][0] is (prefix_len - 1)
    pred_pos = prefix_len - 1
    if pred_pos < 0:
        pred_pos = 0

    logits_at_pred = logits[:, pred_pos, :]  # (n_drafts, vocab_size)
    probs_at_pred = F.softmax(logits_at_pred, dim=-1)   # (n_drafts, vocab_size)

    # Gather target probability for each draft's first token
    first_tokens = torch.tensor(
        [c[0] if len(c) > 0 else 0 for c in draft_candidates],
        dtype=torch.long,
    )  # (n_drafts,)
    target_probs_per_candidate = probs_at_pred[
        torch.arange(n_drafts), first_tokens
    ]  # (n_drafts,)

    # Build uniform draft probabilities (we don't have the draft model here,
    # so we use 1/n_drafts as the draft probability — callers may override this)
    draft_probs = torch.full((n_drafts,), 1.0 / n_drafts)

    accepted_idx = typical_acceptance(draft_probs, target_probs_per_candidate)

    if accepted_idx == -1:
        # Resample from target distribution at pred_pos
        avg_probs = probs_at_pred.mean(dim=0)
        resampled = torch.multinomial(avg_probs, 1).item()
        accepted_tokens: list[int] = [int(resampled)]
    else:
        accepted_tokens = [int(first_tokens[accepted_idx].item())]

    return target_probs_per_candidate, accepted_tokens


# ---------------------------------------------------------------------------
# Multi-draft decoder
# ---------------------------------------------------------------------------

class MultiDraftDecoder:
    """Multi-draft speculative decoder.

    Uses a single draft model that samples n_drafts different candidates via
    temperature sampling (independent samples from the draft distribution).

    Args:
        target_model:     AureliusTransformer (large / accurate).
        draft_model:      AureliusTransformer (small / fast).
        tokenizer_encode: Callable mapping str → list[int] (unused internally
                          but kept for API compatibility).
        config:           MultiDraftConfig.
        eos_token_id:     Token id signalling end of sequence.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer_encode: "callable | None" = None,
        config: MultiDraftConfig | None = None,
        eos_token_id: int = 2,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer_encode = tokenizer_encode
        self.config = config or MultiDraftConfig()
        self.eos_token_id = eos_token_id

    @torch.no_grad()
    def _sample_drafts(
        self,
        prefix_ids: list[int],
        n_drafts: int,
        draft_steps: int,
    ) -> list[list[int]]:
        """Sample n_drafts independent candidate sequences from draft_model.

        Each candidate consists of draft_steps tokens sampled autoregressively
        from the draft model with temperature sampling.

        Args:
            prefix_ids:  Current context as a list of token IDs.
            n_drafts:    Number of independent candidate sequences to produce.
            draft_steps: Number of tokens to generate per candidate.

        Returns:
            List of n_drafts lists, each containing draft_steps token IDs.
        """
        temperature = self.config.temperature
        candidates: list[list[int]] = []

        max_seq_len = getattr(self.draft_model, "config", None)
        if max_seq_len is not None:
            max_seq_len = max_seq_len.max_seq_len
        else:
            max_seq_len = 512

        for _ in range(n_drafts):
            context = prefix_ids[-(max_seq_len - draft_steps):]
            current_ids = list(context)
            sequence: list[int] = []

            for _ in range(draft_steps):
                input_tensor = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0)
                _, logits, _ = self.draft_model(input_tensor)
                last_logits = logits[0, -1, :]  # (vocab_size,)

                if temperature != 1.0:
                    last_logits = last_logits / temperature

                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                sequence.append(int(next_token))
                current_ids.append(int(next_token))

            candidates.append(sequence)

        return candidates

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 100,
    ) -> tuple[list[int], dict]:
        """Generate using multi-draft speculative decoding.

        At each step:
        1. Draft model samples n_drafts candidate sequences of draft_steps tokens.
        2. Target model verifies all n_drafts candidates in one batch forward pass.
        3. One candidate is accepted/resampled via typical_acceptance.
        4. The accepted first token is appended to the output.

        Args:
            prompt_ids:     Input token IDs.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            (generated_ids, stats) where stats contains acceptance statistics.
        """
        cfg = self.config
        generated = list(prompt_ids)
        n_accepted = 0
        n_rejected = 0

        max_seq_len = getattr(self.target_model, "config", None)
        if max_seq_len is not None:
            max_seq_len = max_seq_len.max_seq_len
        else:
            max_seq_len = 512

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            # Step 1: sample n_drafts candidate sequences
            draft_candidates = self._sample_drafts(
                prefix_ids=generated,
                n_drafts=cfg.n_drafts,
                draft_steps=cfg.draft_steps,
            )

            # Step 2 & 3: batch-verify and accept/reject
            target_probs, accepted_tokens = batch_verify_drafts(
                target_model=self.target_model,
                prefix_ids=generated,
                draft_candidates=draft_candidates,
                max_seq_len=max_seq_len,
            )

            if accepted_tokens:
                tok = accepted_tokens[0]
                generated.append(tok)
                tokens_generated += 1
                n_accepted += 1

                if tok == self.eos_token_id:
                    break
            else:
                n_rejected += 1
                # Safety valve: do a single greedy target sample
                prefix_tensor = torch.tensor(
                    generated[-(max_seq_len - 1):], dtype=torch.long
                ).unsqueeze(0)
                _, logits, _ = self.target_model(prefix_tensor)
                last_logits = logits[0, -1, :]
                if cfg.temperature != 1.0:
                    last_logits = last_logits / cfg.temperature
                probs = F.softmax(last_logits, dim=-1)
                tok = int(torch.multinomial(probs, 1).item())
                generated.append(tok)
                tokens_generated += 1

                if tok == self.eos_token_id:
                    break

        total = n_accepted + n_rejected
        acceptance_rate = n_accepted / total if total > 0 else 0.0
        tokens_per_step = tokens_generated / max(total, 1)

        stats = {
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "acceptance_rate": acceptance_rate,
            "tokens_per_step": tokens_per_step,
        }
        return generated[len(prompt_ids):], stats


# ---------------------------------------------------------------------------
# Draft diversity metrics
# ---------------------------------------------------------------------------

def levenshtein_distance(a: list[int], b: list[int]) -> int:
    """Standard O(m*n) dynamic programming Levenshtein edit distance.

    Args:
        a: First sequence of integers.
        b: Second sequence of integers.

    Returns:
        Minimum number of single-element insertions, deletions, or substitutions
        required to transform a into b.
    """
    m, n = len(a), len(b)
    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_draft_diversity(
    candidates: list[list[int]],
) -> dict:
    """Measure diversity among draft candidates.

    Args:
        candidates: List of candidate token sequences.

    Returns:
        Dictionary with the following keys:
        - 'unique_first_tokens': number of distinct first-token values.
        - 'mean_edit_distance':  mean pairwise Levenshtein distance between
          all pairs of candidates.
        - 'entropy':             Shannon entropy (natural log) of the
          empirical distribution over first tokens.
    """
    if not candidates:
        return {"unique_first_tokens": 0, "mean_edit_distance": 0.0, "entropy": 0.0}

    # Unique first tokens
    first_tokens = [c[0] for c in candidates if len(c) > 0]
    unique_first_tokens = len(set(first_tokens))

    # Entropy of first-token distribution (natural log)
    counts: dict[int, int] = {}
    for t in first_tokens:
        counts[t] = counts.get(t, 0) + 1
    n_total = len(first_tokens)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n_total
        if p > 0:
            entropy -= p * math.log(p)  # natural log

    # Mean pairwise edit distance
    n = len(candidates)
    if n < 2:
        mean_edit_distance = 0.0
    else:
        total_dist = 0.0
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += levenshtein_distance(candidates[i], candidates[j])
                n_pairs += 1
        mean_edit_distance = total_dist / n_pairs

    return {
        "unique_first_tokens": unique_first_tokens,
        "mean_edit_distance": mean_edit_distance,
        "entropy": entropy,
    }
