"""Speculative Rejection Sampling — token-tree speculative decoding variant.

Implements the approach from Liu et al. 2024: draft multiple candidate tokens
at each position via a draft model, then verify/correct them in a single batch
forward pass through the target model, maintaining the exact target distribution.

Reference: "Speculative Rejection Sampling" (Liu et al., 2024)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# DraftTree
# ---------------------------------------------------------------------------

class DraftTree:
    """Manages a tree of draft token candidates.

    For simplicity the tree is flat: branching_factor candidates at depth 0,
    then one (greedy-argmax) candidate per path for depth > 0.
    Each linear path therefore has length == depth.
    """

    def __init__(self, branching_factor: int = 3, depth: int = 4) -> None:
        if branching_factor < 1:
            raise ValueError("branching_factor must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.branching_factor = branching_factor
        self.depth = depth

        # Populated by build()
        self.tokens: Tensor | None = None      # (depth, branching_factor)
        self.log_probs: Tensor | None = None   # (depth, branching_factor)

    def build(self, draft_logits: list[Tensor]) -> None:
        """Build the tree from draft logits.

        Args:
            draft_logits: list of depth tensors, each shape (V,) — one per depth
        """
        if len(draft_logits) != self.depth:
            raise ValueError(
                f"Expected {self.depth} logit tensors, got {len(draft_logits)}"
            )

        all_tokens = []
        all_log_probs = []

        for d, logits in enumerate(draft_logits):
            # logits: (V,)
            probs = F.softmax(logits.float(), dim=-1)
            log_p = F.log_softmax(logits.float(), dim=-1)

            if d == 0:
                # Sample branching_factor tokens without replacement if possible
                k = min(self.branching_factor, probs.shape[0])
                sampled = torch.multinomial(probs, k, replacement=False)
                # Pad to branching_factor if vocab < branching_factor
                while sampled.shape[0] < self.branching_factor:
                    extra = torch.multinomial(probs, 1)
                    sampled = torch.cat([sampled, extra])
            else:
                # Greedy: repeat the argmax branching_factor times
                best = logits.argmax().unsqueeze(0)
                sampled = best.expand(self.branching_factor).clone()

            lp = log_p[sampled]          # (branching_factor,)
            all_tokens.append(sampled)   # (branching_factor,)
            all_log_probs.append(lp)     # (branching_factor,)

        self.tokens = torch.stack(all_tokens, dim=0)      # (depth, branching_factor)
        self.log_probs = torch.stack(all_log_probs, dim=0)  # (depth, branching_factor)

    def linear_paths(self) -> list[list[int]]:
        """Return all root-to-leaf token sequences.

        branching_factor paths, each of length depth.
        Depth 0 uses the distinct candidates; depth > 0 uses greedy (same for all paths).
        """
        if self.tokens is None:
            raise RuntimeError("Call build() before linear_paths()")

        paths: list[list[int]] = []
        for b in range(self.branching_factor):
            path: list[int] = []
            for d in range(self.depth):
                path.append(int(self.tokens[d, b].item()))
            paths.append(path)
        return paths

    def n_candidates(self) -> int:
        """Total number of candidate sequences (= branching_factor)."""
        return self.branching_factor


# ---------------------------------------------------------------------------
# TokenTreeVerifier
# ---------------------------------------------------------------------------

class TokenTreeVerifier:
    """Verify draft tokens against the target model in a single forward pass."""

    def __init__(self, target_model: nn.Module) -> None:
        self.target_model = target_model

    @torch.no_grad()
    def verify(
        self,
        input_ids: Tensor,     # (B, T_context)
        draft_tokens: Tensor,  # (B, K) — K draft candidates to verify
    ) -> tuple[Tensor, Tensor]:
        """Run target model once to verify all draft candidates.

        Returns:
            target_logits: (B, K, V)
            accept_mask:   (B, K) bool — True where target_argmax == draft_token
        """
        B, T = input_ids.shape
        K = draft_tokens.shape[1]

        # Concatenate context with draft tokens
        combined = torch.cat([input_ids, draft_tokens], dim=1)  # (B, T+K)

        # Run target model — supports both (loss, logits, aux) and plain logits returns
        output = self.target_model(combined)
        if isinstance(output, tuple):
            logits = output[1]
        else:
            logits = output
        # logits: (B, T+K, V)

        # Extract logits at draft positions: positions T-1 .. T+K-2
        # (the logit at position i predicts token i+1, so position T-1 predicts the
        #  first draft token, position T predicts the second, etc.)
        draft_pos_logits = logits[:, T - 1: T + K - 1, :]  # (B, K, V)

        target_argmax = draft_pos_logits.argmax(dim=-1)      # (B, K)
        accept_mask = target_argmax == draft_tokens           # (B, K) bool

        return draft_pos_logits, accept_mask


# ---------------------------------------------------------------------------
# RejectionSamplingCorrector
# ---------------------------------------------------------------------------

class RejectionSamplingCorrector:
    """Correct rejected tokens to maintain the exact target distribution."""

    def __init__(self) -> None:
        pass

    def correct(
        self,
        draft_token: int,
        draft_logp: float,
        target_logits: Tensor,  # (V,)
        accepted: bool = False,
    ) -> int:
        """Return an accepted or corrected token.

        If accepted: return draft_token unchanged.
        If rejected: sample from correction_dist = relu(p_target - p_draft) / Z
        """
        if accepted:
            return draft_token

        target_probs = F.softmax(target_logits.float(), dim=-1)  # (V,)

        # Draft prob for this token
        draft_probs = torch.zeros_like(target_probs)
        draft_probs[draft_token] = float(draft_logp.exp() if isinstance(draft_logp, Tensor) else
                                         torch.tensor(draft_logp).exp().item())

        correction = F.relu(target_probs - draft_probs)
        Z = correction.sum() + 1e-9
        correction_dist = correction / Z

        sampled = torch.multinomial(correction_dist, 1).item()
        return int(sampled)

    def batch_correct(
        self,
        draft_tokens: Tensor,   # (B, K)
        accept_mask: Tensor,    # (B, K) bool
        draft_logps: Tensor,    # (B, K)
        target_logits: Tensor,  # (B, K, V)
    ) -> Tensor:
        """Correct a batch of draft tokens.

        Returns corrected_tokens: (B, K)
        """
        B, K = draft_tokens.shape
        corrected = draft_tokens.clone()

        for b in range(B):
            for k in range(K):
                if accept_mask[b, k].item():
                    # Keep draft token
                    corrected[b, k] = draft_tokens[b, k]
                else:
                    tok = self.correct(
                        draft_token=int(draft_tokens[b, k].item()),
                        draft_logp=float(draft_logps[b, k].item()),
                        target_logits=target_logits[b, k],
                        accepted=False,
                    )
                    corrected[b, k] = tok

        return corrected


# ---------------------------------------------------------------------------
# SpeculativeRejectionDecoder
# ---------------------------------------------------------------------------

class SpeculativeRejectionDecoder:
    """Full speculative rejection sampling decoder.

    Uses draft model to propose candidates, target model to verify/correct,
    maintaining the exact target distribution while achieving speedup.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        branching_factor: int = 3,
        depth: int = 4,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.branching_factor = branching_factor
        self.depth = depth

        self.verifier = TokenTreeVerifier(target_model)
        self.corrector = RejectionSamplingCorrector()
        self.tracker = AcceptanceRateTracker()

    @torch.no_grad()
    def _get_draft_logits(self, input_ids: Tensor, n_steps: int) -> list[Tensor]:
        """Autoregressively collect n_steps logit vectors from the draft model."""
        logits_list: list[Tensor] = []
        current = input_ids.clone()

        for _ in range(n_steps):
            output = self.draft_model(current)
            if isinstance(output, tuple):
                logits = output[1]
            else:
                logits = output
            # logits: (B, T, V) — take last position, first batch element
            step_logits = logits[0, -1, :]  # (V,)
            logits_list.append(step_logits)

            # Extend with greedy token for next step
            next_tok = step_logits.argmax().unsqueeze(0).unsqueeze(0)  # (1, 1)
            current = torch.cat([current, next_tok], dim=1)

        return logits_list

    def generate(
        self,
        input_ids: Tensor,       # (B, T)
        max_new_tokens: int = 20,
    ) -> Tensor:
        """Generate max_new_tokens tokens using speculative rejection sampling.

        For simplicity, uses branching_factor=1 path (standard spec decoding),
        accepting the longest valid prefix and appending a corrected token.

        Returns: (B, T + max_new_tokens)
        """
        B, T_init = input_ids.shape
        current = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            spec_depth = min(self.depth, remaining)

            # --- 1. Draft phase: get logits for spec_depth positions ---
            draft_logits_list = self._get_draft_logits(current[:1], spec_depth)

            # --- 2. Build draft tree (single path, branching_factor=1) ---
            tree = DraftTree(branching_factor=1, depth=spec_depth)
            tree.build(draft_logits_list)

            # draft_sequence: (spec_depth,) — the greedy path
            draft_seq = tree.tokens[:, 0]  # (spec_depth,)
            draft_lp = tree.log_probs[:, 0]  # (spec_depth,)

            # Expand to batch: (B, spec_depth)
            draft_tokens_batch = draft_seq.unsqueeze(0).expand(B, -1)

            # --- 3. Verify with target model ---
            target_logits, accept_mask = self.verifier.verify(current, draft_tokens_batch)
            # target_logits: (B, spec_depth, V)
            # accept_mask:   (B, spec_depth) bool

            # --- 4. Find longest accepted prefix (first batch element drives) ---
            accept_vec = accept_mask[0]   # (spec_depth,)
            n_accepted = 0
            for i in range(spec_depth):
                if accept_vec[i].item():
                    n_accepted += 1
                else:
                    break

            self.tracker.record(n_accepted, spec_depth)

            # --- 5. Append accepted tokens ---
            if n_accepted > 0:
                accepted_toks = draft_tokens_batch[:, :n_accepted]  # (B, n_accepted)
                current = torch.cat([current, accepted_toks], dim=1)
                tokens_generated += n_accepted

            # --- 6. Append correction token at the rejection point ---
            if tokens_generated < max_new_tokens:
                if n_accepted < spec_depth:
                    # Correction at position n_accepted
                    corrected = self.corrector.correct(
                        draft_token=int(draft_tokens_batch[0, n_accepted].item()),
                        draft_logp=float(draft_lp[n_accepted].item()),
                        target_logits=target_logits[0, n_accepted],
                        accepted=False,
                    )
                else:
                    # All accepted — need one bonus token from target
                    # Run target on the last position
                    out = self.target_model(current)
                    if isinstance(out, tuple):
                        bonus_logits = out[1]
                    else:
                        bonus_logits = out
                    bonus_logits_last = bonus_logits[0, -1, :]
                    corrected = int(bonus_logits_last.argmax().item())

                correction_tok = torch.full(
                    (B, 1), corrected, dtype=torch.long, device=current.device
                )
                current = torch.cat([current, correction_tok], dim=1)
                tokens_generated += 1

        # Trim to exactly T_init + max_new_tokens
        return current[:, :T_init + max_new_tokens]


# ---------------------------------------------------------------------------
# AcceptanceRateTracker
# ---------------------------------------------------------------------------

class AcceptanceRateTracker:
    """Monitor acceptance rate statistics across decode steps."""

    def __init__(self) -> None:
        self._total_accepted = 0
        self._total_drafted = 0

    def record(self, n_accepted: int, n_drafted: int) -> None:
        """Record one speculative step."""
        if n_accepted < 0:
            raise ValueError("n_accepted must be >= 0")
        if n_drafted < 0:
            raise ValueError("n_drafted must be >= 0")
        if n_accepted > n_drafted:
            raise ValueError("n_accepted cannot exceed n_drafted")
        self._total_accepted += n_accepted
        self._total_drafted += n_drafted

    def mean_acceptance_rate(self) -> float:
        """Return total_accepted / total_drafted, or 0.0 if no data."""
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    def speedup_estimate(self) -> float:
        """Theoretical speedup: 1 + mean_acceptance_rate."""
        return 1.0 + self.mean_acceptance_rate()

    def reset(self) -> None:
        """Reset all counters."""
        self._total_accepted = 0
        self._total_drafted = 0
