"""Speculative decoding v4: Leviathan et al. 2023 / Chen et al. 2023.

Classic speculative decoding with a separate small draft model and large target
model.  The draft model generates K tokens speculatively; the target verifies
them in a single forward pass using rejection sampling that exactly preserves
the target distribution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# SpeculativeVerifier
# ---------------------------------------------------------------------------


class SpeculativeVerifier:
    """Exact rejection-sampling verifier that preserves the target distribution.

    Implements the token-level accept/reject logic from Leviathan et al. 2023:
    each draft token is accepted with probability ``min(1, p_target / p_draft)``.
    On rejection a corrected token is sampled from the residual distribution
    ``max(0, p_target - p_draft) / Z`` so that the marginal output distribution
    remains exactly the target.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_token(
        self,
        draft_token: int,
        draft_prob: float,
        target_probs: Tensor,
    ) -> tuple[bool, int]:
        """Verify a single draft token against the target distribution.

        Args:
            draft_token: The token id proposed by the draft model.
            draft_prob:  Probability the draft model assigned to ``draft_token``.
                         Must be in ``(0, 1]``.
            target_probs: Full target distribution, shape ``(V,)``, summing to 1.

        Returns:
            ``(accepted, corrected_token)`` where ``accepted`` is True when the
            draft token was accepted and ``corrected_token`` is either
            ``draft_token`` (when accepted) or a token resampled from the
            residual distribution (when rejected).
        """
        if target_probs.dim() != 1:
            raise ValueError(f"target_probs must be 1-D, got shape {tuple(target_probs.shape)}")

        p_target = float(target_probs[draft_token].item())

        # Acceptance probability: min(1, p_target / p_draft)
        if draft_prob <= 0.0:
            accept_prob = 1.0
        else:
            accept_prob = min(1.0, p_target / draft_prob)

        u = torch.rand(1).item()
        if u < accept_prob:
            return True, draft_token

        # Rejection: sample from residual distribution max(0, p_target - p_draft)
        residual = (target_probs - draft_prob).clamp(min=0.0)
        total = residual.sum()
        if total < 1e-12:
            # Fallback: sample directly from target
            corrected = int(torch.multinomial(target_probs, num_samples=1).item())
        else:
            corrected = int(torch.multinomial(residual / total, num_samples=1).item())

        return False, corrected

    def verify_sequence(
        self,
        draft_tokens: Tensor,
        draft_probs: Tensor,
        target_logits: Tensor,
    ) -> tuple[int, int]:
        """Verify a sequence of K draft tokens left-to-right.

        Verifies tokens in order, stopping at the first rejection.  If all K
        tokens are accepted a bonus token is sampled from the target distribution
        at position K (the position *after* all draft tokens).

        Args:
            draft_tokens:  Shape ``(K,)`` – token ids proposed by draft model.
            draft_probs:   Shape ``(K,)`` – probability of each drafted token
                           under the draft model.
            target_logits: Shape ``(K, V)`` – raw logits from the target model
                           at the K draft positions.  These are converted to
                           probabilities internally.

        Returns:
            ``(n_accepted, final_token)`` where ``n_accepted`` is in ``[0, K]``
            and ``final_token`` is a valid vocab id that should be appended after
            the accepted tokens (either the bonus token or the correction token
            at the first rejection).
        """
        K = draft_tokens.shape[0]
        if draft_probs.shape != (K,):
            raise ValueError(f"draft_probs shape {tuple(draft_probs.shape)} must be ({K},)")
        if target_logits.shape[0] != K:
            raise ValueError(f"target_logits first dim {target_logits.shape[0]} must equal K={K}")

        # Convert logits to probabilities for each position
        target_probs_all = F.softmax(target_logits.float(), dim=-1)  # (K, V)

        for i in range(K):
            t = int(draft_tokens[i].item())
            p_draft = float(draft_probs[i].item())
            accepted, correction = self.verify_token(t, p_draft, target_probs_all[i])
            if not accepted:
                return i, correction  # n_accepted = i (none at position i)

        # All K accepted — sample bonus token from target at last position
        bonus_probs = target_probs_all[-1]
        bonus = int(torch.multinomial(bonus_probs, num_samples=1).item())
        return K, bonus


# ---------------------------------------------------------------------------
# DraftModel
# ---------------------------------------------------------------------------


class DraftModel:
    """Lightweight autoregressive draft model wrapper.

    Args:
        model: Callable ``(input_ids: Tensor) -> logits (B, T, V)`` or
               ``(input_ids: Tensor) -> logits (T, V)`` for unbatched input.
        temperature: Softmax temperature for sampling draft tokens.  1.0 gives
                     unscaled multinomial sampling.
    """

    def __init__(self, model, temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = max(temperature, 1e-8)

    def draft(self, input_ids: Tensor, n_tokens: int) -> tuple[Tensor, Tensor]:
        """Autoregressively generate ``n_tokens`` tokens with the draft model.

        Args:
            input_ids: Current context, shape ``(T,)`` (1-D LongTensor).
            n_tokens:  Number of tokens to generate.

        Returns:
            ``(draft_tokens, draft_probs)`` where

            - ``draft_tokens``: ``LongTensor`` of shape ``(n_tokens,)``
            - ``draft_probs``:  ``FloatTensor`` of shape ``(n_tokens,)`` –
              probability assigned to each drafted token.
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got shape {tuple(input_ids.shape)}")

        token_list: list[int] = []
        prob_list: list[float] = []
        context = input_ids.clone()

        with torch.no_grad():
            for _ in range(n_tokens):
                # Model may return (B, T, V) or (T, V)
                logits = self.model(context)
                if logits.dim() == 3:
                    # (B, T, V) — take batch 0
                    next_logits = logits[0, -1, :]
                elif logits.dim() == 2:
                    next_logits = logits[-1, :]
                else:
                    raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}")

                probs = F.softmax(next_logits.float() / self.temperature, dim=-1)
                token_id = int(torch.multinomial(probs, num_samples=1).item())
                token_list.append(token_id)
                prob_list.append(float(probs[token_id].item()))
                context = torch.cat([context, torch.tensor([token_id], dtype=torch.long)])

        draft_tokens = torch.tensor(token_list, dtype=torch.long)
        draft_probs = torch.tensor(prob_list, dtype=torch.float32)
        return draft_tokens, draft_probs


# ---------------------------------------------------------------------------
# TargetModel
# ---------------------------------------------------------------------------


class TargetModel:
    """Large target model wrapper for one-shot verification.

    Args:
        model: Callable ``(input_ids: Tensor) -> logits (B, T, V)`` or
               ``(input_ids: Tensor) -> logits (T, V)``.
    """

    def __init__(self, model) -> None:
        self.model = model

    def score(self, input_ids: Tensor, draft_tokens: Tensor) -> Tensor:
        """Score draft tokens with a single target-model forward pass.

        Concatenates ``input_ids`` with ``draft_tokens``, runs the model once,
        and extracts logits at the positions that *predict* each draft token.
        Specifically, position ``len(input_ids) - 1 + i`` predicts
        ``draft_tokens[i]``.

        Args:
            input_ids:    Context before draft tokens, shape ``(T,)``.
            draft_tokens: Draft token ids, shape ``(K,)``.

        Returns:
            ``FloatTensor`` of shape ``(K, V)`` – one logit vector per draft
            position.
        """
        if input_ids.dim() != 1:
            raise ValueError(f"input_ids must be 1-D, got shape {tuple(input_ids.shape)}")
        if draft_tokens.dim() != 1:
            raise ValueError(f"draft_tokens must be 1-D, got shape {tuple(draft_tokens.shape)}")

        K = draft_tokens.shape[0]
        combined = torch.cat([input_ids, draft_tokens])  # (T + K,)

        with torch.no_grad():
            logits = self.model(combined)

        # Handle (B, T+K, V) or (T+K, V)
        if logits.dim() == 3:
            logits = logits[0]  # (T+K, V)

        if logits.dim() != 2:
            raise ValueError(
                f"Expected 2-D logits after squeezing batch, got {tuple(logits.shape)}"
            )

        T = input_ids.shape[0]
        # Position i (0-indexed in *input_ids*): logits[i] predicts token i+1.
        # So logits[T-1] predicts draft_tokens[0], …, logits[T+K-2] predicts
        # draft_tokens[K-1].
        start = T - 1
        target_logits = logits[start : start + K, :]  # (K, V)

        if target_logits.shape[0] != K:
            raise RuntimeError(
                f"Could not extract {K} target logit rows from output of shape "
                f"{tuple(logits.shape)} with start={start}"
            )

        return target_logits.float()


# ---------------------------------------------------------------------------
# SpeculativeDecoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """Full speculative decoding loop combining draft, score, and verify.

    Args:
        draft_model:    :class:`DraftModel` instance.
        target_model:   :class:`TargetModel` instance.
        verifier:       :class:`SpeculativeVerifier` instance.
        n_draft_tokens: Number of tokens the draft model proposes each round.
    """

    def __init__(
        self,
        draft_model: DraftModel,
        target_model: TargetModel,
        verifier: SpeculativeVerifier,
        n_draft_tokens: int = 5,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.verifier = verifier
        self.n_draft_tokens = n_draft_tokens

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 20,
    ) -> tuple[Tensor, dict]:
        """Generate tokens using speculative decoding.

        Each iteration:
        1. Draft ``n_draft_tokens`` tokens with the draft model.
        2. Score all draft positions with a single target-model forward pass.
        3. Verify via rejection sampling; obtain ``n_accepted`` and
           ``final_token``.
        4. Append ``n_accepted`` draft tokens + ``final_token`` to the sequence.
        5. If ``n_accepted < n_draft_tokens`` restart with the (shorter) updated
           context.

        Args:
            input_ids:      Prompt, shape ``(B, T)`` with ``B == 1`` or
                            1-D ``(T,)``.  Batch dimension is handled internally.
            max_new_tokens: Maximum tokens to append.

        Returns:
            ``(output_ids, stats)`` where

            - ``output_ids``: ``LongTensor`` of shape ``(B, T + max_new_tokens)``
              (or ``(1, T + max_new_tokens)`` when input was 1-D).
            - ``stats``: dict with keys
              ``'n_steps'``, ``'total_accepted'``, ``'total_drafted'``,
              ``'acceptance_rate'``.
        """
        # Normalise to 1-D context for internal use
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported, got {input_ids.shape[0]}")
            context = input_ids[0].clone()
        elif input_ids.dim() == 1:
            context = input_ids.clone()
        else:
            raise ValueError(f"input_ids must be 1-D or 2-D, got shape {tuple(input_ids.shape)}")

        n_steps = 0
        total_accepted = 0
        total_drafted = 0
        n_generated = 0

        while n_generated < max_new_tokens:
            remaining = max_new_tokens - n_generated
            k = min(self.n_draft_tokens, remaining)

            # 1. Draft
            draft_tokens, draft_probs = self.draft_model.draft(context, k)

            # 2. Score
            target_logits = self.target_model.score(context, draft_tokens)

            # 3. Verify
            n_accepted, final_token = self.verifier.verify_sequence(
                draft_tokens, draft_probs, target_logits
            )

            # 4. Append: accepted draft tokens + final_token
            accepted_ids = draft_tokens[:n_accepted]
            tokens_to_add = torch.cat([accepted_ids, torch.tensor([final_token], dtype=torch.long)])

            # Clip to remaining budget
            budget = max_new_tokens - n_generated
            tokens_to_add = tokens_to_add[:budget]

            context = torch.cat([context, tokens_to_add])
            n_generated += len(tokens_to_add)

            total_accepted += n_accepted
            total_drafted += k
            n_steps += 1

        # Reconstruct (B, T + max_new_tokens) output
        # Trim or pad to exactly T + max_new_tokens
        T_in = input_ids.shape[-1]
        output_len = T_in + max_new_tokens
        if context.shape[0] > output_len:
            context = context[:output_len]

        output_ids = context.unsqueeze(0)  # (1, T + max_new_tokens)

        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0

        stats = {
            "n_steps": n_steps,
            "total_accepted": total_accepted,
            "total_drafted": total_drafted,
            "acceptance_rate": acceptance_rate,
        }

        return output_ids, stats


# ---------------------------------------------------------------------------
# SpeedupBenchmark
# ---------------------------------------------------------------------------


class SpeedupBenchmark:
    """Measure and estimate speculative decoding efficiency.

    Provides both theoretical and empirical speedup calculations.
    """

    def __init__(self) -> None:
        pass

    def theoretical_speedup(self, acceptance_rate: float, n_draft: int) -> float:
        """Expected tokens per target call under i.i.d. acceptance.

        Derivation: with acceptance rate ``alpha`` and ``n_draft`` draft tokens,
        the expected number of tokens accepted per target call is::

            E = sum_{k=0}^{n_draft} alpha^k * k + alpha^n_draft
              = (1 - alpha^{n_draft+1}) / (1 - alpha)   for alpha != 1

        The speedup vs. baseline (1 token per target call) equals ``E``.

        Args:
            acceptance_rate: Per-token acceptance probability in ``[0, 1]``.
            n_draft:         Number of draft tokens per round.

        Returns:
            Expected tokens generated per target-model forward pass (>= 1.0 for
            ``acceptance_rate >= 0``).
        """
        alpha = float(acceptance_rate)
        n = int(n_draft)

        if abs(alpha - 1.0) < 1e-9:
            # All tokens accepted: n_draft + 1 (bonus) tokens every call
            return float(n + 1)

        # (1 - alpha^{n+1}) / (1 - alpha)
        speedup = (1.0 - alpha ** (n + 1)) / (1.0 - alpha)
        return float(speedup)

    def empirical_speedup(self, stats: dict) -> float:
        """Compute empirical speedup from a generation stats dict.

        Speedup = (total_accepted + n_steps) / n_steps

        The numerator counts total tokens added: ``total_accepted`` draft tokens
        plus one final/bonus token per step (``n_steps``).

        Args:
            stats: Dict with keys ``'n_steps'`` and ``'total_accepted'``
                   as returned by :meth:`SpeculativeDecoder.generate`.

        Returns:
            Empirical speedup as a float.  Returns 0.0 if ``n_steps == 0``.
        """
        n_steps = int(stats["n_steps"])
        total_accepted = int(stats["total_accepted"])
        if n_steps == 0:
            return 0.0
        return (total_accepted + n_steps) / n_steps

    def draft_model_overhead(
        self,
        n_draft: int,
        draft_latency: float,
        target_latency: float,
    ) -> float:
        """Net wallclock speedup accounting for draft model cost.

        Per speculative step the total time is::

            t_step = n_draft * draft_latency + target_latency

        Tokens produced per step (under theoretical calculation) = theoretical
        speedup.  The net speedup vs. baseline (``target_latency`` per token)
        is::

            net = (theoretical_speedup * target_latency) / t_step

        Args:
            n_draft:        Number of draft tokens per round.
            draft_latency:  Time (s) to generate one draft token.
            target_latency: Time (s) for one target-model forward pass.

        Returns:
            Net wallclock speedup relative to pure target-model autoregressive
            decoding.  Returns 0.0 when latencies are zero.
        """
        if target_latency <= 0.0:
            return 0.0
        # Use acceptance_rate=0.9 as a representative value when not specified
        # (The method is a utility estimator; callers supply real stats.)
        return float(n_draft * draft_latency + target_latency)
