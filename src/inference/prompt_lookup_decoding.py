"""Prompt Lookup Decoding: draft tokens found in the prompt, not a draft model.

Saxena 2023 — speculative decoding where candidate tokens are n-gram matches
from the existing context/prompt. Zero extra model cost for drafting: ideal for
tasks where outputs copy spans from the input (document QA, code editing,
summarization).

Public API
----------
PromptLookupConfig      — configuration dataclass
find_ngram_matches      — find all positions in context where a query ngram matches
PromptLookupDecoding    — full decode loop with prompt-lookup drafting + statistics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PromptLookupConfig:
    """Configuration for Prompt Lookup Decoding."""

    max_matching_ngram_size: int = 3
    num_speculative_tokens: int = 10
    min_ngram_size: int = 1


# ---------------------------------------------------------------------------
# N-gram matching helper
# ---------------------------------------------------------------------------

def find_ngram_matches(
    context: Tensor,
    query: Tensor,
    max_candidates: int = 10,
) -> List[int]:
    """Find all positions in *context* where *query* ngram matches.

    Searches for every occurrence of the query ngram in the context tensor and
    returns the start indices of the tokens that *follow* each match (i.e.,
    where the candidate continuation begins).

    Parameters
    ----------
    context:
        1-D int64 tensor of context token ids to search within.
    query:
        1-D int64 tensor of the query ngram to look for.
    max_candidates:
        Maximum number of match positions to return.

    Returns
    -------
    List of integer positions — each is the index of the first token *after*
    a match in *context*. Empty list if no matches found.
    """
    if context.ndim != 1 or query.ndim != 1:
        raise ValueError("context and query must be 1-D tensors")

    ngram_size = query.shape[0]
    context_len = context.shape[0]

    if ngram_size == 0 or context_len < ngram_size:
        return []

    positions: List[int] = []
    # Slide over context looking for full ngram match
    for i in range(context_len - ngram_size + 1):
        if torch.equal(context[i : i + ngram_size], query):
            follow_pos = i + ngram_size
            # Only useful if there is at least one token after the match
            if follow_pos < context_len:
                positions.append(follow_pos)
                if len(positions) >= max_candidates:
                    break

    return positions


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PromptLookupDecoding:
    """Speculative decoding using prompt n-gram lookup for free draft tokens.

    Instead of a separate draft model, candidate continuations are found by
    matching the last *ngram_size* tokens of the current sequence against the
    prompt/context.  The main model verifies all candidates in a single forward
    pass.

    Parameters
    ----------
    model:
        A callable (or nn.Module) that accepts ``(1, T)`` int64 tensor and
        returns a tuple where index 1 is logits of shape ``(1, T, vocab_size)``
        or ``(1, vocab_size)`` (the latter for models that collapse the time
        dimension for us).
    max_matching_ngram_size:
        Largest ngram size to try when searching for a match.
    num_speculative_tokens:
        Maximum number of draft tokens to propose per step.
    max_new_tokens:
        Default budget for :meth:`generate`.
    """

    def __init__(
        self,
        model,
        max_matching_ngram_size: int = 3,
        num_speculative_tokens: int = 10,
        max_new_tokens: int = 512,
    ) -> None:
        self.model = model
        self.max_matching_ngram_size = max_matching_ngram_size
        self.num_speculative_tokens = num_speculative_tokens
        self.max_new_tokens = max_new_tokens

        # Statistics
        self._total_tokens: int = 0
        self._accepted_tokens: int = 0
        self._draft_steps: int = 0
        self._fallback_steps: int = 0

    # ------------------------------------------------------------------
    # Candidate finding
    # ------------------------------------------------------------------

    def find_candidate_tokens(
        self,
        input_ids: Tensor,
        ngram_size: int,
    ) -> Optional[Tensor]:
        """Search the context for an ngram match and return candidate tokens.

        Takes the last *ngram_size* tokens of *input_ids* as the query, then
        searches all prior tokens for a matching run.  Returns the tokens that
        follow the first/best match (up to ``num_speculative_tokens`` tokens).

        Parameters
        ----------
        input_ids:
            Shape ``(seq_len,)`` — single sequence of token ids (no batch dim).
        ngram_size:
            Size of the ngram to match.

        Returns
        -------
        Tensor of shape ``(k,)`` with 1 ≤ k ≤ ``num_speculative_tokens``, or
        ``None`` if no matching ngram was found in the context.
        """
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be 1-D (no batch dimension)")

        seq_len = input_ids.shape[0]
        if seq_len < ngram_size + 1:
            # Not enough tokens to form a query *and* have prior context
            return None

        # Query: last ngram_size tokens
        query = input_ids[-ngram_size:]
        # Context to search: all tokens before the query
        context = input_ids[: seq_len - ngram_size]

        positions = find_ngram_matches(context, query, max_candidates=1)

        if not positions:
            return None

        # Candidate tokens start at positions[0] in the *original* input_ids,
        # but we searched in `context` (which is input_ids[:-ngram_size]).
        # The positions returned are indices into `context`, so we map back:
        follow_pos = positions[0]  # index in context == index in input_ids

        # We need candidates from the full input_ids starting at follow_pos
        # (which is within input_ids[:-ngram_size], so it's valid).
        candidates_start = follow_pos
        candidates_end = min(
            candidates_start + self.num_speculative_tokens,
            seq_len - ngram_size,   # don't include query tokens themselves
        )

        if candidates_start >= candidates_end:
            return None

        return input_ids[candidates_start:candidates_end]

    # ------------------------------------------------------------------
    # Candidate verification
    # ------------------------------------------------------------------

    def verify_candidates(
        self,
        model_logits: Tensor,
        candidate_ids: Tensor,
    ) -> Tuple[Tensor, int]:
        """Greedy-verify candidate tokens against model logits.

        For each position *i* we check whether
        ``argmax(model_logits[i]) == candidate_ids[i]``.  We stop at the first
        mismatch and always append the model's own correction token at that
        position so the sequence is never left incomplete.

        Parameters
        ----------
        model_logits:
            Shape ``(n_candidates + 1, vocab_size)`` — the model was run on
            ``[context | candidates]`` and we take the shifted logit slice.
        candidate_ids:
            Shape ``(n_candidates,)`` int64.

        Returns
        -------
        accepted_tokens : Tensor of shape ``(k,)`` with accepted token ids
            (including the correction token at the end).
        n_accepted : int — number of candidate tokens that matched exactly
            (does NOT count the correction/bonus token).
        """
        if model_logits.ndim != 2:
            raise ValueError("model_logits must be 2-D: (n_candidates+1, vocab_size)")
        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids must be 1-D")

        n_candidates = candidate_ids.shape[0]
        accepted: List[Tensor] = []
        n_accepted = 0

        for i in range(n_candidates):
            greedy_tok = model_logits[i].argmax(dim=-1)  # scalar tensor
            if greedy_tok.item() == candidate_ids[i].item():
                accepted.append(greedy_tok)
                n_accepted += 1
            else:
                # Mismatch: append model's correction and stop
                accepted.append(greedy_tok)
                break
        else:
            # All candidates accepted — add the bonus token from the last logit
            bonus_tok = model_logits[n_candidates].argmax(dim=-1)
            accepted.append(bonus_tok)

        accepted_tokens = torch.stack(accepted)  # (k,)
        return accepted_tokens, n_accepted

    # ------------------------------------------------------------------
    # Model call helper
    # ------------------------------------------------------------------

    def _model_logits_for(self, input_ids: Tensor) -> Tensor:
        """Run the model on ``(1, T)`` input and return ``(T, vocab_size)`` logits."""
        with torch.no_grad():
            out = self.model(input_ids)

        # Support several common return conventions:
        # 1. (loss, logits, ...) tuple where logits is (1, T, V)
        # 2. (loss, logits, ...) tuple where logits is (1, V)
        # 3. Raw logits tensor
        if isinstance(out, (tuple, list)):
            logits = out[1]
        else:
            logits = out

        # Remove batch dimension
        if logits.ndim == 3:
            logits = logits[0]   # (T, V)
        elif logits.ndim == 2:
            # Could be (1, V) from models that only return last position
            if logits.shape[0] == 1:
                logits = logits  # keep as (1, V) — caller handles
            # else already (T, V)

        return logits  # (T, V) or (1, V)

    # ------------------------------------------------------------------
    # Full generation loop
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = None,
        temperature: float = 1.0,
    ) -> Tensor:
        """Generate tokens using prompt lookup drafting + main-model verification.

        Falls back to standard greedy decoding when no candidate ngram is found.

        Parameters
        ----------
        input_ids:
            Shape ``(1, seq_len)`` int64 — batched prompt (batch size must be 1).
        max_new_tokens:
            Override for ``self.max_new_tokens``.
        temperature:
            Sampling temperature (currently only greedy / argmax is used during
            verification; temperature affects the fallback path).

        Returns
        -------
        Tensor of shape ``(1, seq_len + generated)`` int64.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids.clone()
        tokens_generated = 0
        temp = max(float(temperature), 1e-8)

        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            seq_1d = generated[0]  # (T,) — unbatched view for ngram search

            # --- Try to find draft candidates from the prompt ---
            candidates: Optional[Tensor] = None
            for ngram_size in range(self.max_matching_ngram_size, 0, -1):
                cands = self.find_candidate_tokens(seq_1d, ngram_size)
                if cands is not None and cands.shape[0] > 0:
                    candidates = cands
                    break

            if candidates is not None:
                # Cap to remaining budget
                candidates = candidates[: min(candidates.shape[0], remaining)]
                n_cands = candidates.shape[0]

                # Build full input: context + candidates
                full_input = torch.cat(
                    [generated, candidates.unsqueeze(0)], dim=1
                )  # (1, T + n_cands)

                # Single forward pass
                all_logits = self._model_logits_for(full_input)  # (T+n_cands, V) or (1, V)

                if all_logits.shape[0] == 1:
                    # Model only returned last position — can't verify speculative tokens
                    # Fall through to greedy fallback
                    candidates = None
                else:
                    # Extract the n_cands + 1 logit positions we need.
                    # Position prompt_len-1+i predicts token at prompt_len+i.
                    prompt_len = generated.shape[1]
                    verify_start = prompt_len - 1
                    verify_end = verify_start + n_cands + 1
                    verify_logits = all_logits[verify_start:verify_end]  # (n_cands+1, V)

                    if verify_logits.shape[0] < n_cands + 1:
                        candidates = None  # insufficient logits, fall back
                    else:
                        accepted_toks, n_accepted = self.verify_candidates(
                            verify_logits, candidates
                        )

                        self._draft_steps += 1
                        self._accepted_tokens += n_accepted
                        self._total_tokens += accepted_toks.shape[0]

                        n_to_add = min(accepted_toks.shape[0], remaining)
                        generated = torch.cat(
                            [generated, accepted_toks[:n_to_add].unsqueeze(0)], dim=1
                        )
                        tokens_generated += n_to_add
                        continue

            # --- Fallback: standard greedy single-token step ---
            logits = self._model_logits_for(generated)  # (T, V) or (1, V)
            last_logits = logits[-1]  # (V,) — last position

            if temp < 1e-7:
                next_tok = last_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(last_logits / temp, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=1)
            self._total_tokens += 1
            self._fallback_steps += 1
            tokens_generated += 1

        return generated

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """Return generation statistics.

        Returns
        -------
        dict with keys:
            ``total_tokens``       — total new tokens generated.
            ``accepted_tokens``    — tokens accepted from draft candidates.
            ``acceptance_rate``    — accepted / total in [0, 1].
            ``speedup_estimate``   — naive estimate: tokens / model_calls.
        """
        total = float(self._total_tokens)
        accepted = float(self._accepted_tokens)
        rate = accepted / total if total > 0 else 0.0

        # Each draft step is one model call (regardless of how many tokens accepted);
        # each fallback step is also one model call.
        model_calls = float(self._draft_steps + self._fallback_steps)
        speedup = total / model_calls if model_calls > 0 else 1.0

        return {
            "total_tokens": total,
            "accepted_tokens": accepted,
            "acceptance_rate": rate,
            "speedup_estimate": speedup,
        }
