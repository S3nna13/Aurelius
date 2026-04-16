"""Counterfactual explanation for token sequences.

Given a model and an input token sequence, find a minimally-edited version of
the sequence that flips the model's prediction at the last position toward a
desired target token.

Two search strategies are provided:
  - greedy_counterfactual: iteratively pick the single-token swap that most
    increases the target logit, until success or budget exhausted.
  - beam_search_counterfactual: maintain a beam of candidate sequences, expanding
    each by considering random candidate token swaps.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config / Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualConfig:
    max_substitutions: int = 3     # max tokens to change
    n_candidates: int = 8          # candidates per position
    beam_width: int = 4            # beam search width
    target_class: int = 0          # target output class/token to flip toward
    temperature: float = 1.0       # sampling temperature for candidates


@dataclass
class CounterfactualResult:
    original_ids: Tensor           # (T,) original token sequence
    counterfactual_ids: Tensor     # (T,) modified token sequence
    changed_positions: List[int]   # which positions were changed
    original_logits: Tensor        # (T, vocab) logits before change
    new_logits: Tensor             # (T, vocab) logits after change
    n_substitutions: int           # number of token substitutions made
    success: bool                  # did we flip the prediction at last position?


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def token_edit_distance(ids_a: Tensor, ids_b: Tensor) -> int:
    """Count number of positions where ids_a and ids_b differ.

    Args:
        ids_a: (T,) integer tensor.
        ids_b: (T,) integer tensor of the same length.

    Returns:
        Number of differing positions as a Python int.
    """
    return int((ids_a != ids_b).sum().item())


def _get_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    """Run model on (1, T) input and return (T, vocab) logits.

    The AureliusTransformer forward returns (loss, logits, kv) where
    logits has shape (1, T, vocab_size).
    """
    with torch.no_grad():
        output = model(input_ids.unsqueeze(0))  # batch of 1
    # output is (loss, logits, kv)
    logits = output[1]          # (1, T, vocab)
    return logits[0]            # (T, vocab)


def _score(logits_T_V: Tensor, target_token: int) -> float:
    """Return the logit for target_token at the last sequence position."""
    return logits_T_V[-1, target_token].item()


def _sample_candidates(vocab_size: int, n_candidates: int, exclude: int, seed_offset: int = 0) -> List[int]:
    """Return a list of n_candidates token ids sampled uniformly, excluding `exclude`."""
    candidates: List[int] = []
    seen = {exclude}
    # Deterministic iteration using a fixed permutation
    g = torch.Generator()
    g.manual_seed(seed_offset)
    perm = torch.randperm(vocab_size, generator=g).tolist()
    for tok in perm:
        if tok not in seen:
            candidates.append(tok)
            if len(candidates) >= n_candidates:
                break
    return candidates


# ---------------------------------------------------------------------------
# Greedy counterfactual
# ---------------------------------------------------------------------------

def greedy_counterfactual(
    model: nn.Module,
    input_ids: Tensor,          # (T,) token sequence
    target_token: int,          # target token id at last position
    config: CounterfactualConfig,
) -> CounterfactualResult:
    """Greedy search: iteratively replace the single token substitution
    that most increases logit of target_token at last position.
    Stop when target_token is top-1 or max_substitutions reached.

    Args:
        model: AureliusTransformer (or compatible) in eval mode.
        input_ids: (T,) 1-D integer tensor of token ids.
        target_token: desired token id to promote at the last position.
        config: CounterfactualConfig controlling search hyperparameters.

    Returns:
        CounterfactualResult with the best counterfactual found.
    """
    T = input_ids.shape[0]
    vocab_size = None  # discovered on first forward pass

    # Original logits
    original_logits = _get_logits(model, input_ids)   # (T, vocab)
    vocab_size = original_logits.shape[-1]

    current_ids = input_ids.clone()
    changed_positions: List[int] = []

    for step in range(config.max_substitutions):
        # Check success: is target_token already top-1 at last pos?
        cur_logits = _get_logits(model, current_ids)
        if int(cur_logits[-1].argmax().item()) == target_token:
            return CounterfactualResult(
                original_ids=input_ids,
                counterfactual_ids=current_ids,
                changed_positions=changed_positions,
                original_logits=original_logits,
                new_logits=cur_logits,
                n_substitutions=len(changed_positions),
                success=True,
            )

        # Find the best single-token swap across all positions
        best_score = _score(cur_logits, target_token)
        best_pos = -1
        best_tok = -1

        already_changed = set(changed_positions)

        for pos in range(T):
            if pos in already_changed:
                continue  # don't re-substitute same position
            candidates = _sample_candidates(
                vocab_size,
                config.n_candidates,
                exclude=int(current_ids[pos].item()),
                seed_offset=step * T * 1000 + pos,
            )
            for cand_tok in candidates:
                trial_ids = current_ids.clone()
                trial_ids[pos] = cand_tok
                trial_logits = _get_logits(model, trial_ids)
                score = _score(trial_logits, target_token)
                if score > best_score:
                    best_score = score
                    best_pos = pos
                    best_tok = cand_tok

        if best_pos == -1:
            # No improvement found; stop early
            break

        current_ids[best_pos] = best_tok
        changed_positions.append(best_pos)

    final_logits = _get_logits(model, current_ids)
    success = int(final_logits[-1].argmax().item()) == target_token

    return CounterfactualResult(
        original_ids=input_ids,
        counterfactual_ids=current_ids,
        changed_positions=changed_positions,
        original_logits=original_logits,
        new_logits=final_logits,
        n_substitutions=len(changed_positions),
        success=success,
    )


# ---------------------------------------------------------------------------
# Beam search counterfactual
# ---------------------------------------------------------------------------

@dataclass
class _BeamState:
    ids: Tensor
    changed: List[int]
    score: float


def beam_search_counterfactual(
    model: nn.Module,
    input_ids: Tensor,          # (T,) token sequence
    target_token: int,          # target token id at last position
    config: CounterfactualConfig,
) -> CounterfactualResult:
    """Beam search over token substitutions.
    Each beam state is a modified sequence; score = logit of target at last pos.
    Returns the beam state with highest target logit score.

    Args:
        model: AureliusTransformer (or compatible) in eval mode.
        input_ids: (T,) 1-D integer tensor of token ids.
        target_token: desired token id to promote at the last position.
        config: CounterfactualConfig controlling search hyperparameters.

    Returns:
        CounterfactualResult for the highest-scoring beam state found.
    """
    T = input_ids.shape[0]

    # Original logits
    original_logits = _get_logits(model, input_ids)   # (T, vocab)
    vocab_size = original_logits.shape[-1]
    init_score = _score(original_logits, target_token)

    # Initialize beam with unmodified sequence
    beam: List[_BeamState] = [
        _BeamState(ids=input_ids.clone(), changed=[], score=init_score)
    ]

    for step in range(config.max_substitutions):
        # Check if any beam state already achieves success
        for state in beam:
            cur_logits = _get_logits(model, state.ids)
            if int(cur_logits[-1].argmax().item()) == target_token:
                return CounterfactualResult(
                    original_ids=input_ids,
                    counterfactual_ids=state.ids,
                    changed_positions=state.changed,
                    original_logits=original_logits,
                    new_logits=cur_logits,
                    n_substitutions=len(state.changed),
                    success=True,
                )

        # Expand each beam state
        candidates: List[_BeamState] = []
        for beam_idx, state in enumerate(beam):
            already_changed = set(state.changed)
            for pos in range(T):
                if pos in already_changed:
                    continue
                cand_tokens = _sample_candidates(
                    vocab_size,
                    config.n_candidates,
                    exclude=int(state.ids[pos].item()),
                    seed_offset=step * len(beam) * T * 1000 + beam_idx * T * 100 + pos,
                )
                for cand_tok in cand_tokens:
                    trial_ids = state.ids.clone()
                    trial_ids[pos] = cand_tok
                    trial_logits = _get_logits(model, trial_ids)
                    score = _score(trial_logits, target_token)
                    new_changed = state.changed + [pos]
                    candidates.append(_BeamState(ids=trial_ids, changed=new_changed, score=score))

        if not candidates:
            break

        # Keep top beam_width states by score
        candidates.sort(key=lambda s: s.score, reverse=True)
        beam = candidates[: config.beam_width]

    # Return highest-scoring state across the final beam
    best_state = max(beam, key=lambda s: s.score)
    final_logits = _get_logits(model, best_state.ids)
    success = int(final_logits[-1].argmax().item()) == target_token

    return CounterfactualResult(
        original_ids=input_ids,
        counterfactual_ids=best_state.ids,
        changed_positions=best_state.changed,
        original_logits=original_logits,
        new_logits=final_logits,
        n_substitutions=len(best_state.changed),
        success=success,
    )


# ---------------------------------------------------------------------------
# Feature importance from counterfactual
# ---------------------------------------------------------------------------

def counterfactual_feature_importance(
    original_ids: Tensor,        # (T,)
    counterfactual_ids: Tensor,  # (T,)
) -> Tensor:
    """Binary mask: 1 at positions that were changed, 0 elsewhere.

    Args:
        original_ids: (T,) original token id sequence.
        counterfactual_ids: (T,) counterfactual token id sequence.

    Returns:
        (T,) float tensor with 1.0 at changed positions and 0.0 elsewhere.
    """
    return (original_ids != counterfactual_ids).float()
