"""Beam search decoding for AureliusTransformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class BeamSearchConfig:
    """Configuration for beam search decoding."""

    num_beams: int = 4
    max_new_tokens: int = 50
    length_penalty: float = 1.0     # score /= len^length_penalty (>1 favors longer)
    eos_token_id: int | None = None
    min_new_tokens: int = 1         # don't stop before this many new tokens


@dataclass
class BeamResult:
    """Result of beam search decoding."""

    sequences: list[list[int]]      # list of num_beams completed sequences (token IDs)
    scores: list[float]             # log-prob scores (higher = better), length-penalized
    # sequences[0] is the best (highest score)


def beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cfg: BeamSearchConfig,
) -> BeamResult:
    """Run beam search from a single prompt.

    Args:
        model: AureliusTransformer (eval mode expected)
        input_ids: (1, S) prompt token IDs
        cfg: BeamSearchConfig

    Returns:
        BeamResult with sequences sorted by score descending.
    """
    device = input_ids.device
    prompt_tokens = input_ids[0].tolist()
    prompt_len = len(prompt_tokens)

    # Initialize: single beam with score 0
    # Each beam is (cumulative_log_prob, token_list)
    beams: list[tuple[float, list[int]]] = [(0.0, list(prompt_tokens))]
    completed: list[tuple[float, list[int]]] = []

    with torch.no_grad():
        for step in range(cfg.max_new_tokens):
            # If all beams are completed, stop
            if not beams:
                break

            all_candidates: list[tuple[float, list[int]]] = []

            for score, tokens in beams:
                # Forward pass
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                output = model(ids)
                logits = output[1]  # (1, S, V)

                # Log-probs at the last position
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)

                # Top-k candidates
                topk_log_probs, topk_indices = torch.topk(log_probs, cfg.num_beams)

                for i in range(cfg.num_beams):
                    token_id = topk_indices[i].item()
                    candidate_score = score + topk_log_probs[i].item()
                    candidate_tokens = tokens + [token_id]
                    all_candidates.append((candidate_score, candidate_tokens))

            # Keep top num_beams candidates
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            all_candidates = all_candidates[: cfg.num_beams]

            # Separate completed and active beams
            beams = []
            new_tokens_generated = step + 1

            for score, tokens in all_candidates:
                if (
                    cfg.eos_token_id is not None
                    and tokens[-1] == cfg.eos_token_id
                    and new_tokens_generated >= cfg.min_new_tokens
                ):
                    completed.append((score, tokens))
                else:
                    beams.append((score, tokens))

    # Add any remaining active beams to completed
    completed.extend(beams)

    # Apply length penalty and sort
    def penalized_score(item: tuple[float, list[int]]) -> float:
        score, tokens = item
        length = len(tokens)
        return score / (length ** cfg.length_penalty)

    completed.sort(key=penalized_score, reverse=True)

    # Take top num_beams results
    completed = completed[: cfg.num_beams]

    sequences = [tokens for _, tokens in completed]
    scores = [penalized_score((s, t)) for s, t in completed]

    return BeamResult(sequences=sequences, scores=scores)
