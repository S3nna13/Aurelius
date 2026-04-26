"""Diverse decoding: diverse beam search, length-penalized beam, and stochastic beam search."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DiverseBeamConfig:
    """Configuration for diverse beam search decoding."""

    num_beams: int = 4
    num_beam_groups: int = 2  # number of diverse groups
    diversity_penalty: float = 0.5  # penalize tokens chosen by previous groups
    length_penalty: float = 1.0  # alpha in score / (length^alpha)
    min_length: int = 0  # minimum generation length
    max_new_tokens: int = 20
    temperature: float = 1.0


@dataclass
class BeamHypothesis:
    """A single beam hypothesis."""

    token_ids: list[int]
    score: float  # cumulative log prob
    length: int
    group_id: int = 0

    def normalized_score(self, length_penalty: float) -> float:
        """Return length-penalized score: score / (length^length_penalty)."""
        if self.length == 0:
            return self.score
        return self.score / (self.length**length_penalty)


class BeamSearchDecoder:
    """Standard beam search decoder (no diverse groups)."""

    def __init__(self, model: torch.nn.Module, config: DiverseBeamConfig) -> None:
        self.model = model
        self.config = config

    def generate(self, input_ids: torch.Tensor) -> list[BeamHypothesis]:
        """Run standard beam search from a single prompt.

        Args:
            input_ids: (1, S) prompt token IDs.

        Returns:
            List of BeamHypothesis sorted by cumulative score descending.
        """
        cfg = self.config
        device = input_ids.device
        prompt_tokens = input_ids[0].tolist()

        # Each beam: (cumulative_log_prob, token_list)
        beams: list[tuple[float, list[int]]] = [(0.0, list(prompt_tokens))]

        with torch.no_grad():
            for _step in range(cfg.max_new_tokens):
                if not beams:
                    break

                all_candidates: list[tuple[float, list[int]]] = []

                for score, tokens in beams:
                    ids = torch.tensor([tokens], dtype=torch.long, device=device)
                    output = self.model(ids)
                    logits = output[1]  # (1, S, V)

                    # Apply temperature
                    last_logits = logits[0, -1, :] / cfg.temperature
                    log_probs = F.log_softmax(last_logits, dim=-1)

                    topk_log_probs, topk_indices = torch.topk(log_probs, cfg.num_beams)

                    for i in range(cfg.num_beams):
                        token_id = topk_indices[i].item()
                        candidate_score = score + topk_log_probs[i].item()
                        candidate_tokens = tokens + [token_id]
                        all_candidates.append((candidate_score, candidate_tokens))

                # Keep top num_beams
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[: cfg.num_beams]

        # Convert to BeamHypothesis
        prompt_len = len(prompt_tokens)
        hypotheses = []
        for score, tokens in beams:
            gen_len = len(tokens) - prompt_len
            hyp = BeamHypothesis(
                token_ids=tokens,
                score=score,
                length=max(gen_len, 1),
                group_id=0,
            )
            hypotheses.append(hyp)

        hypotheses.sort(key=lambda h: h.score, reverse=True)
        return hypotheses


class DiverseBeamSearchDecoder:
    """Diverse beam search decoder: penalizes tokens chosen by earlier groups."""

    def __init__(self, model: torch.nn.Module, config: DiverseBeamConfig) -> None:
        self.model = model
        self.config = config
        self.beams_per_group = config.num_beams // config.num_beam_groups

    def _apply_diversity_penalty(
        self, logits: torch.Tensor, previous_tokens: set[int]
    ) -> torch.Tensor:
        """Subtract diversity_penalty from logits for tokens in previous_tokens.

        Args:
            logits: (V,) raw logits tensor.
            previous_tokens: set of token ids chosen by earlier groups.

        Returns:
            Modified logits with diversity penalty applied.
        """
        logits = logits.clone()
        for token_id in previous_tokens:
            if 0 <= token_id < logits.shape[-1]:
                logits[token_id] -= self.config.diversity_penalty
        return logits

    def generate(self, input_ids: torch.Tensor) -> list[BeamHypothesis]:
        """Run diverse beam search from a single prompt.

        Args:
            input_ids: (1, S) prompt token IDs.

        Returns:
            List of all BeamHypothesis across all groups sorted by normalized_score.
        """
        cfg = self.config
        device = input_ids.device
        prompt_tokens = input_ids[0].tolist()
        n_groups = cfg.num_beam_groups
        bpg = self.beams_per_group

        # Initialize: each group starts with one beam at score 0
        # group_beams[g] = list of (score, token_list)
        group_beams: list[list[tuple[float, list[int]]]] = [
            [(0.0, list(prompt_tokens))] for _ in range(n_groups)
        ]

        with torch.no_grad():
            for _step in range(cfg.max_new_tokens):
                # Tokens chosen this step by each group (for diversity penalty)
                step_chosen_tokens: list[set[int]] = []

                new_group_beams: list[list[tuple[float, list[int]]]] = []

                for g in range(n_groups):
                    beams = group_beams[g]

                    # Collect tokens chosen by earlier groups this step
                    previous_tokens: set[int] = set()
                    for prev_g in range(g):
                        previous_tokens |= step_chosen_tokens[prev_g]

                    all_candidates: list[tuple[float, list[int]]] = []
                    group_token_picks: set[int] = set()

                    for score, tokens in beams:
                        ids = torch.tensor([tokens], dtype=torch.long, device=device)
                        output = self.model(ids)
                        logits = output[1][0, -1, :]  # (V,)

                        # Apply temperature
                        logits = logits / cfg.temperature

                        # Apply diversity penalty for group g > 0
                        if g > 0:
                            logits = self._apply_diversity_penalty(logits, previous_tokens)

                        log_probs = F.log_softmax(logits, dim=-1)

                        topk_log_probs, topk_indices = torch.topk(log_probs, bpg)

                        for i in range(bpg):
                            token_id = topk_indices[i].item()
                            candidate_score = score + topk_log_probs[i].item()
                            candidate_tokens = tokens + [token_id]
                            all_candidates.append((candidate_score, candidate_tokens))
                            group_token_picks.add(token_id)

                    # Keep top bpg for this group
                    all_candidates.sort(key=lambda x: x[0], reverse=True)
                    new_group_beams.append(all_candidates[:bpg])
                    step_chosen_tokens.append(group_token_picks)

                group_beams = new_group_beams

        # Flatten and convert to BeamHypothesis
        prompt_len = len(prompt_tokens)
        hypotheses = []
        for g, beams in enumerate(group_beams):
            for score, tokens in beams:
                gen_len = len(tokens) - prompt_len
                hyp = BeamHypothesis(
                    token_ids=tokens,
                    score=score,
                    length=max(gen_len, 1),
                    group_id=g,
                )
                hypotheses.append(hyp)

        lp = cfg.length_penalty
        hypotheses.sort(key=lambda h: h.normalized_score(lp), reverse=True)
        return hypotheses


def stochastic_beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_beams: int,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> list[BeamHypothesis]:
    """Stochastic beam search: add Gumbel noise for exploration.

    Args:
        model: AureliusTransformer.
        input_ids: (1, S) prompt token IDs.
        num_beams: number of beams to maintain.
        max_new_tokens: maximum tokens to generate.
        temperature: temperature scaling for logits.

    Returns:
        List of BeamHypothesis sorted by score descending.
    """
    device = input_ids.device
    prompt_tokens = input_ids[0].tolist()
    eps = 1e-10

    beams: list[tuple[float, list[int]]] = [(0.0, list(prompt_tokens))]

    with torch.no_grad():
        for _step in range(max_new_tokens):
            if not beams:
                break

            all_candidates: list[tuple[float, list[int]]] = []

            for score, tokens in beams:
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                output = model(ids)
                logits = output[1][0, -1, :]  # (V,)

                # Apply temperature
                logits = logits / temperature

                # Add Gumbel noise: -log(-log(U + eps)) where U ~ Uniform(0, 1)
                uniform = torch.rand_like(logits)
                gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)
                logits = logits + gumbel_noise

                log_probs = F.log_softmax(logits, dim=-1)

                topk_log_probs, topk_indices = torch.topk(log_probs, num_beams)

                for i in range(num_beams):
                    token_id = topk_indices[i].item()
                    candidate_score = score + topk_log_probs[i].item()
                    candidate_tokens = tokens + [token_id]
                    all_candidates.append((candidate_score, candidate_tokens))

            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:num_beams]

    prompt_len = len(prompt_tokens)
    hypotheses = []
    for score, tokens in beams:
        gen_len = len(tokens) - prompt_len
        hyp = BeamHypothesis(
            token_ids=tokens,
            score=score,
            length=max(gen_len, 1),
            group_id=0,
        )
        hypotheses.append(hyp)

    hypotheses.sort(key=lambda h: h.score, reverse=True)
    return hypotheses


def length_penalty_rerank(
    hypotheses: list[BeamHypothesis], length_penalty: float
) -> list[BeamHypothesis]:
    """Re-rank hypotheses by normalized_score(length_penalty) descending.

    Args:
        hypotheses: list of BeamHypothesis to re-rank.
        length_penalty: alpha exponent for length normalization.

    Returns:
        Sorted copy of hypotheses (descending normalized_score).
    """
    return sorted(hypotheses, key=lambda h: h.normalized_score(length_penalty), reverse=True)
