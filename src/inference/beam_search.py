"""Beam search decoding for AureliusTransformer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class BeamSearchConfig:
    """Configuration for beam search decoding."""

    beam_width: int = 4
    max_new_tokens: int = 50
    length_penalty: float = 1.0  # score = log_prob / len^length_penalty
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True


@dataclass
class Beam:
    """A single beam hypothesis during search."""

    token_ids: list[int] = field(default_factory=list)
    log_prob: float = 0.0
    is_done: bool = False

    # length_penalty is stored externally in config; score property
    # needs it, so we accept it as an optional parameter with a default
    # that mirrors config default of 1.0.
    _length_penalty: float = field(default=1.0, repr=False)

    @property
    def score(self) -> float:
        """Length-normalised log-probability (higher is better)."""
        length = len(self.token_ids)
        if length == 0:
            return self.log_prob
        return self.log_prob / (length**self._length_penalty)


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Divide logits for tokens already present in token_ids by *penalty*.

    Tokens with positive logits are divided; tokens with negative logits are
    multiplied (making them more negative), consistent with the HuggingFace
    convention.

    Args:
        logits: 1-D tensor of shape (vocab_size,).
        token_ids: Token ids already generated (including prompt).
        penalty: Repetition penalty > 1 discourages repetition.

    Returns:
        Modified logits tensor (same shape, in-place clone).
    """
    if penalty == 1.0 or not token_ids:
        return logits

    logits = logits.clone()
    unique_ids = list(set(token_ids))
    score = logits[unique_ids]
    # If score is positive, divide; if negative, multiply.
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[unique_ids] = score
    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    token_ids: list[int],
    n: int,
) -> torch.Tensor:
    """Block any token that would complete an n-gram already seen.

    Args:
        logits: 1-D tensor of shape (vocab_size,).
        token_ids: Token ids already generated (including prompt).
        n: n-gram size (e.g. 3 blocks trigrams).  If 0, no-op.

    Returns:
        Modified logits tensor (banned tokens set to -inf).
    """
    if n == 0 or len(token_ids) < n - 1:
        return logits

    logits = logits.clone()

    # Build set of all (n-1)-grams we have seen, mapping to follow-up tokens.
    ngram_follows: dict[tuple[int, ...], list[int]] = {}
    for i in range(len(token_ids) - (n - 1)):
        prefix = tuple(token_ids[i : i + n - 1])
        next_tok = token_ids[i + n - 1]
        ngram_follows.setdefault(prefix, []).append(next_tok)

    # The last (n-1) tokens form the current prefix.
    current_prefix = tuple(token_ids[-(n - 1) :])
    banned = ngram_follows.get(current_prefix, [])
    if banned:
        logits[banned] = float("-inf")
    return logits


def beam_search_step(
    model: torch.nn.Module,
    beams: list[Beam],
    vocab_size: int,
    eos_token_id: int | None,
    config: BeamSearchConfig,
) -> list[Beam]:
    """Expand each active beam with the top-beam_width continuations.

    Args:
        model: AureliusTransformer — called as model(input_ids) -> (loss, logits, pkv).
        beams: Current list of Beam objects (only active beams are expanded).
        vocab_size: Size of the token vocabulary.
        eos_token_id: Token id that signals end-of-sequence.
        config: BeamSearchConfig.

    Returns:
        New list of up to beam_width Beam objects sorted by score descending.
    """
    candidates: list[Beam] = []
    device = torch.device("cpu")

    for beam in beams:
        if beam.is_done:
            # Carry done beams forward unchanged.
            candidates.append(beam)
            continue

        ids = torch.tensor([beam.token_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            _loss, logits, _pkv = model(ids)

        # (vocab_size,) — last token position
        next_logits = logits[0, -1, :].float()

        # Apply decoding constraints.
        if config.repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits, beam.token_ids, config.repetition_penalty
            )
        if config.no_repeat_ngram_size > 0:
            next_logits = apply_no_repeat_ngram(
                next_logits, beam.token_ids, config.no_repeat_ngram_size
            )

        log_probs = F.log_softmax(next_logits, dim=-1)

        # Take top beam_width tokens.
        k = min(config.beam_width, vocab_size)
        topk_log_probs, topk_indices = torch.topk(log_probs, k)

        for i in range(k):
            token_id = int(topk_indices[i].item())
            new_log_prob = beam.log_prob + float(topk_log_probs[i].item())
            new_token_ids = beam.token_ids + [token_id]
            is_done = eos_token_id is not None and token_id == eos_token_id
            candidates.append(
                Beam(
                    token_ids=new_token_ids,
                    log_prob=new_log_prob,
                    is_done=is_done,
                    _length_penalty=config.length_penalty,
                )
            )

    # Sort all candidates by score and keep best beam_width.
    candidates.sort(key=lambda b: b.score, reverse=True)
    return candidates[: config.beam_width]


class BeamSearchDecoder:
    """Full beam search decoder."""

    def __init__(self, config: BeamSearchConfig | None = None) -> None:
        self.config = config or BeamSearchConfig()

    def generate(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Run beam search and return the best token sequence.

        Args:
            model: AureliusTransformer — called as model(input_ids) -> (loss, logits, pkv).
            input_ids: (1, prompt_len) prompt token ids.
            eos_token_id: Token id that signals end-of-sequence.

        Returns:
            1-D LongTensor containing the best sequence (prompt + generated tokens).
        """
        cfg = self.config
        prompt_tokens = input_ids[0].tolist()

        # Initialise beam_width beams from the prompt.
        beams: list[Beam] = [
            Beam(
                token_ids=list(prompt_tokens),
                log_prob=0.0,
                is_done=False,
                _length_penalty=cfg.length_penalty,
            )
        ]

        vocab_size = None  # will be inferred on first step

        for _step in range(cfg.max_new_tokens):
            # Infer vocab size from model on first step if needed.
            if vocab_size is None:
                with torch.no_grad():
                    ids = torch.tensor([beams[0].token_ids], dtype=torch.long)
                    _loss, logits, _pkv = model(ids)
                    vocab_size = logits.shape[-1]

            beams = beam_search_step(model, beams, vocab_size, eos_token_id, cfg)

            # Early stopping: all beams are done.
            if cfg.early_stopping and all(b.is_done for b in beams):
                break

        # Return the best beam's token ids as a tensor.
        best = max(beams, key=lambda b: b.score)
        return torch.tensor(best.token_ids, dtype=torch.long)
