"""Token healing: fix tokenization boundary artifacts by resampling the last token constrained to a prefix."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TokenHealingConfig:
    top_p: float = 0.9
    temperature: float = 1.0
    max_heal_tokens: int = 1  # how many trailing tokens to back up and reheal


def get_valid_token_ids(
    partial_str: str,
    vocab: dict[int, str],
) -> list[int]:
    """Return all token IDs whose string representation starts with partial_str.

    Args:
        partial_str: The string the next token must start with (e.g. "htt")
        vocab: dict mapping token_id -> token_string

    Returns:
        List of token IDs that start with partial_str. Empty list if none.
    """
    return [tid for tid, tok_str in vocab.items() if tok_str.startswith(partial_str)]


def build_prefix_constrained_logits(
    logits: torch.Tensor,
    valid_ids: list[int],
) -> torch.Tensor:
    """Mask logits so only valid_ids have non-inf values.

    Args:
        logits: (V,) logit tensor
        valid_ids: list of allowed token IDs

    Returns:
        (V,) masked logits tensor.
    """
    masked = torch.full_like(logits, float("-inf"))
    if valid_ids:
        valid_tensor = torch.tensor(valid_ids, dtype=torch.long, device=logits.device)
        masked[valid_tensor] = logits[valid_tensor]
    return masked


def heal_tokens(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    partial_str: str,
    vocab: dict[int, str],
    cfg: TokenHealingConfig | None = None,
) -> torch.Tensor:
    """Perform token healing: back up by len(partial_str's token) and resample.

    Algorithm:
    1. Remove the last token from input_ids
    2. Run a forward pass to get logits for the last remaining position
    3. Get valid token IDs (tokens starting with partial_str)
    4. Mask out all invalid tokens (set logits to -inf)
    5. Sample from the constrained distribution using temperature + top-p
    6. Return input_ids with the last token replaced by the healed token

    Args:
        model: AureliusTransformer
        input_ids: (1, S) tensor -- last token should be the partial/boundary token
        partial_str: String that the replacement token must start with
        vocab: dict[int, str] mapping token_id to string
        cfg: TokenHealingConfig

    Returns:
        (1, S) tensor with healed last token. If no valid tokens found, returns input_ids unchanged.
    """
    cfg = cfg or TokenHealingConfig()
    B, S = input_ids.shape
    if S < 2:
        return input_ids  # nothing to back up

    # Back up: remove last token
    prefix_ids = input_ids[:, :-1]  # (1, S-1)

    # Forward pass on prefix
    with torch.no_grad():
        _, logits, _ = model(prefix_ids)  # logits: (1, S-1, V)
    last_logits = logits[0, -1, :]  # (V,)

    # Find valid tokens
    valid_ids = get_valid_token_ids(partial_str, vocab)
    if not valid_ids:
        return input_ids  # no healing possible

    # Apply constraint + temperature
    constrained = build_prefix_constrained_logits(last_logits, valid_ids)
    constrained = constrained / max(cfg.temperature, 1e-8)
    probs = F.softmax(constrained, dim=-1)

    # Top-p sampling over constrained distribution
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = sorted_probs.cumsum(0)
    cutoff = (cumsum - sorted_probs) > cfg.top_p
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum() + 1e-12
    healed_idx_in_sorted = torch.multinomial(sorted_probs, 1)
    healed_token = sorted_idx[healed_idx_in_sorted]

    # Replace last token
    result = input_ids.clone()
    result[0, -1] = healed_token
    return result
