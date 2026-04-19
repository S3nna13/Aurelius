"""Integration test for JSON-mode constrained decoder."""

from __future__ import annotations

import torch

import src.inference as inference
from src.inference import (
    JSONDecoderState,
    JSONMaskBuilder,
    is_valid_json_prefix,
)


def test_exports_exposed_on_package() -> None:
    assert hasattr(inference, "JSONMaskBuilder")
    assert hasattr(inference, "JSONDecoderState")
    assert hasattr(inference, "is_valid_json_prefix")
    # Pre-existing entries remain intact.
    assert hasattr(inference, "ContinuousBatchingScheduler")
    assert "continuous_batching" in inference.SCHEDULER_REGISTRY


def test_end_to_end_forward_then_mask() -> None:
    torch.manual_seed(0)
    vocab = ["{", "}", "[", "]", ",", ":", '"', "a", "b", "1", "2", " "]
    V = len(vocab)

    # A tiny "forward pass": a random linear over a fake hidden.
    hidden = torch.randn(1, 8)
    head = torch.nn.Linear(8, V, bias=False)
    with torch.no_grad():
        logits = head(hidden).squeeze(0)  # [V]

    builder = JSONMaskBuilder(vocab)
    state: JSONDecoderState = builder.reset()

    masked = builder.mask_logits(logits, state)
    # At least one admissible token exists at start.
    finite = torch.isfinite(masked)
    assert finite.any().item() is True
    assert finite.sum().item() >= 1

    # Greedy-sample an admissible token and advance state.
    tok_id = int(torch.argmax(masked).item())
    tok_str = vocab[tok_id]
    # The chosen token must be admissible (finite logit).
    assert torch.isfinite(masked[tok_id]).item() is True
    state = builder.update(state, tok_str)

    # Do one more step and verify mask is still non-empty for typical
    # openings of a JSON document.
    masked2 = builder.mask_logits(logits, state)
    assert torch.isfinite(masked2).sum().item() >= 1


def test_prefix_validity_helper() -> None:
    assert is_valid_json_prefix('{"k":1') is True
    assert is_valid_json_prefix("[1, 2, 3]") is True
    assert is_valid_json_prefix("][") is False
