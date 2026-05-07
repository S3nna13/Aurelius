"""Unit tests for :mod:`src.safety.safety_token_regularization`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.safety.safety_token_regularization import (
    DEFAULT_SAFETY_TEMPLATES,
    SafetyTokenRegularizer,
)


class _MockTokenizer:
    """Whitespace tokenizer for testing."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.vocab_size = max(vocab.values()) + 1

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = text.lower().split()
        return [self.vocab.get(t, 0) for t in tokens]


class _MockLM(nn.Module):
    """Tiny LM that returns deterministic logits."""

    def __init__(self, vocab_size: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        with torch.no_grad():
            nn.init.normal_(self.embed.weight, 0.0, 0.1)
            nn.init.normal_(self.lm_head.weight, 0.0, 0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list]:
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )
        return loss, logits, []


def _build_fixture() -> tuple[SafetyTokenRegularizer, _MockLM, _MockTokenizer]:
    vocab = {
        "i": 1,
        "cannot": 2,
        "i'm": 3,
        "sorry": 4,
        "apologize": 5,
        "that": 6,
        "request": 7,
        "is": 8,
        "harmful": 9,
        "can't": 10,
        "assist": 11,
        "am": 12,
        "not": 13,
        "able": 14,
        "to": 15,
        "comply": 16,
        "unable": 17,
        "must": 18,
        "refuse": 19,
        "this": 20,
        "inappropriate": 21,
    }
    tokenizer = _MockTokenizer(vocab)
    model = _MockLM(tokenizer.vocab_size)
    regularizer = SafetyTokenRegularizer(
        model=model,
        tokenizer=tokenizer,
        lambda_str=0.01,
        top_k_per_template=3,
    )
    return regularizer, model, tokenizer


def test_init_extracts_safety_tokens() -> None:
    regularizer, _, _ = _build_fixture()
    assert regularizer.safety_token_ids.numel() > 0
    assert regularizer.ref_logits.numel() == regularizer.safety_token_ids.numel()


def test_compute_str_loss_returns_scalar() -> None:
    regularizer, model, tokenizer = _build_fixture()
    ids = tokenizer.encode("i cannot assist", add_bos=False, add_eos=False)
    input_ids = torch.tensor([ids])
    _, logits, _ = model(input_ids)
    loss = regularizer.compute_str_loss(logits)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_zero_loss_when_logits_match_reference() -> None:
    """If safety-token logits equal the cached reference exactly, loss is zero."""
    regularizer, _, _ = _build_fixture()
    B, T, V = 2, 5, 32
    logits = torch.randn(B, T, V)
    for idx, token_id in enumerate(regularizer.safety_token_ids.tolist()):
        logits[:, :, token_id] = regularizer.ref_logits[idx]
    loss = regularizer.compute_str_loss(logits)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_lambda_str_scaling() -> None:
    regularizer, model, tokenizer = _build_fixture()
    ids = tokenizer.encode("i cannot", add_bos=False, add_eos=False)
    input_ids = torch.tensor([ids])
    _, logits, _ = model(input_ids)

    regularizer.lambda_str = 0.01
    loss1 = regularizer.compute_str_loss(logits)

    regularizer.lambda_str = 0.02
    loss2 = regularizer.compute_str_loss(logits)

    assert pytest.approx(loss2.item(), rel=1e-5) == 2.0 * loss1.item()


def test_invalid_logits_dim_raises() -> None:
    regularizer, _, _ = _build_fixture()
    with pytest.raises(ValueError):
        regularizer.compute_str_loss(torch.randn(5))


def test_templates_default_not_empty() -> None:
    assert len(DEFAULT_SAFETY_TEMPLATES) > 0


def test_model_training_mode_restored() -> None:
    """After initialization the model's original train/eval state is restored."""
    vocab = {"i": 1, "cannot": 2}
    tokenizer = _MockTokenizer(vocab)
    model = _MockLM(tokenizer.vocab_size)
    model.train()
    assert model.training is True

    _ = SafetyTokenRegularizer(
        model=model,
        tokenizer=tokenizer,
        templates=["i cannot"],
        lambda_str=0.01,
    )

    assert model.training is True


def test_empty_templates_fallback() -> None:
    """With empty templates the regularizer falls back to a no-op token set."""
    vocab = {"i": 1, "cannot": 2}
    tokenizer = _MockTokenizer(vocab)
    model = _MockLM(tokenizer.vocab_size)
    regularizer = SafetyTokenRegularizer(
        model=model,
        tokenizer=tokenizer,
        templates=[],
        lambda_str=0.01,
    )
    assert regularizer.safety_token_ids.numel() == 1
    logits = torch.randn(1, 3, tokenizer.vocab_size)
    loss = regularizer.compute_str_loss(logits)
    assert loss.dim() == 0
