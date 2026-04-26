"""Tests for EAGLE speculative decoding (eagle_speculative.py).

Tiny config: d_model=64, vocab_size=256, n_draft_steps=3,
             draft_hidden_dim=32, batch=1, seq_len=8
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.eagle_speculative import (
    EAGLEConfig,
    EAGLEDecoder,
    EAGLEDraftHead,
)

# ---------------------------------------------------------------------------
# Mock target model (no imports from src/model/transformer.py)
# ---------------------------------------------------------------------------


class MockTargetModel(nn.Module):
    """Minimal mock that returns (logits, hidden_state) from forward().

    Owns a real nn.Embedding so EAGLEDecoder can retrieve token embeddings.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape
        logits = torch.randn(B, T, self.vocab_size, device=input_ids.device)
        hidden = torch.randn(B, T, self.d_model, device=input_ids.device)
        return logits, hidden


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 64
VOCAB_SIZE = 256
N_DRAFT_STEPS = 3
DRAFT_HIDDEN_DIM = 32
BATCH = 1
SEQ_LEN = 8


@pytest.fixture
def cfg() -> EAGLEConfig:
    return EAGLEConfig(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_draft_steps=N_DRAFT_STEPS,
        draft_hidden_dim=DRAFT_HIDDEN_DIM,
        temperature=1.0,
    )


@pytest.fixture
def draft_head(cfg: EAGLEConfig) -> EAGLEDraftHead:
    torch.manual_seed(0)
    return EAGLEDraftHead(cfg)


@pytest.fixture
def mock_target(cfg: EAGLEConfig) -> MockTargetModel:
    torch.manual_seed(1)
    return MockTargetModel(cfg.vocab_size, cfg.d_model)


@pytest.fixture
def decoder(
    mock_target: MockTargetModel, draft_head: EAGLEDraftHead, cfg: EAGLEConfig
) -> EAGLEDecoder:
    return EAGLEDecoder(mock_target, draft_head, cfg)


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_draft_head_output_shape(draft_head: EAGLEDraftHead, cfg: EAGLEConfig) -> None:
    """EAGLEDraftHead must output (B, T, vocab_size)."""
    B, T = BATCH, SEQ_LEN
    hidden = torch.randn(B, T, cfg.d_model)
    token_embed = torch.randn(B, T, cfg.d_model)
    logits = draft_head(hidden, token_embed)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    )


def test_draft_returns_correct_shape(
    decoder: EAGLEDecoder, input_ids: torch.Tensor, cfg: EAGLEConfig
) -> None:
    """draft() must return (B, n_draft_steps) integer tensor."""
    features = torch.randn(BATCH, SEQ_LEN, cfg.d_model)
    draft_ids = decoder.draft(input_ids, features)
    assert draft_ids.shape == (BATCH, cfg.n_draft_steps), (
        f"Expected ({BATCH}, {cfg.n_draft_steps}), got {draft_ids.shape}"
    )
    assert draft_ids.dtype in (torch.int64, torch.long)


def test_verify_and_accept_returns_ids_and_int(
    decoder: EAGLEDecoder, input_ids: torch.Tensor, cfg: EAGLEConfig
) -> None:
    """verify_and_accept must return (Tensor, int)."""
    n_draft = cfg.n_draft_steps
    draft_ids = torch.randint(0, cfg.vocab_size, (BATCH, n_draft))
    target_logits = torch.randn(BATCH, SEQ_LEN + n_draft, cfg.vocab_size)
    draft_logits = torch.randn(BATCH, n_draft, cfg.vocab_size)

    result = decoder.verify_and_accept(input_ids, draft_ids, target_logits, draft_logits)
    assert isinstance(result, tuple) and len(result) == 2

    accepted_ids, n_accepted = result
    assert isinstance(accepted_ids, torch.Tensor)
    assert isinstance(n_accepted, int)


def test_n_accepted_in_valid_range(
    decoder: EAGLEDecoder, input_ids: torch.Tensor, cfg: EAGLEConfig
) -> None:
    """n_accepted must be between 0 and n_draft_steps (inclusive)."""
    n_draft = cfg.n_draft_steps
    draft_ids = torch.randint(0, cfg.vocab_size, (BATCH, n_draft))
    target_logits = torch.randn(BATCH, SEQ_LEN + n_draft, cfg.vocab_size)
    draft_logits = torch.randn(BATCH, n_draft, cfg.vocab_size)

    _, n_accepted = decoder.verify_and_accept(input_ids, draft_ids, target_logits, draft_logits)
    assert 0 <= n_accepted <= n_draft, f"n_accepted={n_accepted} out of range [0, {n_draft}]"


def test_generate_step_returns_new_tokens(decoder: EAGLEDecoder, input_ids: torch.Tensor) -> None:
    """generate_step must return a non-empty tensor of new token ids."""
    new_ids, stats = decoder.generate_step(input_ids)
    assert isinstance(new_ids, torch.Tensor)
    assert new_ids.ndim == 2
    assert new_ids.shape[0] == BATCH
    assert new_ids.shape[1] >= 1, "generate_step must produce at least one token"


def test_generate_step_stats_contains_n_accepted(
    decoder: EAGLEDecoder, input_ids: torch.Tensor
) -> None:
    """The stats dict returned by generate_step must contain 'n_accepted'."""
    _, stats = decoder.generate_step(input_ids)
    assert "n_accepted" in stats, f"'n_accepted' key missing from stats: {stats}"


def test_draft_head_has_trainable_params(draft_head: EAGLEDraftHead) -> None:
    """EAGLEDraftHead must have at least one trainable (requires_grad) parameter."""
    trainable = [p for p in draft_head.parameters() if p.requires_grad]
    assert len(trainable) > 0, "EAGLEDraftHead has no trainable parameters"
    total = sum(p.numel() for p in trainable)
    assert total > 0, "Total trainable parameter count is 0"


def test_acceptance_rate_one_when_draft_equals_target(cfg: EAGLEConfig) -> None:
    """When draft and target distributions are identical (temperature=0 / greedy),
    all draft tokens should be accepted."""
    # Use temperature=0 equivalent: make draft token always the argmax of both dists.
    n_draft = cfg.n_draft_steps
    vocab = cfg.vocab_size

    # Craft logits where each position has a clear argmax at a fixed token index
    fixed_tokens = torch.arange(n_draft) % vocab  # (n_draft,)

    target_logits = torch.full((BATCH, SEQ_LEN + n_draft, vocab), -1e9)
    draft_logits = torch.full((BATCH, n_draft, vocab), -1e9)

    for i in range(n_draft):
        tok = fixed_tokens[i].item()
        target_logits[0, SEQ_LEN - 1 + i, tok] = 1e9  # position predicts draft tok i
        draft_logits[0, i, tok] = 1e9

    draft_ids = fixed_tokens.unsqueeze(0)  # (1, n_draft)
    input_ids = torch.randint(0, vocab, (BATCH, SEQ_LEN))

    zero_temp_cfg = EAGLEConfig(
        d_model=cfg.d_model,
        vocab_size=vocab,
        n_draft_steps=n_draft,
        draft_hidden_dim=cfg.draft_hidden_dim,
        temperature=1e-9,  # near-zero → near-deterministic
    )
    mock = MockTargetModel(vocab, cfg.d_model)
    head = EAGLEDraftHead(zero_temp_cfg)
    dec = EAGLEDecoder(mock, head, zero_temp_cfg)

    _, n_accepted = dec.verify_and_accept(input_ids, draft_ids, target_logits, draft_logits)
    assert n_accepted == n_draft, f"Expected all {n_draft} tokens accepted, got {n_accepted}"


def test_backward_through_draft_head(draft_head: EAGLEDraftHead, cfg: EAGLEConfig) -> None:
    """Backpropagation through EAGLEDraftHead must update its weights."""
    import torch.optim as optim

    optimizer = optim.SGD(draft_head.parameters(), lr=0.1)
    before = [p.clone().detach() for p in draft_head.parameters()]

    B, T = BATCH, SEQ_LEN
    hidden = torch.randn(B, T, cfg.d_model, requires_grad=False)
    token_embed = torch.randn(B, T, cfg.d_model, requires_grad=False)

    logits = draft_head(hidden, token_embed)  # (B, T, vocab_size)
    targets = torch.randint(0, cfg.vocab_size, (B * T,))
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), targets)
    loss.backward()
    optimizer.step()

    after = list(draft_head.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "Draft head weights did not change after backward pass"


def test_different_seeds_give_different_drafts(decoder: EAGLEDecoder, cfg: EAGLEConfig) -> None:
    """Different random seeds must produce different draft token sequences."""
    results = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ_LEN))
        features = torch.randn(BATCH, SEQ_LEN, cfg.d_model)
        draft = decoder.draft(ids, features)
        results.append(draft)

    # At least two of the three must differ
    all_same = all(torch.equal(results[0], r) for r in results[1:])
    assert not all_same, "All drafts are identical across different seeds"


def test_draft_head_output_all_finite(draft_head: EAGLEDraftHead, cfg: EAGLEConfig) -> None:
    """EAGLEDraftHead logits must be finite (no NaN / Inf)."""
    B, T = BATCH, SEQ_LEN
    hidden = torch.randn(B, T, cfg.d_model)
    token_embed = torch.randn(B, T, cfg.d_model)
    logits = draft_head(hidden, token_embed)
    assert torch.isfinite(logits).all(), "Draft head produced non-finite logits"


def test_generate_step_stats_keys(decoder: EAGLEDecoder, input_ids: torch.Tensor) -> None:
    """Stats dict must also contain 'n_draft' and 'acceptance_rate'."""
    _, stats = decoder.generate_step(input_ids)
    for key in ("n_accepted", "n_draft", "acceptance_rate"):
        assert key in stats, f"Missing key '{key}' in stats dict: {stats}"
