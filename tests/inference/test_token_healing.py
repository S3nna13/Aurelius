"""Tests for src/inference/token_healing.py.

Uses a tiny mock model (random logits) with VOCAB_SIZE=32.
All tests use pure PyTorch — no HuggingFace, scipy, or sklearn.
"""

from __future__ import annotations

import torch

from src.inference.token_healing import (
    TokenHealer,
    TokenHealingConfig,
    apply_logit_bias,
    get_token_prefix_logit_bias,
    greedy_extend,
)

# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 32
BATCH = 1
SEQ_LEN = 8


def make_mock_model(seed: int | None = None) -> callable:
    """Return a deterministic mock model_fn: (B, T) -> (B, T, VOCAB_SIZE)."""
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    def model_fn(ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        # Deterministic given the seed so top-k is stable across calls
        torch.manual_seed(seed if seed is not None else 0)
        return torch.randn(B, T, VOCAB_SIZE)

    return model_fn


def make_ids(B: int = BATCH, T: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, T), dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. TokenHealingConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = TokenHealingConfig()
    assert cfg.n_rollback_tokens == 1
    assert cfg.top_k_candidates == 10
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 50


def test_config_custom():
    cfg = TokenHealingConfig(
        n_rollback_tokens=2, top_k_candidates=5, temperature=0.7, max_new_tokens=20
    )
    assert cfg.n_rollback_tokens == 2
    assert cfg.top_k_candidates == 5
    assert cfg.temperature == 0.7
    assert cfg.max_new_tokens == 20


# ---------------------------------------------------------------------------
# 2. get_token_prefix_logit_bias
# ---------------------------------------------------------------------------


def test_logit_bias_shape():
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, [0, 5, 10])
    assert bias.shape == (VOCAB_SIZE,)


def test_logit_bias_allowed_are_zero():
    allowed = [1, 7, 15]
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, allowed)
    for idx in allowed:
        assert bias[idx].item() == 0.0, f"Allowed token {idx} should have bias 0.0"


def test_logit_bias_blocked_are_large_negative():
    allowed = [3, 9]
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, allowed)
    blocked = [i for i in range(VOCAB_SIZE) if i not in allowed]
    for idx in blocked:
        assert bias[idx].item() < -1e8, f"Blocked token {idx} should have large negative bias"


def test_logit_bias_empty_allowed_all_blocked():
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, [])
    assert (bias < -1e8).all(), "Empty allowed list should block every token"


def test_logit_bias_all_allowed_all_zero():
    allowed = list(range(VOCAB_SIZE))
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, allowed)
    assert (bias == 0.0).all()


# ---------------------------------------------------------------------------
# 3. apply_logit_bias
# ---------------------------------------------------------------------------


def test_apply_logit_bias_output_shape():
    logits = torch.zeros(BATCH, SEQ_LEN, VOCAB_SIZE)
    bias = torch.zeros(VOCAB_SIZE)
    out = apply_logit_bias(logits, bias)
    assert out.shape == logits.shape


def test_apply_logit_bias_adds_correctly():
    logits = torch.ones(BATCH, 1, VOCAB_SIZE)
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, [0])
    out = apply_logit_bias(logits, bias)
    # Token 0: 1.0 + 0.0 = 1.0
    assert abs(out[0, 0, 0].item() - 1.0) < 1e-6
    # Token 1: 1.0 + (-1e9) ≈ -1e9
    assert out[0, 0, 1].item() < -1e8


def test_apply_logit_bias_blocks_forbidden_tokens():
    """argmax of biased logits should never fall outside the allowed set."""
    torch.manual_seed(42)
    allowed = [2, 17, 25]
    logits = torch.randn(1, 1, VOCAB_SIZE)
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, allowed)
    biased = apply_logit_bias(logits, bias)
    winner = biased[0, 0, :].argmax().item()
    assert winner in allowed, f"argmax {winner} is not in allowed set {allowed}"


# ---------------------------------------------------------------------------
# 4. TokenHealer.rollback_tokens
# ---------------------------------------------------------------------------


def test_rollback_tokens_shapes():
    model_fn = make_mock_model(seed=0)
    healer = TokenHealer(model_fn)
    ids = make_ids(B=1, T=8)
    prefix, removed = healer.rollback_tokens(ids, n=1)
    assert prefix.shape == (1, 7)
    assert removed.shape == (1, 1)


def test_rollback_tokens_n2():
    model_fn = make_mock_model(seed=0)
    healer = TokenHealer(model_fn)
    ids = make_ids(B=1, T=10)
    prefix, removed = healer.rollback_tokens(ids, n=2)
    assert prefix.shape == (1, 8)
    assert removed.shape == (1, 2)


def test_rollback_tokens_content():
    model_fn = make_mock_model(seed=0)
    healer = TokenHealer(model_fn)
    ids = torch.arange(10, dtype=torch.long).unsqueeze(0)  # [[0,1,...,9]]
    prefix, removed = healer.rollback_tokens(ids, n=3)
    assert torch.equal(prefix[0], torch.arange(7, dtype=torch.long))
    assert torch.equal(removed[0], torch.tensor([7, 8, 9], dtype=torch.long))


# ---------------------------------------------------------------------------
# 5. TokenHealer.get_continuation_candidates
# ---------------------------------------------------------------------------


def test_candidates_shape():
    model_fn = make_mock_model(seed=1)
    healer = TokenHealer(model_fn)
    prefix = make_ids(B=1, T=6)
    removed = make_ids(B=1, T=1)
    candidates = healer.get_continuation_candidates(prefix, removed, top_k=10)
    assert candidates.shape == (1, 10)


def test_candidates_in_valid_vocab_range():
    model_fn = make_mock_model(seed=2)
    healer = TokenHealer(model_fn)
    prefix = make_ids(B=1, T=5)
    removed = make_ids(B=1, T=1)
    candidates = healer.get_continuation_candidates(prefix, removed, top_k=10)
    assert (candidates >= 0).all()
    assert (candidates < VOCAB_SIZE).all()


def test_candidates_top_k_clamped_to_vocab():
    """Asking for more candidates than vocab_size should clamp silently."""
    model_fn = make_mock_model(seed=3)
    healer = TokenHealer(model_fn)
    prefix = make_ids(B=1, T=4)
    removed = make_ids(B=1, T=1)
    # Request more than VOCAB_SIZE
    candidates = healer.get_continuation_candidates(prefix, removed, top_k=VOCAB_SIZE + 100)
    assert candidates.shape[1] == VOCAB_SIZE


# ---------------------------------------------------------------------------
# 6. TokenHealer.heal — output shape matches input
# ---------------------------------------------------------------------------


def test_heal_output_shape_matches_input():
    model_fn = make_mock_model(seed=4)
    healer = TokenHealer(model_fn)
    ids = make_ids(B=1, T=SEQ_LEN)
    healed = healer.heal(ids)
    assert healed.shape == ids.shape, (
        f"heal() output shape {healed.shape} != input shape {ids.shape}"
    )


def test_heal_prefix_unchanged():
    """All tokens except the last should be unchanged after healing."""
    model_fn = make_mock_model(seed=5)
    healer = TokenHealer(model_fn)
    ids = make_ids(B=1, T=SEQ_LEN)
    healed = healer.heal(ids)
    assert torch.equal(healed[:, :-1], ids[:, :-1]), "Prefix tokens should be unchanged"


def test_heal_last_token_in_vocab():
    model_fn = make_mock_model(seed=6)
    healer = TokenHealer(model_fn)
    ids = make_ids(B=1, T=6)
    healed = healer.heal(ids)
    last_tok = healed[0, -1].item()
    assert 0 <= last_tok < VOCAB_SIZE


# ---------------------------------------------------------------------------
# 7. TokenHealer.heal_and_continue — output length
# ---------------------------------------------------------------------------


def test_heal_and_continue_length():
    """heal_and_continue should return input_len + n_new tokens."""
    model_fn = make_mock_model(seed=7)
    healer = TokenHealer(model_fn)
    n_new = 5
    ids = make_ids(B=1, T=SEQ_LEN)
    result = healer.heal_and_continue(ids, n_new=n_new)
    expected_len = SEQ_LEN + n_new
    assert result.shape == (1, expected_len), f"Expected (1, {expected_len}), got {result.shape}"


def test_heal_and_continue_zero_new_tokens():
    """With n_new=0 the result should be the same length as input (just healed)."""
    model_fn = make_mock_model(seed=8)
    healer = TokenHealer(model_fn)
    ids = make_ids(B=1, T=6)
    result = healer.heal_and_continue(ids, n_new=0)
    assert result.shape == ids.shape


# ---------------------------------------------------------------------------
# 8. greedy_extend
# ---------------------------------------------------------------------------


def test_greedy_extend_length():
    model_fn = make_mock_model(seed=9)
    ids = make_ids(B=1, T=4)
    n_steps = 6
    out = greedy_extend(model_fn, ids, n_steps)
    assert out.shape == (1, 4 + n_steps)


def test_greedy_extend_zero_steps():
    model_fn = make_mock_model(seed=10)
    ids = make_ids(B=1, T=5)
    out = greedy_extend(model_fn, ids, 0)
    assert torch.equal(out, ids)


def test_greedy_extend_tokens_in_vocab():
    model_fn = make_mock_model(seed=11)
    ids = make_ids(B=1, T=3)
    out = greedy_extend(model_fn, ids, 4)
    new_tokens = out[:, 3:]
    assert (new_tokens >= 0).all()
    assert (new_tokens < VOCAB_SIZE).all()


# ---------------------------------------------------------------------------
# 9. Logit bias blocks forbidden tokens end-to-end (argmax test)
# ---------------------------------------------------------------------------


def test_logit_bias_blocks_all_but_one():
    """When only one token is allowed, argmax must be that token."""
    only_allowed = 13
    logits = torch.randn(1, VOCAB_SIZE)
    bias = get_token_prefix_logit_bias(VOCAB_SIZE, [only_allowed])
    biased = apply_logit_bias(logits, bias)
    assert biased.argmax(dim=-1).item() == only_allowed
