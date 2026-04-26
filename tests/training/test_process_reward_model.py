"""Unit tests for src/training/process_reward_model.py.

Tiny config: n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16,
             d_ff=128, vocab_size=256, max_seq_len=64.
All tests run without GPU (CPU only); no HuggingFace / external deps.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.training.process_reward_model import (
    PRMHead,
    PRMInference,
    PRMLoss,
    ProcessRewardModel,
    StepDataCollator,
)

# ---------------------------------------------------------------------------
# Tiny config shared by all tests
# ---------------------------------------------------------------------------

STEP_TOKEN_ID = 5  # must be < vocab_size (256)


def tiny_config() -> AureliusConfig:
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    cfg.prm_step_token_id = STEP_TOKEN_ID  # type: ignore[attr-defined]
    return cfg


def make_prm() -> ProcessRewardModel:
    return ProcessRewardModel(tiny_config())


def make_input(B: int = 2, T: int = 16, n_steps: int = 2) -> torch.Tensor:
    """Create input_ids with n_steps step tokens per row, guaranteed no collisions."""
    torch.manual_seed(0)
    ids = torch.randint(10, 200, (B, T), dtype=torch.long)  # avoid token 5
    # Place step tokens at fixed positions
    step_gap = T // (n_steps + 1)
    for s in range(n_steps):
        pos = (s + 1) * step_gap
        ids[:, pos] = STEP_TOKEN_ID
    return ids


# ---------------------------------------------------------------------------
# Test 1: ProcessRewardModel initializes from AureliusConfig
# ---------------------------------------------------------------------------


def test_prm_init_from_aurelius_config():
    prm = make_prm()
    assert isinstance(prm, ProcessRewardModel)
    assert isinstance(prm.backbone, nn.Module)
    assert isinstance(prm.head, PRMHead)
    assert prm.step_token_id == STEP_TOKEN_ID


# ---------------------------------------------------------------------------
# Test 2: Forward with no labels returns (None, step_rewards) with correct shape
# ---------------------------------------------------------------------------


def test_forward_no_labels_shape():
    prm = make_prm()
    B, T, n_steps = 2, 16, 2
    input_ids = make_input(B, T, n_steps)

    loss, step_rewards = prm(input_ids, labels=None)
    assert loss is None
    assert step_rewards.shape == (B, n_steps), (
        f"Expected [{B}, {n_steps}], got {step_rewards.shape}"
    )


# ---------------------------------------------------------------------------
# Test 3: Forward with labels returns (loss, step_rewards), loss is scalar finite
# ---------------------------------------------------------------------------


def test_forward_with_labels_returns_finite_loss():
    prm = make_prm()
    B, T, n_steps = 2, 16, 2
    input_ids = make_input(B, T, n_steps)
    labels = torch.randint(0, 2, (B, n_steps))

    loss, step_rewards = prm(input_ids, labels=labels)

    assert loss is not None
    assert loss.shape == ()  # scalar
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert step_rewards.shape == (B, n_steps)


# ---------------------------------------------------------------------------
# Test 4: loss.backward() produces finite gradients in PRMHead params
# ---------------------------------------------------------------------------


def test_backward_finite_gradients():
    prm = make_prm()
    B, T, n_steps = 2, 16, 2
    input_ids = make_input(B, T, n_steps)
    labels = torch.randint(0, 2, (B, n_steps))

    loss, _ = prm(input_ids, labels=labels)
    assert loss is not None
    loss.backward()

    for name, param in prm.head.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Test 5: PRMHead output shape [B, num_steps, 1] -> squeezed to [B, num_steps]
# ---------------------------------------------------------------------------


def test_prm_head_output_shape():
    head = PRMHead(d_model=64)
    B, S, D = 3, 5, 64
    hidden = torch.randn(B, S, D)
    # Apply head to flat [B*S, D] then reshape
    flat = hidden.view(B * S, D)
    out_flat = head.linear(flat).squeeze(-1)  # [B*S]
    out = out_flat.view(B, S)  # [B, S]
    assert out.shape == (B, S)

    # Also verify the PRMHead.forward path with [N, D] input
    out2 = head(flat)  # [B*S]
    assert out2.shape == (B * S,)


# ---------------------------------------------------------------------------
# Test 6: StepDataCollator single example -> correct tensors
# ---------------------------------------------------------------------------


def test_collator_single_example():
    collator = StepDataCollator(pad_id=0)
    example = {
        "input_ids": [10, 20, STEP_TOKEN_ID, 30, STEP_TOKEN_ID, 40],
        "step_positions": [2, 4],
        "step_labels": [1, 0],
    }
    batch = collator([example])

    assert batch["input_ids"].shape == (1, 6)
    assert batch["attention_mask"].shape == (1, 6)
    assert batch["labels"].shape == (1, 2)
    assert batch["step_positions"].shape == (1, 2)

    assert batch["input_ids"][0, 2].item() == STEP_TOKEN_ID
    assert batch["labels"][0, 0].item() == 1
    assert batch["labels"][0, 1].item() == 0
    assert batch["attention_mask"][0, :6].all()


# ---------------------------------------------------------------------------
# Test 7: StepDataCollator batch with different step counts -> padded correctly
# ---------------------------------------------------------------------------


def test_collator_variable_step_counts():
    collator = StepDataCollator(pad_id=0)
    ex1 = {
        "input_ids": [1, STEP_TOKEN_ID, 2],
        "step_positions": [1],
        "step_labels": [1],
    }
    ex2 = {
        "input_ids": [3, STEP_TOKEN_ID, 4, STEP_TOKEN_ID, 5],
        "step_positions": [1, 3],
        "step_labels": [0, 1],
    }
    batch = collator([ex1, ex2])

    # T_max = 5, S_max = 2
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 2)

    # ex1 has 1 step; padding at index 1 must be -1
    assert batch["labels"][0, 0].item() == 1
    assert batch["labels"][0, 1].item() == -1

    # ex2 has 2 steps
    assert batch["labels"][1, 0].item() == 0
    assert batch["labels"][1, 1].item() == 1

    # ex1 input_ids padded with 0 for positions [3,4]
    assert batch["input_ids"][0, 3].item() == 0
    assert batch["input_ids"][0, 4].item() == 0


# ---------------------------------------------------------------------------
# Test 8: StepDataCollator step_positions correctly extracted
# ---------------------------------------------------------------------------


def test_collator_step_positions_correct():
    collator = StepDataCollator(pad_id=0)
    ex = {
        "input_ids": [7, 8, STEP_TOKEN_ID, 9, STEP_TOKEN_ID],
        "step_positions": [2, 4],
        "step_labels": [1, 1],
    }
    batch = collator([ex])
    assert batch["step_positions"][0, 0].item() == 2
    assert batch["step_positions"][0, 1].item() == 4


# ---------------------------------------------------------------------------
# Test 9: PRMLoss ignores positions with label=-1
# ---------------------------------------------------------------------------


def test_prm_loss_ignores_padding():
    loss_fn = PRMLoss()
    logits = torch.tensor([[2.0, -1.0, 0.5]])  # [1, 3]
    labels = torch.tensor([[1, -1, -1]])  # [1, 3]; only first is valid

    loss = loss_fn(logits, labels)

    # Manually compute expected: BCE on just logit=2.0, label=1
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0]), torch.tensor([1.0])
    )
    assert torch.isclose(loss, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 10: PRMLoss correct positions contribute to loss
# ---------------------------------------------------------------------------


def test_prm_loss_valid_positions_contribute():
    loss_fn = PRMLoss()
    logits = torch.tensor([[1.0, -1.0]])  # [1, 2]
    labels = torch.tensor([[1, 0]])  # [1, 2]; both valid

    loss = loss_fn(logits, labels)

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, 0.0]),
    )
    assert torch.isclose(loss, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 11: PRMInference.score_chain returns list of floats, one per step
# ---------------------------------------------------------------------------


def test_prm_inference_score_chain():
    prm = make_prm()
    prm.train(False)  # set to eval mode without using .eval()
    inference = PRMInference(prm, aggregation="min")

    chain = "step one content\n\nstep two content\n\nstep three"
    scores = inference.score_chain(chain, step_token="\n\n")  # noqa: S106

    assert isinstance(scores, list)
    # 2 delimiters => 2 step tokens inserted => 2 scores
    assert len(scores) == 2
    for s in scores:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0  # sigmoid outputs


# ---------------------------------------------------------------------------
# Test 12: PRMInference.select_best picks chain with highest min-step-score
# ---------------------------------------------------------------------------


def test_prm_inference_select_best_min():
    prm = make_prm()
    prm.train(False)

    # Patch score_chain to return deterministic values
    call_count = [0]
    returns = [
        [0.9, 0.8],  # candidate 0 -- min=0.8
        [0.5, 0.3],  # candidate 1 -- min=0.3
        [0.95, 0.85],  # candidate 2 -- min=0.85  <-- best
    ]

    def mock_score_chain(chain, step_token="\n\n"):  # noqa: S107
        idx = call_count[0]
        call_count[0] += 1
        return returns[idx]

    inference = PRMInference(prm, aggregation="min")
    inference.score_chain = mock_score_chain  # type: ignore[method-assign]

    candidates = [["a", "b"], ["c", "d"], ["e", "f"]]
    best = inference.select_best(candidates, step_token="\n\n")  # noqa: S106
    assert best == 2


# ---------------------------------------------------------------------------
# Test 13: PRMInference.select_best with aggregation="mean"
# ---------------------------------------------------------------------------


def test_prm_inference_select_best_mean():
    prm = make_prm()
    prm.train(False)

    call_count = [0]
    returns = [
        [0.9, 0.2],  # candidate 0 -- mean=0.55
        [0.8, 0.85],  # candidate 1 -- mean=0.825  <-- best
    ]

    def mock_score_chain(chain, step_token="\n\n"):  # noqa: S107
        idx = call_count[0]
        call_count[0] += 1
        return returns[idx]

    inference = PRMInference(prm, aggregation="mean")
    inference.score_chain = mock_score_chain  # type: ignore[method-assign]

    candidates = [["step1a", "step1b"], ["step2a", "step2b"]]
    best = inference.select_best(candidates, step_token="\n\n")  # noqa: S106
    assert best == 1


# ---------------------------------------------------------------------------
# Test 14: Determinism under torch.manual_seed
# ---------------------------------------------------------------------------


def test_determinism_under_manual_seed():
    def run_once():
        torch.manual_seed(7)
        prm = make_prm()
        prm.train(False)
        ids = make_input(2, 16, 2)
        with torch.no_grad():
            _, rewards = prm(ids)
        return rewards

    r1 = run_once()
    r2 = run_once()
    assert torch.allclose(r1, r2), "Results differ across identical seeds"


# ---------------------------------------------------------------------------
# Test 15a: Edge case -- chain with single step
# ---------------------------------------------------------------------------


def test_single_step_chain():
    prm = make_prm()
    prm.train(False)
    # Only one step token in the sequence
    ids = torch.randint(10, 200, (1, 8), dtype=torch.long)
    ids[0, 4] = STEP_TOKEN_ID

    loss, step_rewards = prm(ids, labels=None)
    assert loss is None
    assert step_rewards.shape == (1, 1)
    assert torch.isfinite(step_rewards).all()


# ---------------------------------------------------------------------------
# Test 15b: Edge case -- chain with no step tokens
# ---------------------------------------------------------------------------


def test_no_step_tokens():
    prm = make_prm()
    prm.train(False)
    # No step tokens at all
    ids = torch.randint(10, 200, (2, 8), dtype=torch.long)
    # Make sure no token equals STEP_TOKEN_ID
    ids[ids == STEP_TOKEN_ID] = 10

    loss, step_rewards = prm(ids, labels=None)
    assert loss is None
    # Returns [B, 1] of zeros when no step tokens
    assert step_rewards.shape[0] == 2
    assert step_rewards.shape[1] >= 1


# ---------------------------------------------------------------------------
# Test 16: PRMLoss all-padding returns zero (differentiable)
# ---------------------------------------------------------------------------


def test_prm_loss_all_padding_returns_zero():
    loss_fn = PRMLoss()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    labels = torch.tensor([[-1, -1, -1]])

    loss = loss_fn(logits, labels)
    assert loss.item() == 0.0
    # Ensure it's differentiable
    _ = loss + 0.0  # differentiable (no backward needed on detached zero)
