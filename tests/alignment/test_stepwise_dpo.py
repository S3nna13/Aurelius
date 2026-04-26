"""Tests for src/alignment/stepwise_dpo.py

Uses a tiny 2-layer transformer (vocab_size=256, d_model=64) to verify
Stepwise DPO behaviour end-to-end without any HuggingFace dependency.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.alignment.stepwise_dpo import (
    ReasoningStep,
    StepwiseDPOConfig,
    StepwiseDPOTrainer,
    label_steps_by_prefix_match,
    parse_reasoning_steps,
)

# ---------------------------------------------------------------------------
# Tiny 2-layer transformer (no HuggingFace)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
SEQ_LEN = 16


class _TinyTransformer(nn.Module):
    """Minimal causal transformer: embedding -> 2x TransformerEncoderLayer -> lm_head."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns (B, T, vocab_size) logits."""
        x = self.embed(input_ids)  # (B, T, D)
        x = self.transformer(x)  # (B, T, D)
        return self.lm_head(x)  # (B, T, V)


def _make_model() -> _TinyTransformer:
    torch.manual_seed(0)
    return _TinyTransformer()


def _make_trainer(beta: float = 0.1, step_weight_decay: float = 0.9) -> StepwiseDPOTrainer:
    policy = _make_model()
    ref = copy.deepcopy(policy)
    return StepwiseDPOTrainer(
        policy=policy,
        ref_policy=ref,
        beta=beta,
        step_weight_decay=step_weight_decay,
    )


def _make_steps(n: int, step_len: int = 4, vocab_size: int = VOCAB_SIZE) -> list[ReasoningStep]:
    torch.manual_seed(1)
    return [
        ReasoningStep(
            step_ids=torch.randint(1, vocab_size, (step_len,)),
            is_correct=(i % 2 == 0),
            step_reward=float(i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 1: StepwiseDPOConfig defaults correct
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = StepwiseDPOConfig()
    assert cfg.beta == 0.1
    assert cfg.step_weight_decay == 0.9
    assert cfg.normalize_weights is True
    assert cfg.min_step_weight == 0.01
    assert cfg.aggregate_method == "weighted_sum"


# ---------------------------------------------------------------------------
# Test 2: compute_step_weights returns correct shape (n_steps,)
# ---------------------------------------------------------------------------


def test_step_weights_shape():
    trainer = _make_trainer()
    for n in (1, 3, 5, 10):
        w = trainer.compute_step_weights(n)
        assert w.shape == (n,), f"Expected ({n},), got {w.shape}"


# ---------------------------------------------------------------------------
# Test 3: compute_step_weights sums to 1.0 when normalize=True
# ---------------------------------------------------------------------------


def test_step_weights_sum_to_one():
    trainer = _make_trainer()
    for n in (1, 3, 7):
        w = trainer.compute_step_weights(n)
        assert abs(w.sum().item() - 1.0) < 1e-5, (
            f"n={n}: weights sum to {w.sum().item()}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Test 4: Earlier steps have lower weight (decay^(n-i) pattern)
# ---------------------------------------------------------------------------


def test_step_weights_decay_pattern():
    """With decay=0.9, the spec formula is decay^(n_steps - step_idx).
    With step_idx in [0, n-1]:
      - step 0 (earliest): exponent = n_steps (largest), smallest weight when decay<1
      - step n-1 (latest): exponent = 1 (smallest), largest weight when decay<1
    So later steps have HIGHER weight (more credit for steps closer to the answer).
    After normalization the relative ordering is preserved.
    """
    trainer = _make_trainer(step_weight_decay=0.9)
    n = 4
    w = trainer.compute_step_weights(n)  # normalized

    # step n-1 (latest, closest to answer) should have highest weight
    assert w[-1].item() > w[0].item(), (
        f"Expected w[-1] > w[0], got w[-1]={w[-1].item():.4f}, w[0]={w[0].item():.4f}"
    )
    # Strictly increasing (later steps weighted more)
    for i in range(n - 1):
        assert w[i].item() < w[i + 1].item(), (
            f"Expected w[{i}] < w[{i + 1}], got {w[i].item():.4f} vs {w[i + 1].item():.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: parse_reasoning_steps splits at separator correctly
# ---------------------------------------------------------------------------


def test_parse_reasoning_steps_splits():
    # [1, 2, 0, 3, 4, 0, 5]  separator=0  -> [[1,2], [3,4], [5]]
    ids = torch.tensor([1, 2, 0, 3, 4, 0, 5])
    steps = parse_reasoning_steps(ids, separator_id=0)
    assert len(steps) == 3
    assert steps[0].tolist() == [1, 2]
    assert steps[1].tolist() == [3, 4]
    assert steps[2].tolist() == [5]


# ---------------------------------------------------------------------------
# Test 6: parse_reasoning_steps handles no separator (returns single step)
# ---------------------------------------------------------------------------


def test_parse_reasoning_steps_no_separator():
    ids = torch.tensor([1, 2, 3, 4, 5])
    steps = parse_reasoning_steps(ids, separator_id=0)
    assert len(steps) == 1
    assert steps[0].tolist() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Test 7: label_steps_by_prefix_match returns correct bool list
# ---------------------------------------------------------------------------


def test_label_steps_by_prefix_match():
    steps = [
        torch.tensor([10, 20, 30]),  # first token 10
        torch.tensor([99, 1, 2]),  # first token 99
        torch.tensor([5, 6, 7]),  # first token 5
    ]
    correct = [
        torch.tensor([10, 99]),  # first token 10 - matches
        torch.tensor([50, 1]),  # first token 50 - no match
        torch.tensor([5]),  # first token 5 - matches
    ]
    labels = label_steps_by_prefix_match(steps, correct)
    assert labels == [True, False, True]


def test_label_steps_extra_steps_are_false():
    """Steps beyond len(correct_steps) should be labeled False."""
    steps = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
    correct = [torch.tensor([1])]  # only one correct step
    labels = label_steps_by_prefix_match(steps, correct)
    assert labels == [True, False, False]


# ---------------------------------------------------------------------------
# Test 8: compute_stepwise_dpo_loss returns scalar tensor
# ---------------------------------------------------------------------------


def test_compute_stepwise_dpo_loss_scalar():
    trainer = _make_trainer()
    n = 3
    step_len = 4

    chosen_steps = _make_steps(n, step_len)
    rejected_steps = _make_steps(n, step_len)
    context_ids = torch.randint(1, VOCAB_SIZE, (8,))

    chosen_logps = trainer.compute_step_log_probs(
        trainer.policy, [s.step_ids for s in chosen_steps], context_ids
    )
    rejected_logps = trainer.compute_step_log_probs(
        trainer.policy, [s.step_ids for s in rejected_steps], context_ids
    )

    with torch.no_grad():
        ref_chosen_logps = trainer.compute_step_log_probs(
            trainer.ref_policy, [s.step_ids for s in chosen_steps], context_ids
        )
        ref_rejected_logps = trainer.compute_step_log_probs(
            trainer.ref_policy, [s.step_ids for s in rejected_steps], context_ids
        )

    loss, metrics = trainer.compute_stepwise_dpo_loss(
        chosen_steps,
        rejected_steps,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, f"Expected scalar (0-D), got shape {loss.shape}"


# ---------------------------------------------------------------------------
# Test 9: Metrics dict has required keys
# ---------------------------------------------------------------------------


def test_metrics_keys():
    trainer = _make_trainer()
    n = 3
    step_len = 4

    chosen_steps = _make_steps(n, step_len)
    rejected_steps = _make_steps(n, step_len)
    context_ids = torch.randint(1, VOCAB_SIZE, (8,))

    chosen_logps = trainer.compute_step_log_probs(
        trainer.policy, [s.step_ids for s in chosen_steps], context_ids
    )
    rejected_logps = trainer.compute_step_log_probs(
        trainer.policy, [s.step_ids for s in rejected_steps], context_ids
    )

    with torch.no_grad():
        ref_chosen_logps = trainer.compute_step_log_probs(
            trainer.ref_policy, [s.step_ids for s in chosen_steps], context_ids
        )
        ref_rejected_logps = trainer.compute_step_log_probs(
            trainer.ref_policy, [s.step_ids for s in rejected_steps], context_ids
        )

    _, metrics = trainer.compute_stepwise_dpo_loss(
        chosen_steps,
        rejected_steps,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
    )

    required_keys = {
        "loss",
        "mean_chosen_reward",
        "mean_rejected_reward",
        "reward_accuracy",
        "n_steps",
        "step_weights",
    }
    assert required_keys.issubset(set(metrics.keys())), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 10: train_step returns dict with loss key
# ---------------------------------------------------------------------------


def test_train_step_returns_loss():
    trainer = _make_trainer()
    n = 2
    chosen_steps = _make_steps(n)
    rejected_steps = _make_steps(n)
    context_ids = torch.randint(1, VOCAB_SIZE, (6,))

    result = trainer.train_step(chosen_steps, rejected_steps, context_ids)

    assert isinstance(result, dict), "train_step must return a dict"
    assert "loss" in result, "train_step result must have 'loss' key"
    assert isinstance(result["loss"], torch.Tensor), "loss must be a torch.Tensor"


# ---------------------------------------------------------------------------
# Test 11: Gradient flows through stepwise DPO loss
# ---------------------------------------------------------------------------


def test_gradient_flows():
    trainer = _make_trainer()
    n = 2
    chosen_steps = _make_steps(n, step_len=3)
    rejected_steps = _make_steps(n, step_len=3)
    context_ids = torch.randint(1, VOCAB_SIZE, (5,))

    result = trainer.train_step(chosen_steps, rejected_steps, context_ids)
    loss = result["loss"]

    # Loss should require grad since policy parameters are trainable
    assert loss.requires_grad, "Loss must require gradients"

    # Backward should succeed without errors
    loss.backward()

    # At least one policy parameter should have a non-None gradient
    grads = [p.grad for p in trainer.policy.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found in policy parameters after backward()"

    # Gradient magnitudes should be finite
    for g in grads:
        assert torch.isfinite(g).all(), "Found non-finite gradient values"
