"""Tests for Online DPO module (src/training/online_dpo.py).

15 tests covering OnlinePairGenerator, IPOLoss, SLiCLoss,
DPOVariantTrainer, and OnlineDPOTrainer.

Tiny config: d_model=16, vocab=16, seq_len=8, batch=2,
             n_candidates=3, max_new_tokens=4.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.training.online_dpo import (
    DPOVariantTrainer,
    IPOLoss,
    OnlineDPOTrainer,
    OnlinePairGenerator,
    SLiCLoss,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 16
SEQ_LEN = 8
BATCH = 2
N_CANDS = 3
MAX_NEW = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal Embedding + Linear language model for testing."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))  # (B, T, V)


def make_models() -> tuple[TinyLM, TinyLM]:
    policy = TinyLM()
    ref = copy.deepcopy(policy)
    return policy, ref


def make_reward_fn(vocab: int = VOCAB) -> object:
    """Return a simple reward function that scores by mean token id."""
    def reward_fn(generated_ids: torch.Tensor) -> float:
        return generated_ids.float().mean().item()
    return reward_fn


def make_pair_generator(
    policy: TinyLM,
    n_candidates: int = N_CANDS,
) -> OnlinePairGenerator:
    return OnlinePairGenerator(
        model=policy,
        reward_fn=make_reward_fn(),
        n_candidates=n_candidates,
        temperature=0.8,
    )


def make_variant_trainer(
    loss_type: str = "dpo",
) -> tuple[DPOVariantTrainer, TinyLM, TinyLM]:
    policy, ref = make_models()
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    trainer = DPOVariantTrainer(
        policy_model=policy,
        ref_model=ref,
        optimizer=opt,
        loss_type=loss_type,
        beta=0.1,
    )
    return trainer, policy, ref


def make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    rejected_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return chosen_ids, chosen_ids.clone(), rejected_ids, rejected_ids.clone()


def make_online_trainer(
    loss_type: str = "dpo",
) -> tuple[OnlineDPOTrainer, TinyLM, TinyLM]:
    policy, ref = make_models()
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    gen = make_pair_generator(policy)
    trainer = OnlineDPOTrainer(
        policy_model=policy,
        ref_model=ref,
        reward_fn=make_reward_fn(),
        optimizer=opt,
        pair_generator=gen,
        loss_type=loss_type,
    )
    return trainer, policy, ref


# ---------------------------------------------------------------------------
# OnlinePairGenerator tests
# ---------------------------------------------------------------------------

def test_pair_generator_returns_tensors() -> None:
    """generate_pair must return (chosen, rejected) as Tensors."""
    policy, _ = make_models()
    gen = make_pair_generator(policy)
    prompt = torch.randint(0, VOCAB, (1, SEQ_LEN))
    result = gen.generate_pair(prompt, max_new_tokens=MAX_NEW)
    if result is None:
        pytest.skip("All candidates tied -- rare but valid.")
    chosen_ids, rejected_ids, chosen_r, rejected_r = result
    assert isinstance(chosen_ids, torch.Tensor), "chosen_ids must be a Tensor"
    assert isinstance(rejected_ids, torch.Tensor), "rejected_ids must be a Tensor"
    assert chosen_ids.dim() == 2, "chosen_ids must be 2-D (1, T)"
    assert rejected_ids.dim() == 2, "rejected_ids must be 2-D (1, T)"


def test_pair_generator_chosen_reward_ge_rejected() -> None:
    """chosen_reward must be >= rejected_reward when a pair is returned."""
    policy, _ = make_models()
    gen = make_pair_generator(policy)
    prompt = torch.randint(0, VOCAB, (1, SEQ_LEN))
    result = gen.generate_pair(prompt, max_new_tokens=MAX_NEW)
    if result is None:
        pytest.skip("All candidates tied -- rare but valid.")
    _, _, chosen_r, rejected_r = result
    assert chosen_r >= rejected_r, (
        f"chosen_reward {chosen_r} must be >= rejected_reward {rejected_r}"
    )


def test_pair_generator_n_candidates_one() -> None:
    """n_candidates=1 -- single candidate, no comparison, must return None."""
    policy, _ = make_models()
    gen = OnlinePairGenerator(
        model=policy,
        reward_fn=make_reward_fn(),
        n_candidates=1,
        temperature=0.8,
    )
    prompt = torch.randint(0, VOCAB, (1, SEQ_LEN))
    # With one candidate best == worst, so always returns None
    result = gen.generate_pair(prompt, max_new_tokens=MAX_NEW)
    assert result is None, "n_candidates=1 must always return None (tied)"


# ---------------------------------------------------------------------------
# IPOLoss tests
# ---------------------------------------------------------------------------

def test_ipo_loss_scalar_finite() -> None:
    """IPO loss must be a finite scalar."""
    ipo = IPOLoss(tau=0.1)
    policy_c = torch.randn(BATCH, requires_grad=True)
    policy_r = torch.randn(BATCH, requires_grad=True)
    ref_c = torch.randn(BATCH)
    ref_r = torch.randn(BATCH)
    loss, h_c, h_r = ipo(policy_c, policy_r, ref_c, ref_r)
    assert loss.shape == torch.Size([]), "IPO loss must be scalar"
    assert torch.isfinite(loss), "IPO loss must be finite"
    assert torch.isfinite(h_c).all(), "h_chosen must be finite"
    assert torch.isfinite(h_r).all(), "h_rejected must be finite"


def test_ipo_loss_well_calibrated_near_zero() -> None:
    """When chosen >> rejected the IPO loss should be small."""
    ipo = IPOLoss(tau=0.1)
    target = 1.0 / (2.0 * 0.1)  # = 5.0
    # Construct inputs so h_w - h_l == target exactly
    # h_w = policy_c - ref_c; h_l = policy_r - ref_r
    ref_c = torch.zeros(BATCH)
    ref_r = torch.zeros(BATCH)
    policy_c = torch.full((BATCH,), target, requires_grad=True)
    policy_r = torch.zeros(BATCH, requires_grad=True)
    loss, _, _ = ipo(policy_c, policy_r, ref_c, ref_r)
    assert loss.item() < 1e-6, f"Well-calibrated IPO loss should be near 0, got {loss.item()}"


def test_ipo_loss_symmetry() -> None:
    """Swapping chosen/rejected swaps the h values (opposite implicit rewards).

    IPO loss is NOT symmetric (the target 1/(2*tau) has a sign), but the
    implicit reward values must swap roles: h_c_fwd == h_r_rev and
    h_r_fwd == h_c_rev.
    """
    ipo = IPOLoss(tau=0.1)
    torch.manual_seed(0)
    policy_c = torch.randn(BATCH, requires_grad=True)
    policy_r = torch.randn(BATCH, requires_grad=True)
    ref_c = torch.randn(BATCH)
    ref_r = torch.randn(BATCH)

    loss_fwd, h_c_fwd, h_r_fwd = ipo(policy_c, policy_r, ref_c, ref_r)

    # Swap chosen <-> rejected (and their reference counterparts)
    policy_c2 = policy_r.detach().requires_grad_(True)
    policy_r2 = policy_c.detach().requires_grad_(True)
    loss_rev, h_c_rev, h_r_rev = ipo(policy_c2, policy_r2, ref_r, ref_c)

    # h_chosen of forward pass == h_rejected of reversed pass (same sequences)
    assert torch.allclose(h_c_fwd.detach(), h_r_rev.detach(), atol=1e-5), (
        "h_chosen (fwd) must equal h_rejected (rev) -- same sequence, swapped roles"
    )
    # h_rejected of forward pass == h_chosen of reversed pass
    assert torch.allclose(h_r_fwd.detach(), h_c_rev.detach(), atol=1e-5), (
        "h_rejected (fwd) must equal h_chosen (rev) -- same sequence, swapped roles"
    )
    # Both losses must still be finite
    assert torch.isfinite(loss_fwd) and torch.isfinite(loss_rev), "IPO losses must be finite"


# ---------------------------------------------------------------------------
# SLiCLoss tests
# ---------------------------------------------------------------------------

def test_slic_rank_loss_nonnegative() -> None:
    """rank_loss (hinge) must always be >= 0."""
    slic = SLiCLoss(delta=1.0, lm_weight=0.1)
    policy_c = torch.randn(BATCH, requires_grad=True)
    policy_r = torch.randn(BATCH, requires_grad=True)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    policy_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
    _, rank_loss, reg_loss = slic(policy_c, policy_r, input_ids, policy_logits)
    assert rank_loss.item() >= 0.0, f"rank_loss must be >= 0, got {rank_loss.item()}"
    assert torch.isfinite(reg_loss), "reg_loss must be finite"


def test_slic_margin_satisfied_rank_loss_zero() -> None:
    """When chosen - rejected > delta, rank_loss must be 0."""
    slic = SLiCLoss(delta=1.0, lm_weight=0.1)
    # chosen logp >> rejected logp by more than delta=1.0
    policy_c = torch.full((BATCH,), 3.0, requires_grad=True)
    policy_r = torch.full((BATCH,), 0.0, requires_grad=True)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    policy_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
    _, rank_loss, _ = slic(policy_c, policy_r, input_ids, policy_logits)
    assert rank_loss.item() < 1e-6, (
        f"rank_loss should be 0 when margin satisfied, got {rank_loss.item()}"
    )


# ---------------------------------------------------------------------------
# DPOVariantTrainer tests
# ---------------------------------------------------------------------------

def test_dpo_trainer_accuracy_in_range() -> None:
    """DPO trainer accuracy must be in [0, 1]."""
    trainer, _, _ = make_variant_trainer("dpo")
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    acc = result["accuracy"].item()
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} not in [0, 1]"
    assert torch.isfinite(result["loss"]), "DPO loss must be finite"


def test_ipo_trainer_accuracy_and_reward_diff_finite() -> None:
    """IPO trainer accuracy in [0,1], implicit_reward_diff finite."""
    trainer, _, _ = make_variant_trainer("ipo")
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    acc = result["accuracy"].item()
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} not in [0, 1]"
    assert torch.isfinite(result["implicit_reward_diff"]), "implicit_reward_diff must be finite"


def test_slic_trainer_loss_finite_grad_flows() -> None:
    """SLiC trainer loss must be finite and grad must flow to policy params."""
    policy, ref = make_models()
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    trainer = DPOVariantTrainer(
        policy_model=policy,
        ref_model=ref,
        optimizer=opt,
        loss_type="slic",
        beta=0.1,
    )
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    assert torch.isfinite(result["loss"]), "SLiC loss must be finite"
    # Check at least one parameter received a gradient
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in policy.parameters()
    )
    assert has_grad, "No finite gradient reached policy parameters in SLiC step"


def test_variant_trainer_ref_params_unchanged() -> None:
    """Reference model parameters must not change after a train_step."""
    policy, ref = make_models()
    ref_params_before = [p.data.clone() for p in ref.parameters()]

    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    trainer = DPOVariantTrainer(
        policy_model=policy,
        ref_model=ref,
        optimizer=opt,
        loss_type="dpo",
        beta=0.1,
    )
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)

    for before, after in zip(ref_params_before, ref.parameters()):
        assert torch.equal(before, after.data), "ref model params changed after train_step"


# ---------------------------------------------------------------------------
# OnlineDPOTrainer tests
# ---------------------------------------------------------------------------

def test_online_step_keys_present() -> None:
    """online_step must return dict with loss, n_valid_pairs, mean_reward_gap."""
    trainer, _, _ = make_online_trainer()
    prompts = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.online_step(prompts, max_new_tokens=MAX_NEW)
    for key in ("loss", "n_valid_pairs", "mean_reward_gap"):
        assert key in result, f"Missing key '{key}' in online_step result"
    assert 0 <= result["n_valid_pairs"] <= BATCH


def test_online_ref_update_changes_ref_params() -> None:
    """ref_update_step with alpha in (0,1) should change ref params."""
    trainer, policy, ref = make_online_trainer()

    # Ensure policy and ref differ
    with torch.no_grad():
        for p in policy.parameters():
            p.data.add_(torch.randn_like(p) * 0.5)

    ref_before = [p.data.clone() for p in ref.parameters()]
    trainer.ref_update_step(alpha=0.1)
    ref_after = [p.data.clone() for p in ref.parameters()]

    changed = any(
        not torch.equal(b, a) for b, a in zip(ref_before, ref_after)
    )
    assert changed, "ref params should change after ref_update_step with alpha=0.1"


def test_online_ref_update_alpha_zero_no_change() -> None:
    """alpha=0 must leave ref params completely unchanged."""
    trainer, policy, ref = make_online_trainer()

    # Diverge policy from ref first
    with torch.no_grad():
        for p in policy.parameters():
            p.data.add_(1.0)

    ref_before = [p.data.clone() for p in ref.parameters()]
    trainer.ref_update_step(alpha=0.0)

    for before, after in zip(ref_before, ref.parameters()):
        assert torch.equal(before, after.data), "ref params changed with alpha=0"


def test_online_ref_update_alpha_one_equals_policy() -> None:
    """alpha=1 must copy policy params exactly into ref."""
    trainer, policy, ref = make_online_trainer()

    # Diverge ref from policy
    with torch.no_grad():
        for p in ref.parameters():
            p.data.fill_(0.0)

    trainer.ref_update_step(alpha=1.0)

    for pol_p, ref_p in zip(policy.parameters(), ref.parameters()):
        assert torch.allclose(pol_p.data, ref_p.data, atol=1e-6), (
            "After alpha=1 ref params must equal policy params"
        )


def test_online_loop_loss_remains_finite() -> None:
    """Three consecutive online_steps must all produce finite loss."""
    trainer, _, _ = make_online_trainer()
    prompts = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))

    for step in range(3):
        result = trainer.online_step(prompts, max_new_tokens=MAX_NEW)
        loss_val = result["loss"]
        if isinstance(loss_val, torch.Tensor):
            assert torch.isfinite(loss_val), f"Step {step}: loss is not finite"
        else:
            assert loss_val == loss_val and loss_val != float("inf"), (
                f"Step {step}: loss is not finite"
            )
        trainer.ref_update_step(alpha=0.05)
