"""Integration tests for KTO v2."""

from __future__ import annotations

import torch


def test_exposed_via_src_alignment():
    import src.alignment as A

    assert hasattr(A, "KTOv2Loss")
    assert hasattr(A, "kto_v2_loss_functional")


def test_existing_kto_still_importable():
    # Baseline KTO module must remain importable and untouched in surface.
    import src.alignment.kto as base_kto  # noqa: F401

    assert base_kto is not None


def test_end_to_end_tiny_batch_training_step():
    """Run a handful of optimizer steps over a synthetic mixed batch."""
    from src.alignment import KTOv2Loss

    torch.manual_seed(0)
    # Tiny "policy": a single learnable bias applied to fixed reference logprobs.
    ref = torch.tensor([0.3, -0.2, 0.1, -0.4, 0.0, 0.25, -0.15, 0.05])
    is_des = torch.tensor([True, False, True, False, True, False, True, False])

    bias = torch.nn.Parameter(torch.zeros(()))
    loss_fn = KTOv2Loss(beta_desirable=0.2, beta_undesirable=0.3, z_ref_ema=0.85)
    opt = torch.optim.SGD([bias], lr=0.5)

    losses = []
    for _ in range(15):
        # Policy = ref + bias, but we want the policy to push positive
        # contributions for desirable and negative for undesirable.
        # Simulate directional signal by encoding desirability into the shift.
        shift = torch.where(is_des, bias, -bias)
        policy = ref + shift
        loss = loss_fn(policy, ref, is_des)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    # Loss should decrease overall from first to last step.
    assert losses[-1] < losses[0]
    # z_ref should be non-zero after training.
    assert loss_fn.z_ref.item() != 0.0
