"""Tests for MoE improvements: Switch routing, Z-loss, expert dropout, overflow handling."""

import torch

from src.model.moe_improvements import (
    ExpertWithDropout,
    ImprovedMoELayer,
    MoEImprovementsConfig,
    SwitchRouter,
    add_routing_noise,
    compute_switch_aux_loss,
    compute_z_loss,
)

torch.manual_seed(0)

# Shared test dimensions
D_MODEL = 64
D_FF = 128
N_EXPERTS = 4
CAPACITY_FACTOR = 1.5
B, T = 2, 8


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_moe_improvements_config_defaults():
    cfg = MoEImprovementsConfig()
    assert cfg.n_experts == 8
    assert cfg.top_k == 1
    assert cfg.capacity_factor == 1.25
    assert cfg.z_loss_coeff == 0.001
    assert cfg.expert_dropout == 0.0
    assert cfg.noise_std == 0.01
    assert cfg.use_aux_loss is True


# ---------------------------------------------------------------------------
# 2. Z-loss positive
# ---------------------------------------------------------------------------


def test_compute_z_loss_positive():
    torch.manual_seed(0)
    logits = torch.randn(B * T, N_EXPERTS)
    z = compute_z_loss(logits, coeff=0.001)
    assert z.ndim == 0, "z_loss should be a scalar"
    assert z.item() >= 0.0, "z_loss must be non-negative"


# ---------------------------------------------------------------------------
# 3. Z-loss near zero for zero logits
# ---------------------------------------------------------------------------


def test_compute_z_loss_zero_logits():
    logits = torch.zeros(B * T, N_EXPERTS)
    z = compute_z_loss(logits, coeff=0.001)
    # log(sum(exp(0_vector))) = log(n_experts); should be small for coeff=0.001
    assert z.item() >= 0.0
    # With zero logits: log(4) ≈ 1.386, (1.386)^2 ≈ 1.92, * 0.001 ≈ 0.00192
    assert z.item() < 0.01, f"Expected small z_loss for zero logits, got {z.item()}"


# ---------------------------------------------------------------------------
# 4. aux_loss is scalar
# ---------------------------------------------------------------------------


def test_compute_switch_aux_loss_shape():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(B * T, N_EXPERTS), dim=-1)
    indices = probs.argmax(dim=-1)
    aux = compute_switch_aux_loss(probs, indices)
    assert aux.ndim == 0, "aux_loss should be a scalar"
    assert torch.isfinite(aux), "aux_loss must be finite"


# ---------------------------------------------------------------------------
# 5. Routing noise preserves shape
# ---------------------------------------------------------------------------


def test_add_routing_noise_shape():
    torch.manual_seed(0)
    logits = torch.randn(B * T, N_EXPERTS)
    noisy = add_routing_noise(logits, noise_std=0.01, training=True)
    assert noisy.shape == logits.shape, "Noisy logits must have same shape as input"


# ---------------------------------------------------------------------------
# 6. SwitchRouter output shapes
# ---------------------------------------------------------------------------


def test_switch_router_output_shapes():
    torch.manual_seed(0)
    router = SwitchRouter(D_MODEL, N_EXPERTS, CAPACITY_FACTOR)
    hidden = torch.randn(B, T, D_MODEL)
    router_probs, expert_indices, dispatch_mask, overflow_count = router(hidden)
    N = B * T
    assert router_probs.shape == (N, N_EXPERTS), (
        f"Expected ({N}, {N_EXPERTS}), got {router_probs.shape}"
    )
    assert expert_indices.shape == (N,), f"Expected ({N},), got {expert_indices.shape}"
    assert dispatch_mask.shape == (N,), f"Expected ({N},), got {dispatch_mask.shape}"
    assert dispatch_mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 7. SwitchRouter overflow_count >= 0
# ---------------------------------------------------------------------------


def test_switch_router_capacity():
    torch.manual_seed(0)
    router = SwitchRouter(D_MODEL, N_EXPERTS, CAPACITY_FACTOR)
    hidden = torch.randn(B, T, D_MODEL)
    _, _, dispatch_mask, overflow_count = router(hidden)
    assert overflow_count >= 0, "overflow_count must be non-negative"
    # Total dispatched + overflow should equal total tokens
    assert dispatch_mask.sum().item() + overflow_count == B * T


# ---------------------------------------------------------------------------
# 8. ExpertWithDropout output shape
# ---------------------------------------------------------------------------


def test_expert_with_dropout_shape():
    torch.manual_seed(0)
    expert = ExpertWithDropout(D_MODEL, D_FF, expert_dropout=0.0)
    x = torch.randn(10, D_MODEL)
    out = expert(x, training=False)
    assert out.shape == (10, D_MODEL), f"Expected (10, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 9. Expert with dropout=0 does not change output
# ---------------------------------------------------------------------------


def test_expert_no_dropout_same():
    torch.manual_seed(0)
    expert = ExpertWithDropout(D_MODEL, D_FF, expert_dropout=0.0)
    x = torch.randn(5, D_MODEL)
    out1 = expert(x, training=True)
    out2 = expert(x, training=False)
    # With dropout_p=0.0, both should produce identical outputs
    assert torch.allclose(out1, out2), "dropout_p=0 should yield identical outputs for train/eval"


# ---------------------------------------------------------------------------
# 10. ImprovedMoELayer output shape
# ---------------------------------------------------------------------------


def test_improved_moe_layer_output_shape():
    torch.manual_seed(0)
    cfg = MoEImprovementsConfig(n_experts=N_EXPERTS, capacity_factor=CAPACITY_FACTOR)
    layer = ImprovedMoELayer(D_MODEL, D_FF, cfg)
    hidden = torch.randn(B, T, D_MODEL)
    output, _ = layer(hidden)
    assert output.shape == (B, T, D_MODEL), f"Expected ({B}, {T}, {D_MODEL}), got {output.shape}"


# ---------------------------------------------------------------------------
# 11. ImprovedMoELayer loss dict keys
# ---------------------------------------------------------------------------


def test_improved_moe_layer_loss_keys():
    torch.manual_seed(0)
    cfg = MoEImprovementsConfig(n_experts=N_EXPERTS, capacity_factor=CAPACITY_FACTOR)
    layer = ImprovedMoELayer(D_MODEL, D_FF, cfg)
    hidden = torch.randn(B, T, D_MODEL)
    _, aux_dict = layer(hidden)
    assert "z_loss" in aux_dict, "aux_dict must contain 'z_loss'"
    assert "aux_loss" in aux_dict, "aux_dict must contain 'aux_loss'"
    assert "overflow_fraction" in aux_dict, "aux_dict must contain 'overflow_fraction'"
    assert isinstance(aux_dict["overflow_fraction"], float)


# ---------------------------------------------------------------------------
# 12. Gradient flow through ImprovedMoELayer
# ---------------------------------------------------------------------------


def test_improved_moe_layer_gradient_flow():
    torch.manual_seed(0)
    cfg = MoEImprovementsConfig(
        n_experts=N_EXPERTS, capacity_factor=CAPACITY_FACTOR, use_aux_loss=True
    )
    layer = ImprovedMoELayer(D_MODEL, D_FF, cfg)
    hidden = torch.randn(B, T, D_MODEL, requires_grad=True)
    output, aux_dict = layer(hidden)
    loss = output.sum() + aux_dict["z_loss"] + aux_dict["aux_loss"]
    loss.backward()
    assert hidden.grad is not None, "Gradient must flow back to input"
    assert hidden.grad.abs().sum().item() > 0, "Input gradient must be non-zero"
    # Gate weight should also have gradient
    assert layer.router.gate.weight.grad is not None, "Router gate must receive gradients"
