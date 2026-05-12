import torch

from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.praxis_loss import PRAXISLoss


def make_loss(B=2, T=6, cfg=None):
    cfg = cfg or PRAXISConfig(d_model=8)
    return PRAXISLoss(cfg), B, T

def test_loss_returns_scalar():
    loss_fn, B, T = make_loss()
    log_probs     = torch.randn(B, T)
    old_log_probs = torch.randn(B, T)
    advantages    = torch.randn(B, T)
    fused_rewards = torch.randn(B)
    mask          = torch.ones(B, T, dtype=torch.bool)
    entropy       = torch.rand(B, T) * 2.0

    result, metrics = loss_fn.forward(
        log_probs, old_log_probs, advantages, fused_rewards, mask, entropy=entropy
    )
    assert result.shape == (), f"expected scalar loss: {result.shape}"
    assert "dapo_loss" in metrics
    assert "kl_penalty" in metrics

def test_const_gate_blocks_grad():
    cfg = PRAXISConfig(d_model=8, tau_gate=0.4)
    loss_fn, B, T = make_loss(cfg=cfg)
    log_probs     = torch.randn(B, T, requires_grad=True)
    old_log_probs = torch.randn(B, T)
    advantages    = torch.randn(B, T)
    fused_rewards = torch.randn(B)
    mask          = torch.ones(B, T, dtype=torch.bool)

    # Constitutional scores all below gate threshold → loss should be 0 or very small
    const_scores  = torch.ones(B) * 0.1   # all below tau_gate=0.4
    result, _     = loss_fn.forward(
        log_probs, old_log_probs, advantages, fused_rewards, mask,
        const_scores=const_scores
    )
    assert result.item() == 0.0, f"below-gate sequences should produce zero loss: {result.item()}"