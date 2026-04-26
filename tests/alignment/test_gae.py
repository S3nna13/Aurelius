import pytest
import torch

from src.alignment.gae import compute_gae, compute_returns, normalize_advantages


def test_gae_output_shapes():
    T = 10
    rewards = torch.ones(T)
    values = torch.zeros(T)
    dones = torch.zeros(T)
    adv, ret = compute_gae(rewards, values, dones)
    assert adv.shape == (T,)
    assert ret.shape == (T,)


def test_gae_no_done_returns_sum_of_rewards():
    """With no dones and values=0, returns should be discounted sum of rewards."""
    rewards = torch.ones(5)
    values = torch.zeros(5)
    dones = torch.zeros(5)
    gamma = 0.9
    adv, ret = compute_gae(rewards, values, dones, gamma=gamma, lam=1.0, next_value=0.0)
    # returns[0] = 1 + 0.9 + 0.81 + 0.729 + 0.6561 ≈ 4.0951
    expected = sum(gamma**i for i in range(5))
    assert ret[0].item() == pytest.approx(expected, rel=1e-4)


def test_gae_done_resets():
    """Episode termination (done=1) should reset the bootstrap."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.zeros(3)
    dones = torch.tensor([0.0, 1.0, 0.0])  # done after step 1
    adv, ret = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)
    # ret[2] = 1.0 (last, no next)
    # ret[1] = 1.0 (done=1 resets, so ret[1] = reward[1] + 0)
    # ret[0] = 1.0 + 0.9 * ret[1] = 1 + 0.9 = 1.9
    assert ret[2].item() == pytest.approx(1.0)
    assert ret[1].item() == pytest.approx(1.0)
    assert ret[0].item() == pytest.approx(1.9, rel=1e-4)


def test_gae_advantages_equal_returns_minus_values():
    rewards = torch.randn(8)
    values = torch.randn(8)
    dones = torch.zeros(8)
    adv, ret = compute_gae(rewards, values, dones)
    assert torch.allclose(ret, adv + values, atol=1e-5)


def test_compute_returns_shape():
    rewards = torch.ones(7)
    dones = torch.zeros(7)
    ret = compute_returns(rewards, dones)
    assert ret.shape == (7,)


def test_compute_returns_discounted():
    rewards = torch.ones(3)
    dones = torch.zeros(3)
    ret = compute_returns(rewards, dones, gamma=0.5, next_value=0.0)
    # ret[2] = 1
    # ret[1] = 1 + 0.5 = 1.5
    # ret[0] = 1 + 0.5 * 1.5 = 1.75
    assert ret[2].item() == pytest.approx(1.0)
    assert ret[1].item() == pytest.approx(1.5)
    assert ret[0].item() == pytest.approx(1.75)


def test_normalize_advantages_mean_zero():
    adv = torch.randn(100)
    normalized = normalize_advantages(adv)
    assert abs(normalized.mean().item()) < 1e-5


def test_normalize_advantages_std_one():
    adv = torch.randn(100)
    normalized = normalize_advantages(adv)
    assert abs(normalized.std().item() - 1.0) < 0.01


def test_gae_lam_zero_equals_td():
    """Lambda=0 should give single-step TD advantage."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    values = torch.tensor([0.5, 1.0, 1.5])
    dones = torch.zeros(3)
    adv, _ = compute_gae(rewards, values, dones, gamma=0.9, lam=0.0)
    # delta_t = r_t + gamma * V(t+1) - V(t)
    # adv[2] = 3.0 + 0 - 1.5 = 1.5
    # adv[1] = 2.0 + 0.9*1.5 - 1.0 = 2.35
    # adv[0] = 1.0 + 0.9*1.0 - 0.5 = 1.4
    assert adv[2].item() == pytest.approx(1.5)
    assert adv[1].item() == pytest.approx(2.35)
    assert adv[0].item() == pytest.approx(1.4)
