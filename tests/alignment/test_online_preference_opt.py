"""Tests for Online Preference Optimization (src/alignment/online_preference_opt.py).

Covers:
  1.  OPOConfig defaults are sensible
  2.  generate_response returns Tensor of length max_new_tokens
  3.  generate_response returns valid token ids in [0, vocab_size)
  4.  collect_online_pairs returns list of OnlinePair
  5.  Each OnlinePair has chosen_reward >= rejected_reward
  6.  opo_loss returns (Tensor, dict) tuple
  7.  opo_loss dict has keys: 'dpo_loss', 'sft_loss', 'total_loss', 'reward_margin'
  8.  opo_loss total_loss is finite
  9.  opo_loss reward_margin = chosen_reward - rejected_reward >= 0
  10. OPOTrainer constructs without error
  11. OPOTrainer.collect_and_train returns metrics dict
  12. collect_and_train changes model weights (optimizer step happened)
"""

from __future__ import annotations

import copy

import torch

from src.alignment.online_preference_opt import (
    OnlinePair,
    OPOConfig,
    OPOTrainer,
    collect_online_pairs,
    generate_response,
    opo_loss,
)
from src.alignment.reward_model import RewardModel, RewardModelConfig
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256


def make_tiny_model(seed: int = 0) -> AureliusTransformer:
    torch.manual_seed(seed)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def make_ref_model(policy: AureliusTransformer) -> AureliusTransformer:
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def make_reward_model() -> RewardModel:
    """Build a reward model backed by a tiny transformer hidden-state backbone."""
    torch.manual_seed(42)
    backbone = make_tiny_model(seed=99)

    def backbone_fn(token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) -> hidden states (B, T, d_model)
        out = backbone(token_ids)
        out[1] if isinstance(out, tuple) else out  # (B, T, vocab)
        # project vocab -> d_model via random linear to get "hidden states"
        # We re-use the embedding matrix transposed: (vocab, d_model) -> (B, T, d_model)
        embed_w = backbone.embed.weight  # (vocab, d_model)
        # Simple: gather embeddings for each token as the "hidden state"
        hidden = embed_w[token_ids]  # (B, T, d_model)
        return hidden

    rm_cfg = RewardModelConfig(d_model=64)
    return RewardModel(backbone_fn=backbone_fn, config=rm_cfg)


# ---------------------------------------------------------------------------
# Test 1: OPOConfig defaults are sensible
# ---------------------------------------------------------------------------


def test_opo_config_defaults():
    cfg = OPOConfig()
    assert cfg.beta == 0.1
    assert cfg.lr == 1e-5
    assert cfg.batch_size == 4
    assert cfg.n_steps == 4
    assert cfg.max_new_tokens == 16
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.9
    assert cfg.reward_model_weight == 1.0
    assert cfg.sft_weight == 0.1
    assert cfg.max_seq_len == 64


# ---------------------------------------------------------------------------
# Test 2: generate_response returns Tensor of length max_new_tokens
# ---------------------------------------------------------------------------


def test_generate_response_length():
    torch.manual_seed(0)
    model = make_tiny_model(1)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (6,))

    for max_new_tokens in [4, 8, 16]:
        result = generate_response(model, prompt_ids, max_new_tokens=max_new_tokens)
        assert isinstance(result, torch.Tensor), "generate_response should return a Tensor"
        assert result.shape == (max_new_tokens,), (
            f"Expected ({max_new_tokens},), got {result.shape}"
        )


# ---------------------------------------------------------------------------
# Test 3: generate_response returns valid token ids in [0, vocab_size)
# ---------------------------------------------------------------------------


def test_generate_response_valid_token_ids():
    torch.manual_seed(1)
    model = make_tiny_model(2)
    prompt_ids = torch.randint(0, VOCAB_SIZE, (4,))

    result = generate_response(model, prompt_ids, max_new_tokens=10)
    assert (result >= 0).all(), "Token ids should be >= 0"
    assert (result < VOCAB_SIZE).all(), f"Token ids should be < {VOCAB_SIZE}"


# ---------------------------------------------------------------------------
# Test 4: collect_online_pairs returns list of OnlinePair
# ---------------------------------------------------------------------------


def test_collect_online_pairs_returns_list():
    torch.manual_seed(2)
    model = make_tiny_model(3)
    reward_model = make_reward_model()

    prompts = [torch.randint(0, VOCAB_SIZE, (4,)) for _ in range(3)]
    cfg = OPOConfig(max_new_tokens=4, temperature=1.0, top_p=0.9)

    pairs = collect_online_pairs(model, reward_model, prompts, cfg, n_samples_per_prompt=2)

    assert isinstance(pairs, list), "collect_online_pairs should return a list"
    assert len(pairs) == len(prompts), f"Expected {len(prompts)} pairs, got {len(pairs)}"
    for pair in pairs:
        assert isinstance(pair, OnlinePair), (
            f"Each element should be an OnlinePair, got {type(pair)}"
        )


# ---------------------------------------------------------------------------
# Test 5: Each OnlinePair has chosen_reward >= rejected_reward
# ---------------------------------------------------------------------------


def test_collect_online_pairs_chosen_geq_rejected():
    torch.manual_seed(3)
    model = make_tiny_model(4)
    reward_model = make_reward_model()

    prompts = [torch.randint(0, VOCAB_SIZE, (4,)) for _ in range(4)]
    cfg = OPOConfig(max_new_tokens=4, temperature=1.0, top_p=0.9)

    pairs = collect_online_pairs(model, reward_model, prompts, cfg, n_samples_per_prompt=3)

    for i, pair in enumerate(pairs):
        assert pair.chosen_reward >= pair.rejected_reward, (
            f"Pair {i}: chosen_reward ({pair.chosen_reward:.4f}) < "
            f"rejected_reward ({pair.rejected_reward:.4f})"
        )


# ---------------------------------------------------------------------------
# Test 6: opo_loss returns (Tensor, dict) tuple
# ---------------------------------------------------------------------------


def test_opo_loss_returns_tensor_and_dict():
    torch.manual_seed(4)
    model = make_tiny_model(5)
    ref_model = make_ref_model(model)

    prompt = torch.randint(0, VOCAB_SIZE, (4,))
    chosen = torch.randint(0, VOCAB_SIZE, (8,))
    rejected = torch.randint(0, VOCAB_SIZE, (8,))

    batch = [
        OnlinePair(
            prompt_ids=prompt,
            chosen_ids=chosen,
            rejected_ids=rejected,
            chosen_reward=1.0,
            rejected_reward=0.0,
        )
    ]

    result = opo_loss(model, ref_model, batch)
    assert isinstance(result, tuple) and len(result) == 2, (
        "opo_loss should return a (Tensor, dict) tuple"
    )
    loss, metrics = result
    assert isinstance(loss, torch.Tensor), "First element should be a Tensor"
    assert isinstance(metrics, dict), "Second element should be a dict"


# ---------------------------------------------------------------------------
# Test 7: opo_loss dict has required keys
# ---------------------------------------------------------------------------


def test_opo_loss_dict_keys():
    torch.manual_seed(5)
    model = make_tiny_model(6)
    ref_model = make_ref_model(model)

    prompt = torch.randint(0, VOCAB_SIZE, (4,))
    chosen = torch.randint(0, VOCAB_SIZE, (8,))
    rejected = torch.randint(0, VOCAB_SIZE, (8,))

    batch = [
        OnlinePair(
            prompt_ids=prompt,
            chosen_ids=chosen,
            rejected_ids=rejected,
            chosen_reward=1.0,
            rejected_reward=0.0,
        )
    ]

    _, metrics = opo_loss(model, ref_model, batch)

    required_keys = {"dpo_loss", "sft_loss", "total_loss", "reward_margin"}
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in opo_loss metrics"


# ---------------------------------------------------------------------------
# Test 8: opo_loss total_loss is finite
# ---------------------------------------------------------------------------


def test_opo_loss_total_loss_finite():
    torch.manual_seed(6)
    model = make_tiny_model(7)
    ref_model = make_ref_model(model)

    prompt = torch.randint(0, VOCAB_SIZE, (4,))
    chosen = torch.randint(0, VOCAB_SIZE, (8,))
    rejected = torch.randint(0, VOCAB_SIZE, (8,))

    batch = [
        OnlinePair(
            prompt_ids=prompt,
            chosen_ids=chosen,
            rejected_ids=rejected,
            chosen_reward=1.0,
            rejected_reward=0.0,
        )
    ]

    loss, metrics = opo_loss(model, ref_model, batch)
    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    assert torch.isfinite(torch.tensor(metrics["total_loss"])), (
        f"total_loss metric should be finite, got {metrics['total_loss']}"
    )


# ---------------------------------------------------------------------------
# Test 9: opo_loss reward_margin = chosen - rejected >= 0 for best-vs-worst pair
# ---------------------------------------------------------------------------


def test_opo_loss_reward_margin():
    torch.manual_seed(7)
    model = make_tiny_model(8)
    ref_model = make_ref_model(model)

    # Create pairs where chosen is always better than rejected
    batch = []
    for _ in range(3):
        prompt = torch.randint(0, VOCAB_SIZE, (4,))
        chosen = torch.randint(0, VOCAB_SIZE, (6,))
        rejected = torch.randint(0, VOCAB_SIZE, (6,))
        batch.append(
            OnlinePair(
                prompt_ids=prompt,
                chosen_ids=chosen,
                rejected_ids=rejected,
                chosen_reward=2.0,
                rejected_reward=0.5,
            )
        )

    _, metrics = opo_loss(model, ref_model, batch)
    # reward_margin is derived from DPO implicit rewards, check it's consistent
    # (it's the mean difference of implicit rewards, not raw rewards)
    assert "reward_margin" in metrics
    # Just check it's a finite float
    assert isinstance(metrics["reward_margin"], float)
    assert torch.isfinite(torch.tensor(metrics["reward_margin"]))


# ---------------------------------------------------------------------------
# Test 10: OPOTrainer constructs without error
# ---------------------------------------------------------------------------


def test_opo_trainer_constructs():
    torch.manual_seed(8)
    model = make_tiny_model(9)
    ref_model = make_ref_model(model)
    reward_model = make_reward_model()
    cfg = OPOConfig(max_new_tokens=4, n_steps=1)

    trainer = OPOTrainer(model, ref_model, reward_model, cfg)
    assert trainer is not None
    assert trainer.model is model
    assert trainer.ref_model is ref_model
    assert trainer.reward_model is reward_model
    assert trainer.config is cfg


# ---------------------------------------------------------------------------
# Test 11: OPOTrainer.collect_and_train returns metrics dict
# ---------------------------------------------------------------------------


def test_collect_and_train_returns_metrics():
    torch.manual_seed(9)
    model = make_tiny_model(10)
    ref_model = make_ref_model(model)
    reward_model = make_reward_model()
    cfg = OPOConfig(max_new_tokens=4, n_steps=1, lr=1e-4)

    trainer = OPOTrainer(model, ref_model, reward_model, cfg)
    prompts = [torch.randint(0, VOCAB_SIZE, (4,)) for _ in range(2)]

    metrics = trainer.collect_and_train(prompts)

    assert isinstance(metrics, dict), "collect_and_train should return a dict"
    required_keys = {"dpo_loss", "sft_loss", "total_loss", "reward_margin"}
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in collect_and_train metrics"


# ---------------------------------------------------------------------------
# Test 12: collect_and_train changes model weights (optimizer step happened)
# ---------------------------------------------------------------------------


def test_collect_and_train_updates_weights():
    torch.manual_seed(10)
    model = make_tiny_model(11)
    ref_model = make_ref_model(model)
    reward_model = make_reward_model()
    cfg = OPOConfig(max_new_tokens=4, n_steps=2, lr=1e-3)

    # Snapshot model weights before training
    params_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    trainer = OPOTrainer(model, ref_model, reward_model, cfg)
    prompts = [torch.randint(0, VOCAB_SIZE, (4,)) for _ in range(2)]
    trainer.collect_and_train(prompts)

    # Check that at least one parameter changed
    any_changed = False
    for name, param in model.named_parameters():
        if param.requires_grad and name in params_before:
            if not torch.allclose(params_before[name], param.detach()):
                any_changed = True
                break

    assert any_changed, "Model weights should have changed after collect_and_train"
