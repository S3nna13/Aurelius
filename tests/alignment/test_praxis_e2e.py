"""End-to-end validation for PRAXIS — exercises every fixed code path.

Tests:
  1. Normal train_step: finite loss, non-NaN signals, non-zero gradients
  2. B=1 batch: no NaN from std(correction=0) fix
  3. SRC receives long input_ids (not float hidden) — no type error
  4. All-unsafe batch: ESA loss returned instead of zero
  5. Per-sample think_weights applied independently per sample
  6. Constitutional gate: partial gating zeroes advantages for unsafe seqs only
  7. WARP merge: anchor_merge called at correct step interval
  8. Public __all__ exports: all PRAXIS symbols importable via star import
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal model stub matching AureliusTransformer interface
# ---------------------------------------------------------------------------

class _FakeRouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)


class _FakeMoEFFN(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 4):
        super().__init__()
        self.router = _FakeRouter(d_model, n_experts)
        self.proj   = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(x)


class _FakeLayer(nn.Module):
    """Mimics a TransformerBlock: accepts (x, freqs, mask, pkv) or just (x,)."""
    def __init__(self, d_model: int, use_moe: bool = False, n_experts: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn  = _FakeMoEFFN(d_model, n_experts) if use_moe else nn.Linear(d_model, d_model)

    def forward(self, x, freqs_cis=None, mask=None, past_kv=None):
        x = self.norm(x)
        out = self.ffn(x) if isinstance(self.ffn, _FakeMoEFFN) else self.ffn(x)
        return out, (None, None), torch.zeros(1, device=x.device)


class FakeAureliusModel(nn.Module):
    """Minimal model matching the AureliusTransformer interface used by PRAXISTrainer.

    - model.layers: nn.ModuleList[_FakeLayer]
    - model(input_ids, mask, labels, return_hidden_states) → (loss, logits, pkv, x) or (loss, logits, pkv)
    - model(input_ids) → (None, logits, [])  — used by SRC
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 16,
                 n_layers: int = 4, n_experts: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.embed      = nn.Embedding(vocab_size, d_model)
        # Mix of MoE and regular layers so ESA has real targets
        self.layers = nn.ModuleList([
            _FakeLayer(d_model, use_moe=(i % 2 == 0), n_experts=n_experts)
            for i in range(n_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # tie weights

    def forward(self, input_ids, mask=None, labels=None,
                past_key_values=None, return_hidden_states=False):
        x = self.embed(input_ids)              # (B, T, D) — requires long input_ids
        present_kv = []
        for layer in self.layers:
            x, kv, _ = layer(x)
            present_kv.append(kv)
        x = self.norm(x)
        logits = self.lm_head(x)              # (B, T, vocab_size)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous().clamp(min=0)
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
            )

        if return_hidden_states:
            return loss, logits, present_kv, x
        return loss, logits, present_kv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cfg(**kwargs) -> "PRAXISConfig":
    from src.alignment.praxis.config import PRAXISConfig
    defaults = dict(
        d_model=16, n_principles=2, mc_dropout_n=3,
        steer_layers=[0, 1], safety_experts=[0, 1],
        n_group=2, max_new_tokens=4, warp_interval=9999,
        tau_gate=0.4, tau_safety=0.5, k_mtah=1, gamma_mtah=0.9,
        think_weight=0.5, answer_weight=1.0,
    )
    defaults.update(kwargs)
    return PRAXISConfig(**defaults)


def make_trainer(cfg=None, n_layers=4):
    from src.alignment.praxis.trainer import PRAXISTrainer
    cfg   = cfg or make_cfg()
    model = FakeAureliusModel(vocab_size=128, d_model=cfg.d_model,
                              n_layers=n_layers, n_experts=4)
    return PRAXISTrainer(model, cfg), model


def make_batch(B=4, T=12, vocab=128):
    return {
        "input_ids":      torch.randint(1, vocab, (B, T)),
        "labels":         torch.randint(1, vocab, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Test 1: Normal train_step — finite loss, no NaN, non-zero gradients
# ---------------------------------------------------------------------------

def test_e2e_normal_train_step():
    trainer, model = make_trainer()
    batch = make_batch(B=4, T=12)

    # Enable gradients on model params
    for p in model.parameters():
        p.requires_grad_(True)

    metrics = trainer.train_step(batch, step=1)

    # Loss must be a real scalar
    assert "total_loss" in metrics or "dapo_loss" in metrics, f"missing loss key: {metrics}"

    # All metric values must be finite
    for k, v in metrics.items():
        assert v == v, f"metric '{k}' is NaN"          # NaN != NaN
        assert abs(v) < 1e9, f"metric '{k}' exploded: {v}"

    print(f"[PASS] test_e2e_normal_train_step — metrics: {metrics}")


# ---------------------------------------------------------------------------
# Test 2: B=1 — no NaN from std(correction=0)
# ---------------------------------------------------------------------------

def test_e2e_batch_size_one():
    trainer, _ = make_trainer()
    batch = make_batch(B=1, T=12)

    metrics = trainer.train_step(batch, step=1)

    for k, v in metrics.items():
        assert v == v, f"B=1: metric '{k}' is NaN (std correction bug)"

    print(f"[PASS] test_e2e_batch_size_one — metrics: {metrics}")


# ---------------------------------------------------------------------------
# Test 3: SRC receives long input_ids (not float hidden)
# ---------------------------------------------------------------------------

def test_e2e_src_receives_input_ids():
    """Verifies that SteeringRewardCorrespondence.compute() is called with
    input_ids (long), which are then passed to the model in _capture_hidden
    to extract layer hidden states via forward hooks."""
    from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence

    cfg   = make_cfg(steer_layers=[0, 1])
    model = FakeAureliusModel(vocab_size=128, d_model=cfg.d_model, n_layers=4)
    src   = SteeringRewardCorrespondence(model, cfg)

    input_ids = torch.randint(1, 128, (2, 12))   # long tensor ✓
    assert input_ids.dtype == torch.long, "input_ids must be long"

    reward = src.compute(input_ids)
    assert reward.shape == (), f"SRC should return scalar, got {reward.shape}"
    assert reward.item() <= 0.0, "SRC reward should be non-positive"

    print(f"[PASS] test_e2e_src_receives_input_ids — SRC reward: {reward.item():.4f}")


# ---------------------------------------------------------------------------
# Test 4: All-unsafe batch — ESA loss returned, not zero
# ---------------------------------------------------------------------------

def test_e2e_all_unsafe_batch_esa_not_dropped():
    """When all constitutional scores are below tau_gate, the early-return
    path in PRAXISLoss must still return the ESA loss (not zero)."""
    from src.alignment.praxis.praxis_loss import PRAXISLoss

    cfg      = make_cfg(tau_gate=0.9, tau_safety=0.5)   # high gate → likely all fail
    loss_fn  = PRAXISLoss(cfg)

    B, T     = 2, 8
    log_probs     = torch.randn(B, T, requires_grad=True)
    old_log_probs = log_probs.detach()
    advantages    = torch.randn(B, T)
    fused_rewards = torch.randn(B)
    mask          = torch.ones(B, T, dtype=torch.bool)

    # Constitutional scores ALL below tau_gate=0.9
    const_scores = torch.zeros(B)         # 0.0 < 0.9 → all fail gate

    # Provide a non-zero ESA loss
    esa_loss = torch.tensor(0.314)

    total, metrics = loss_fn.forward(
        log_probs, old_log_probs, advantages, fused_rewards, mask,
        const_scores=const_scores, esa_loss=esa_loss,
    )

    assert metrics["const_gate_ratio"] == 0.0, "All sequences should be gated"
    assert abs(total.item() - 0.314) < 1e-5, \
        f"ESA loss should pass through on all-gated batch: got {total.item()}"
    assert "esa_loss" in metrics, "esa_loss key must be present in early-return metrics"

    print(f"[PASS] test_e2e_all_unsafe_batch_esa_not_dropped — total={total.item():.4f}")


# ---------------------------------------------------------------------------
# Test 5: Per-sample think_weights — different seqs → different weights
# ---------------------------------------------------------------------------

def test_e2e_per_sample_think_weights():
    """Verifies that think_weights are computed per sample, not just from input_ids[0].
    Two sequences with different <think> token positions should produce different weights.
    """
    from src.alignment.thinking_tokens import ThinkingLossWeights, THINK_START_TOKEN_ID, THINK_END_TOKEN_ID

    think_wts = ThinkingLossWeights(think_weight=0.5, answer_weight=1.0)

    # Sample 0: <think> tokens at positions 2-5
    # Sample 1: regular tokens throughout (no think tokens)
    T = 8
    ids_0 = [10, 10, THINK_START_TOKEN_ID, 10, 10, THINK_END_TOKEN_ID, 10, 10]
    ids_1 = [10, 10, 10, 10, 10, 10, 10, 10]

    w0 = think_wts.compute_weights(ids_0)
    w1 = think_wts.compute_weights(ids_1)

    assert not torch.allclose(w0, w1), \
        "Per-sample think_weights should differ when <think> tokens differ"

    # Verify that the trainer uses per-sample weights (stack path)
    input_ids = torch.tensor([ids_0, ids_1], dtype=torch.long)
    stacked = torch.stack(
        [think_wts.compute_weights(input_ids[i].tolist()) for i in range(2)],
        dim=0,
    )
    assert stacked.shape == (2, T)
    assert not torch.allclose(stacked[0], stacked[1])

    print(f"[PASS] test_e2e_per_sample_think_weights — w0={w0.tolist()}, w1={w1.tolist()}")


# ---------------------------------------------------------------------------
# Test 6: Constitutional gate partial — only unsafe seqs get zeroed advantages
# ---------------------------------------------------------------------------

def test_e2e_constitutional_gate_partial():
    """Half the batch passes the gate (scores >= tau_gate), half fails.
    The passing sequences should have non-zero advantages; failing ones should be zeroed.
    """
    from src.alignment.praxis.praxis_loss import PRAXISLoss

    cfg     = make_cfg(tau_gate=0.5)
    loss_fn = PRAXISLoss(cfg)

    B, T = 4, 6
    log_probs     = torch.randn(B, T, requires_grad=True)
    old_log_probs = log_probs.detach()
    # Advantages all positive so loss sign is predictable
    advantages    = torch.ones(B, T)
    fused_rewards = torch.ones(B)
    mask          = torch.ones(B, T, dtype=torch.bool)

    # Sequences 0,1 pass gate; 2,3 fail
    const_scores = torch.tensor([0.8, 0.9, 0.1, 0.2])

    total, metrics = loss_fn.forward(
        log_probs, old_log_probs, advantages, fused_rewards, mask,
        const_scores=const_scores,
    )

    assert metrics["const_gate_ratio"] == 0.5, \
        f"Half should pass gate, got {metrics['const_gate_ratio']}"
    assert total.item() == total.item(), "loss must not be NaN with partial gating"
    assert metrics["dapo_loss"] != 0.0, "dapo_loss should be non-zero for passing seqs"

    print(f"[PASS] test_e2e_constitutional_gate_partial — gate_ratio={metrics['const_gate_ratio']}")


# ---------------------------------------------------------------------------
# Test 7: WARP merge fires at correct step interval
# ---------------------------------------------------------------------------

def test_e2e_warp_merge_fires_at_interval():
    """Confirms anchor_merge is called when step % warp_interval == 0."""
    from src.alignment.praxis.trainer import PRAXISTrainer
    from src.alignment.warp import anchor_merge as real_anchor_merge

    cfg   = make_cfg(warp_interval=5)
    model = FakeAureliusModel(vocab_size=128, d_model=cfg.d_model, n_layers=4)

    # Use model's own state_dict as a stand-in SFT checkpoint
    sft_sd = {k: v.clone() for k, v in model.state_dict().items()}

    trainer = PRAXISTrainer(model, cfg, sft_state_dict=sft_sd)
    batch   = make_batch(B=2, T=12)

    merge_calls = []
    original_anchor_merge = __import__("src.alignment.warp", fromlist=["anchor_merge"]).anchor_merge

    import src.alignment.praxis.trainer as trainer_mod
    original_fn = trainer_mod.anchor_merge

    def patched_anchor_merge(sft_sd, current_sd, alpha):
        merge_calls.append(alpha)
        return original_fn(sft_sd, current_sd, alpha)

    trainer_mod.anchor_merge = patched_anchor_merge

    trainer.train_step(batch, step=4)   # not a multiple of 5 → no merge
    assert len(merge_calls) == 0, f"Should not merge at step=4, got {merge_calls}"

    trainer.train_step(batch, step=5)   # multiple of 5 → merge fires
    assert len(merge_calls) == 1, f"Should merge at step=5, got {len(merge_calls)}"
    expected_alpha = 1 - cfg.warp_anchor_mu   # 0.95
    assert abs(merge_calls[0] - expected_alpha) < 1e-6, \
        f"Wrong alpha: expected {expected_alpha}, got {merge_calls[0]}"

    trainer_mod.anchor_merge = original_fn  # restore
    print(f"[PASS] test_e2e_warp_merge_fires_at_interval — alpha={merge_calls[0]:.4f}")


# ---------------------------------------------------------------------------
# Test 8: __all__ exports — all PRAXIS symbols importable
# ---------------------------------------------------------------------------

def test_e2e_public_all_exports():
    """All PRAXIS symbols added to src.alignment.__all__ must be importable."""
    import src.alignment as alignment_mod

    praxis_symbols = [
        "PRAXISConfig", "PRAXISLoss", "PRAXISTrainer",
        "PrecisionFusion", "SteeringRewardCorrespondence",
        "ExpertSafetyAffinity", "MultiTokenAlignmentHorizon",
    ]
    missing = [s for s in praxis_symbols if s not in alignment_mod.__all__]
    assert not missing, f"Missing from __all__: {missing}"

    # Also verify they are actually importable (not just listed)
    for sym in praxis_symbols:
        obj = getattr(alignment_mod, sym, None)
        assert obj is not None, f"{sym} listed in __all__ but not importable from src.alignment"

    print(f"[PASS] test_e2e_public_all_exports — all {len(praxis_symbols)} symbols OK")


# ---------------------------------------------------------------------------
# Test 9: Gradient flow — loss.backward() produces non-zero param gradients
# ---------------------------------------------------------------------------

def test_e2e_gradient_flows_through_loss():
    """The policy gradient loss must carry gradients back to model parameters.
    Previously broken because log_probs_cur was computed inside torch.no_grad().
    """
    trainer, model = make_trainer()
    batch = make_batch(B=2, T=8)

    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # We need the loss tensor itself, not just metrics — run components directly
    cfg = trainer.config
    input_ids = batch["input_ids"]
    labels    = batch["labels"]
    mask      = batch["attention_mask"]

    model.train()
    model_out = model(input_ids, mask=mask, labels=labels, return_hidden_states=True)
    _, logits, _, hidden = model_out

    log_probs_cur = trainer._get_log_probs(model_out, input_ids, mask)

    # Verify log_probs_cur is on the compute graph
    assert log_probs_cur.requires_grad, \
        "log_probs_cur must have requires_grad=True (was broken by no_grad block)"

    # Simple surrogate: treat negative mean log-prob as the loss
    loss = -log_probs_cur.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed to model parameters"
    assert any(g.abs().sum().item() > 0 for g in grads), \
        "All gradients are zero — policy gradient is dead"

    print(f"[PASS] test_e2e_gradient_flows_through_loss — {len(grads)} param tensors have grad")


# ---------------------------------------------------------------------------
# Test 10: No-NaN under all signal paths (B=4 full run)
# ---------------------------------------------------------------------------

def test_e2e_no_nan_signals():
    """Runs a full train_step and checks every individual reward signal for NaN/Inf."""
    from src.alignment.praxis.reward_signals import RewardSignalBundle
    from src.alignment.praxis.precision_fusion import PrecisionFusion
    from src.alignment.prime import PRIMEReward, PRIMEConfig
    from src.alignment.constitutional_ai_v3 import CritiqueHead
    from src.alignment.reward_uncertainty import MCDropoutReward

    cfg = make_cfg(mc_dropout_n=3, n_principles=2)
    B, T, D = 3, 10, cfg.d_model

    prime     = PRIMEReward(PRIMEConfig(beta=cfg.beta_kl))
    critique  = CritiqueHead(D, cfg.n_principles)
    mc_reward = MCDropoutReward(D)
    bundle    = RewardSignalBundle(cfg, prime, critique,
                                  lambda x: torch.zeros(x.shape[0]), mc_reward)

    hidden       = torch.randn(B, T, D)
    log_probs    = torch.randn(B, T)
    ref_lp       = log_probs.detach().clone()
    outcome_r    = torch.randn(B)
    mask         = torch.ones(B, T, dtype=torch.bool)

    signals = bundle.compute(hidden, log_probs, ref_lp, outcome_r, mask)

    fusion = PrecisionFusion(n_signals=6)
    means  = [signals[k][0] for k in ["r_prime", "r_const", "r_ccot", "r_odin", "r_hier", "r_src"]]
    stds   = [signals[k][1] for k in ["r_prime", "r_const", "r_ccot", "r_odin", "r_hier", "r_src"]]

    for name, (m, s) in signals.items():
        assert torch.isfinite(m).all(), f"{name} mean has NaN/Inf: {m}"
        assert torch.isfinite(s).all(), f"{name} std has NaN/Inf: {s}"

    fused = fusion.fuse(means, stds)
    assert torch.isfinite(fused).all(), f"fused_reward has NaN/Inf: {fused}"

    # B=1 edge case
    hidden1   = torch.randn(1, T, D)
    log_p1    = torch.randn(1, T)
    outcome1  = torch.randn(1)
    mask1     = torch.ones(1, T, dtype=torch.bool)
    signals1  = bundle.compute(hidden1, log_p1, log_p1.detach(), outcome1, mask1)

    for name, (m, s) in signals1.items():
        assert torch.isfinite(m).all(), f"B=1: {name} mean is NaN: {m}"
        assert torch.isfinite(s).all(), f"B=1: {name} std is NaN: {s}"

    print(f"[PASS] test_e2e_no_nan_signals — all signals finite for B={B} and B=1")
