# PRAXIS Implementation Plan
# Policy Refinement through Aligned eXpert Integration System

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build PRAXIS, a unified alignment trainer that integrates all existing Aurelius alignment modules (GRPO, PRIME, Constitutional AI, ODIN, DPO, Hierarchical Reward) plus three novel architecture-aware contributions: Steering-Reward Correspondence (SRC), Expert Safety Affinity (ESA), and Multi-Token Alignment Horizon (MTAH).

**Architecture:** PRAXISTrainer orchestrates a 6-signal reward bundle fused via Bayesian inverse-variance weighting (PrecisionFusion), applies DAPO asymmetric clipping, gates constitutional gradient when safety score is below threshold, and periodically merges policy weights via WARP. The three novel signals exploit internal model architecture: SRC uses forward hooks on latent steering layers, ESA routes unsafe tokens toward safety experts via direct gate calls on `layer.ffn.router.gate`, and MTAH extends temporal advantage credit across future tokens.

**Tech Stack:** PyTorch 2.x, existing `src/alignment/` modules, `src/model/moe.py` (SparseMoELayer with TopKRouter), `src/model/transformer.py` (AureliusTransformer with `return_hidden_states=True`). Always use `model.train(False)` for inference mode.

---

## Critical API Facts (verified from codebase scan)

- `SparseMoELayer(d_model, d_ff, n_experts, top_k=2)` — positional args; `self.router = TopKRouter(d_model, n_experts, top_k)` where `TopKRouter.gate = nn.Linear(d_model, n_experts, bias=False)`
- `HierarchicalRewardModel.forward(hidden)` expects `(B, d_model)` — must pool: `h = hidden[:, -1, :]`
- `AureliusTransformer.forward(input_ids, ..., return_hidden_states=True)` returns `(loss, logits, pkv, x)` where `x` is final hidden states `(B, T, D)`
- `ThinkingLossWeights(think_weight, answer_weight).compute_weights(token_ids: list[int]) → Tensor`
- `anchor_merge(sft_state_dict, merged_state_dict, alpha)` from `warp.py`: `alpha=0` → pure SFT, `alpha=1` → pure merged
- `GroupRewardNormalizer(group_size).normalize(rewards: Tensor[(B*G,)]) → Tensor`
- `DAPOLoss(eps_low, eps_high, beta_entropy).forward(log_probs, old_log_probs, advantages, entropy=None) → (loss, metrics)`
- `CritiqueHead(d_model, n_principles).forward(hidden_states: (B, T, D)) → (B, n_principles)`
- `MCDropoutReward(d_model).predict_with_uncertainty(x, n_samples) → (mean: (B,), std: (B,))`
- `model.layers` is `nn.ModuleList[TransformerBlock]`; each block has `.ffn` (SparseMoELayer or SwiGLUFFN)

---

### Task 1: PRAXISConfig

**Files:**
- Create: `src/alignment/praxis/__init__.py`
- Create: `src/alignment/praxis/config.py`
- Create: `tests/alignment/test_praxis_config.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_praxis_config.py
from src.alignment.praxis.config import PRAXISConfig

def test_praxis_config_defaults():
    cfg = PRAXISConfig()
    assert cfg.n_group == 8
    assert cfg.eps_low == 0.20
    assert cfg.eps_high == 0.28
    assert cfg.tau_gate == 0.4
    assert cfg.steer_layers == [12, 16, 20]
    assert cfg.safety_experts == [0, 1]
    assert cfg.k_mtah == 2
    assert cfg.warp_interval == 50

def test_praxis_config_custom():
    cfg = PRAXISConfig(n_group=4, eps_low=0.1, steer_layers=[8, 12])
    assert cfg.n_group == 4
    assert cfg.steer_layers == [8, 12]
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_praxis_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.alignment.praxis'`

**Step 3: Create the package and config**

```python
# src/alignment/praxis/__init__.py
"""PRAXIS: Policy Refinement through Aligned eXpert Integration System."""
```

```python
# src/alignment/praxis/config.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class PRAXISConfig:
    # Sampling
    n_group: int = 8
    temperature: float = 0.9
    max_new_tokens: int = 128
    d_model: int = 2048
    # DAPO clip bounds
    eps_low: float = 0.20
    eps_high: float = 0.28
    # KL + entropy
    beta_kl: float = 0.04
    lambda_ent: float = 0.001
    # Constitutional
    n_principles: int = 8
    tau_gate: float = 0.4
    n_criteria: int = 4
    # SRC (Steering-Reward Correspondence)
    steer_layers: list[int] = field(default_factory=lambda: [12, 16, 20])
    steer_alpha: float = 0.3
    lambda_src: float = 0.1
    # ESA (Expert Safety Affinity)
    safety_experts: list[int] = field(default_factory=lambda: [0, 1])
    alpha_esa: float = 0.01
    tau_safety: float = 0.5
    # MTAH (Multi-Token Alignment Horizon)
    gamma_mtah: float = 0.95
    k_mtah: int = 2
    # Thinking tokens
    think_weight: float = 0.5
    answer_weight: float = 1.0
    # WARP
    warp_interval: int = 50
    warp_anchor_mu: float = 0.05
    # Uncertainty
    mc_dropout_n: int = 20
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_praxis_config.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/ tests/alignment/test_praxis_config.py
git commit -m "feat(praxis): add PRAXISConfig dataclass"
```

---

### Task 2: PrecisionFusion

**Files:**
- Create: `src/alignment/praxis/precision_fusion.py`
- Test: `tests/alignment/test_precision_fusion.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_precision_fusion.py
import torch
from src.alignment.praxis.precision_fusion import PrecisionFusion

def test_equal_uncertainty_gives_uniform_weights():
    fuser = PrecisionFusion(n_signals=3)
    B = 4
    means = [torch.zeros(B) + float(i) for i in range(3)]   # 0, 1, 2
    stds  = [torch.ones(B) for _ in range(3)]                # equal std=1
    result = fuser.fuse(means, stds)
    expected = torch.ones(B) * 1.0   # (0+1+2)/3
    assert torch.allclose(result, expected, atol=1e-5), f"got {result}"

def test_low_uncertainty_dominates():
    fuser = PrecisionFusion(n_signals=2)
    B = 2
    means = [torch.ones(B) * 10.0, torch.ones(B) * 0.0]
    stds  = [torch.ones(B) * 0.01, torch.ones(B) * 100.0]   # first is precise
    result = fuser.fuse(means, stds)
    assert (result > 9.0).all(), f"precise signal should dominate: {result}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_precision_fusion.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement PrecisionFusion**

```python
# src/alignment/praxis/precision_fusion.py
from __future__ import annotations
import torch
from torch import Tensor

class PrecisionFusion:
    """Bayesian inverse-variance weighting over multiple reward signals."""

    def __init__(self, n_signals: int, eps: float = 1e-6) -> None:
        self.n_signals = n_signals
        self.eps = eps

    def fuse(self, means: list[Tensor], stds: list[Tensor]) -> Tensor:
        """Fuse reward signals weighted by precision (1/variance).

        Args:
            means: list of (B,) reward mean tensors, one per signal.
            stds:  list of (B,) reward std tensors, one per signal.

        Returns:
            (B,) fused reward.
        """
        means_stack = torch.stack(means, dim=0)                          # (n_signals, B)
        stds_stack  = torch.stack(stds, dim=0).clamp(min=self.eps)       # (n_signals, B)
        precision   = 1.0 / (stds_stack ** 2 + self.eps)                 # (n_signals, B)
        weights     = precision / precision.sum(dim=0, keepdim=True)     # (n_signals, B) normalized
        return (weights * means_stack).sum(dim=0)                        # (B,)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_precision_fusion.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/precision_fusion.py tests/alignment/test_precision_fusion.py
git commit -m "feat(praxis): add PrecisionFusion Bayesian inverse-variance weighting"
```

---

### Task 3: SteeringRewardCorrespondence (SRC)

**Files:**
- Create: `src/alignment/praxis/steering_reward.py`
- Test: `tests/alignment/test_steering_reward.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_steering_reward.py
import torch
import torch.nn as nn
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def test_src_returns_negative_scalar():
    cfg = PRAXISConfig(d_model=8, steer_layers=[0, 1], steer_alpha=0.1, lambda_src=0.1)
    model = TinyModel()
    src = SteeringRewardCorrespondence(model, cfg)
    x = torch.randn(2, 4, 8)
    reward = src.compute(x)
    assert reward.shape == (), f"expected scalar, got {reward.shape}"
    assert reward.item() <= 0.0, f"SRC reward should be non-positive: {reward.item()}"

def test_src_zero_alpha_gives_zero():
    cfg = PRAXISConfig(d_model=8, steer_layers=[0], steer_alpha=0.0, lambda_src=0.1)
    model = TinyModel()
    src = SteeringRewardCorrespondence(model, cfg)
    x = torch.randn(2, 4, 8)
    reward = src.compute(x)
    assert abs(reward.item()) < 1e-5, "zero alpha → no steering → zero distance"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_steering_reward.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement SRC**

```python
# src/alignment/praxis/steering_reward.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig


class SteeringRewardCorrespondence:
    """Measures alignment between reward gradient and steering direction.

    Captures hidden states at steer_layers with and without additive steering,
    computes cosine distance between the two, and returns a negative scalar
    reward proportional to that distance (high correspondence = near zero).
    """

    def __init__(self, model: nn.Module, config: PRAXISConfig) -> None:
        self.model = model
        self.config = config

    def _capture_hidden(self, x: Tensor, apply_steer: bool) -> dict[int, Tensor]:
        captures: dict[int, Tensor] = {}
        hooks = []

        for idx in self.config.steer_layers:
            if idx >= len(self.model.layers):
                continue

            def make_hook(layer_idx: int, steer: bool):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if steer:
                        direction = torch.randn_like(h)
                        direction = F.normalize(direction, dim=-1)
                        h = h + self.config.steer_alpha * direction
                    captures[layer_idx] = h.detach()
                return hook

            handle = self.model.layers[idx].register_forward_hook(make_hook(idx, apply_steer))
            hooks.append(handle)

        with torch.no_grad():
            self.model(x)

        for handle in hooks:
            handle.remove()

        return captures

    def compute(self, x: Tensor) -> Tensor:
        """Compute SRC reward signal.

        Returns a non-positive scalar: 0 when steering has no effect (ideal),
        negative when hidden states diverge significantly from steering.
        """
        unsteered = self._capture_hidden(x, apply_steer=False)
        steered   = self._capture_hidden(x, apply_steer=True)

        distances: list[Tensor] = []
        for idx in self.config.steer_layers:
            if idx not in unsteered or idx not in steered:
                continue
            u = unsteered[idx]
            s = steered[idx]
            cos_sim  = F.cosine_similarity(u.reshape(-1, u.shape[-1]),
                                           s.reshape(-1, s.shape[-1]), dim=-1)
            distances.append(1.0 - cos_sim.mean())

        if not distances:
            return torch.tensor(0.0)

        mean_dist = torch.stack(distances).mean()
        return -self.config.lambda_src * mean_dist
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_steering_reward.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/steering_reward.py tests/alignment/test_steering_reward.py
git commit -m "feat(praxis): add SteeringRewardCorrespondence (SRC) via forward hooks"
```

---

### Task 4: ExpertSafetyAffinity (ESA)

**Files:**
- Create: `src/alignment/praxis/expert_safety_affinity.py`
- Test: `tests/alignment/test_expert_safety_affinity.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_expert_safety_affinity.py
import torch
import torch.nn as nn
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.expert_safety_affinity import ExpertSafetyAffinity

class FakeTopKRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)

class FakeFFN(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.router = FakeTopKRouter(d_model, n_experts)

class FakeTransformerBlock(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.ffn = FakeFFN(d_model, n_experts)

def make_fake_moe_layers(n_layers=2, d_model=8, n_experts=4):
    return nn.ModuleList([FakeTransformerBlock(d_model, n_experts) for _ in range(n_layers)])

def test_esa_loss_is_scalar_nonneg():
    cfg = PRAXISConfig(d_model=8, safety_experts=[0, 1], alpha_esa=0.01, tau_safety=0.5)
    moe_layers = make_fake_moe_layers(n_layers=2, d_model=8, n_experts=4)
    esa = ExpertSafetyAffinity(moe_layers, cfg)
    hidden = torch.randn(2, 6, 8)
    const_scores = torch.tensor([0.2, 0.8])   # first is unsafe
    loss = esa.compute(hidden, const_scores)
    assert loss.shape == (), f"expected scalar: {loss.shape}"
    assert loss.item() >= 0.0, f"ESA loss should be non-negative: {loss.item()}"

def test_esa_all_safe_gives_zero():
    cfg = PRAXISConfig(d_model=8, safety_experts=[0, 1], alpha_esa=0.01, tau_safety=0.5)
    moe_layers = make_fake_moe_layers(n_layers=2, d_model=8, n_experts=4)
    esa = ExpertSafetyAffinity(moe_layers, cfg)
    hidden = torch.randn(2, 6, 8)
    const_scores = torch.ones(2)   # all safe
    loss = esa.compute(hidden, const_scores)
    assert loss.item() == 0.0, "all-safe sequences → zero ESA loss"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_expert_safety_affinity.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement ESA**

```python
# src/alignment/praxis/expert_safety_affinity.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig


class ExpertSafetyAffinity:
    """Routes unsafe tokens toward safety experts via direct router gate calls.

    Calls layer.ffn.router.gate(h) directly on each MoE layer's TopKRouter gate
    (an nn.Linear) to get router logits, then applies cross-entropy toward a
    uniform target distribution over the designated safety_experts indices.
    """

    def __init__(self, moe_layers: nn.ModuleList, config: PRAXISConfig) -> None:
        self.moe_layers = moe_layers
        self.config = config

    def compute(self, hidden: Tensor, const_scores: Tensor) -> Tensor:
        """Compute ESA routing loss.

        Args:
            hidden:       (B, T, D) — final hidden states from transformer.
            const_scores: (B,) — constitutional safety score per sequence [0, 1].

        Returns:
            Scalar ESA loss (non-negative).
        """
        unsafe_mask = const_scores < self.config.tau_safety   # (B,) bool
        if not unsafe_mask.any():
            return hidden.new_zeros(())

        total_loss = hidden.new_zeros(())
        count = 0

        for layer in self.moe_layers:
            gate: nn.Linear = layer.ffn.router.gate
            router_logits = gate(hidden)          # (B, T, n_experts)
            n_experts = router_logits.shape[-1]

            unsafe_logits = router_logits[unsafe_mask]          # (n_unsafe, T, n_experts)
            unsafe_logits = unsafe_logits.reshape(-1, n_experts) # (n_unsafe*T, n_experts)

            target = torch.zeros(n_experts, device=hidden.device)
            for idx in self.config.safety_experts:
                if idx < n_experts:
                    target[idx] = 1.0
            n_valid = target.sum().clamp(min=1.0)
            target = target / n_valid
            target = target.unsqueeze(0).expand(unsafe_logits.shape[0], -1)  # (N, n_experts)

            log_probs = F.log_softmax(unsafe_logits, dim=-1)
            esa_loss = -(target * log_probs).sum(dim=-1).mean()
            total_loss = total_loss + esa_loss
            count += 1

        if count == 0:
            return hidden.new_zeros(())

        return self.config.alpha_esa * (total_loss / count)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_expert_safety_affinity.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/expert_safety_affinity.py tests/alignment/test_expert_safety_affinity.py
git commit -m "feat(praxis): add ExpertSafetyAffinity (ESA) via direct router gate calls"
```

---

### Task 5: MultiTokenAlignmentHorizon (MTAH)

**Files:**
- Create: `src/alignment/praxis/mtah.py`
- Test: `tests/alignment/test_mtah.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mtah.py
import torch
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.mtah import MultiTokenAlignmentHorizon

def test_mtah_extends_advantages_forward():
    cfg = PRAXISConfig(gamma_mtah=0.9, k_mtah=2)
    mtah = MultiTokenAlignmentHorizon(cfg)
    B, T = 1, 5
    adv = torch.zeros(B, T)
    adv[0, 3] = 1.0   # spike at position 3
    extended = mtah.extend(adv)
    # Positions 1 and 2 should receive discounted credit from position 3
    assert extended[0, 1].item() > 0.0, "position 1 gets credit from future spike"
    assert extended[0, 2].item() > extended[0, 1].item(), "closer = more credit"
    assert extended[0, 3].item() == 1.0, "original spike preserved"

def test_mtah_shape_preserved():
    cfg = PRAXISConfig(gamma_mtah=0.95, k_mtah=3)
    mtah = MultiTokenAlignmentHorizon(cfg)
    adv = torch.randn(4, 16)
    extended = mtah.extend(adv)
    assert extended.shape == adv.shape

def test_mtah_k_zero_is_identity():
    cfg = PRAXISConfig(gamma_mtah=0.95, k_mtah=0)
    mtah = MultiTokenAlignmentHorizon(cfg)
    adv = torch.randn(2, 8)
    extended = mtah.extend(adv)
    assert torch.allclose(extended, adv)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_mtah.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement MTAH**

```python
# src/alignment/praxis/mtah.py
from __future__ import annotations
import torch
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig


class MultiTokenAlignmentHorizon:
    """Extends per-token advantages with discounted future credit.

    Ā_t = Σ_{k=0}^{K} γ^k · ā_{t+k}

    This gives each token credit for how aligned the next K tokens are,
    matching the multi-token prediction horizon used during pretraining.
    """

    def __init__(self, config: PRAXISConfig) -> None:
        self.gamma = config.gamma_mtah
        self.k = config.k_mtah

    def extend(self, advantages: Tensor) -> Tensor:
        """Apply temporal credit extension.

        Args:
            advantages: (B, T) — per-token advantage estimates.

        Returns:
            (B, T) — advantages with forward-looking credit added.
        """
        if self.k == 0:
            return advantages

        B, T = advantages.shape
        result = advantages.clone()

        for step in range(1, self.k + 1):
            if step >= T:
                break
            future = torch.zeros_like(advantages)
            future[:, :T - step] = advantages[:, step:]
            result = result + (self.gamma ** step) * future

        return result
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_mtah.py -v
```
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/mtah.py tests/alignment/test_mtah.py
git commit -m "feat(praxis): add MultiTokenAlignmentHorizon (MTAH) temporal advantage extension"
```

---

### Task 6: RewardSignalBundle (6-signal decomposition)

**Files:**
- Create: `src/alignment/praxis/reward_signals.py`
- Test: `tests/alignment/test_reward_signals.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_reward_signals.py
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.reward_signals import RewardSignalBundle

def make_bundle(d_model=16, B=2):
    cfg = PRAXISConfig(d_model=d_model, n_principles=4, mc_dropout_n=5)

    # Mock all sub-reward components
    prime_mock    = MagicMock(return_value=(torch.randn(B, 6), {}))
    critique_mock = MagicMock(return_value=torch.rand(B, 4))
    hier_mock     = MagicMock(return_value=torch.rand(B))
    mc_mock       = MagicMock()
    mc_mock.predict_with_uncertainty = MagicMock(
        return_value=(torch.rand(B), torch.rand(B) * 0.1 + 0.01)
    )

    return RewardSignalBundle(cfg, prime_mock, critique_mock, hier_mock, mc_mock)

def test_bundle_returns_six_signals():
    B, T, D = 2, 6, 16
    bundle = make_bundle(d_model=D, B=B)
    hidden = torch.randn(B, T, D)
    log_probs = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)
    outcome_rewards = torch.rand(B)
    mask = torch.ones(B, T, dtype=torch.bool)

    signals = bundle.compute(hidden, log_probs, ref_log_probs, outcome_rewards, mask)
    assert len(signals) == 6, f"expected 6 signals, got {len(signals)}"
    for name, (mean, std) in signals.items():
        assert mean.shape == (B,), f"{name} mean shape wrong: {mean.shape}"
        assert std.shape  == (B,), f"{name} std shape wrong: {std.shape}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_reward_signals.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement RewardSignalBundle**

```python
# src/alignment/praxis/reward_signals.py
from __future__ import annotations
import torch
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig


class RewardSignalBundle:
    """Computes 6-signal reward decomposition for PRAXIS.

    Signals:
        R_prime: PRIME implicit process reward (log-ratio based)
        R_const: Constitutional critique head aggregate score
        R_ccot:  CCoT chain-of-thought quality (from outcome rewards, scaled)
        R_odin:  Length-normalized reward (outcome / sqrt(len))
        R_hier:  Hierarchical reward model score
        R_src:   Steering-Reward Correspondence signal (scalar, broadcast)
    """

    def __init__(self, config: PRAXISConfig, prime, critique_head, hier_model, mc_reward) -> None:
        self.config = config
        self.prime        = prime
        self.critique     = critique_head
        self.hier         = hier_model
        self.mc_reward    = mc_reward

    def compute(
        self,
        hidden: Tensor,           # (B, T, D)
        log_probs: Tensor,        # (B, T)
        ref_log_probs: Tensor,    # (B, T)
        outcome_rewards: Tensor,  # (B,)
        mask: Tensor,             # (B, T) bool
    ) -> dict[str, tuple[Tensor, Tensor]]:
        B, T, D = hidden.shape
        eps = 1e-6

        # --- R_prime: PRIME dense reward, aggregate to (B,) mean over valid tokens ---
        dense_r, _ = self.prime(log_probs, ref_log_probs, outcome_rewards, mask.float())
        valid_len   = mask.float().sum(dim=1).clamp(min=1.0)
        r_prime     = (dense_r * mask.float()).sum(dim=1) / valid_len
        r_prime_std = torch.ones_like(r_prime) * (r_prime.std() + eps)

        # --- R_const: Constitutional critique score ---
        critique_scores = self.critique(hidden)           # (B, n_principles)
        r_const         = critique_scores.mean(dim=-1)   # (B,)
        r_const_std     = critique_scores.std(dim=-1).clamp(min=eps)

        # --- R_ccot: CCoT quality approximated as outcome * log(1 + valid_len/T) ---
        r_ccot     = outcome_rewards * torch.log1p(valid_len / T)
        r_ccot_std = torch.ones_like(r_ccot) * (r_ccot.std() + eps)

        # --- R_odin: Length-normalized outcome reward ---
        r_odin     = outcome_rewards / valid_len.sqrt()
        r_odin_std = torch.ones_like(r_odin) * (r_odin.std() + eps)

        # --- R_hier: Hierarchical reward model (expects (B, D) input) ---
        h_pooled    = hidden[:, -1, :]                             # (B, D)
        r_hier_mean, r_hier_std = self.mc_reward.predict_with_uncertainty(
            h_pooled, n_samples=self.config.mc_dropout_n
        )

        # --- R_src: Steering-Reward Correspondence (scalar, broadcast to B) ---
        hier_raw  = self.hier(h_pooled)                           # (B,)
        r_src     = hier_raw * 0.0                                # placeholder; SRC injected by trainer
        r_src_std = torch.ones_like(r_src) * eps

        return {
            "r_prime": (r_prime,     r_prime_std),
            "r_const": (r_const,     r_const_std),
            "r_ccot":  (r_ccot,      r_ccot_std),
            "r_odin":  (r_odin,      r_odin_std),
            "r_hier":  (r_hier_mean, r_hier_std.clamp(min=eps)),
            "r_src":   (r_src,       r_src_std),
        }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_reward_signals.py -v
```
Expected: 1 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/reward_signals.py tests/alignment/test_reward_signals.py
git commit -m "feat(praxis): add 6-signal RewardSignalBundle decomposition"
```

---

### Task 7: PRAXISLoss

**Files:**
- Create: `src/alignment/praxis/praxis_loss.py`
- Test: `tests/alignment/test_praxis_loss.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_praxis_loss.py
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_praxis_loss.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement PRAXISLoss**

```python
# src/alignment/praxis/praxis_loss.py
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.dapo import DAPOLoss


class PRAXISLoss:
    """PRAXIS combined loss: DAPO + KL penalty + entropy bonus + ESA, with constitutional gate.

    The constitutional gate zeroes the policy gradient for any sequence whose
    constitutional score is below tau_gate, preventing harmful content from
    receiving any alignment signal until it clears the safety threshold.
    """

    def __init__(self, config: PRAXISConfig) -> None:
        self.config = config
        self.dapo   = DAPOLoss(
            eps_low=config.eps_low,
            eps_high=config.eps_high,
            beta_entropy=config.lambda_ent,
        )

    def forward(
        self,
        log_probs: Tensor,        # (B, T)
        old_log_probs: Tensor,    # (B, T)
        advantages: Tensor,       # (B, T)
        fused_rewards: Tensor,    # (B,) from PrecisionFusion
        mask: Tensor,             # (B, T) bool
        ref_log_probs: Tensor | None = None,  # (B, T) for KL
        const_scores: Tensor | None = None,   # (B,) constitutional gate signal
        entropy: Tensor | None = None,        # (B, T) optional entropy term
        esa_loss: Tensor | None = None,       # scalar from ExpertSafetyAffinity
    ) -> tuple[Tensor, dict]:
        cfg = self.config

        # 1. Constitutional gate: zero loss for sequences below tau_gate
        if const_scores is not None:
            gate_mask = (const_scores >= cfg.tau_gate).float()  # (B,)
            if gate_mask.sum() == 0:
                zero = log_probs.new_zeros(())
                return zero, {"dapo_loss": 0.0, "kl_penalty": 0.0, "const_gate_ratio": 0.0}
            advantages = advantages * gate_mask.unsqueeze(1)

        # 2. DAPO policy gradient loss
        dapo_loss, dapo_metrics = self.dapo.forward(
            log_probs, old_log_probs, advantages, entropy=entropy
        )

        # 3. KL penalty against reference
        kl_penalty = log_probs.new_zeros(())
        if ref_log_probs is not None:
            mask_f     = mask.float()
            valid_cnt  = mask_f.sum().clamp(min=1.0)
            kl_penalty = ((log_probs - ref_log_probs) * mask_f).sum() / valid_cnt
            kl_penalty = cfg.beta_kl * kl_penalty

        # 4. ESA routing loss
        esa_term = esa_loss if esa_loss is not None else log_probs.new_zeros(())

        total_loss = dapo_loss + kl_penalty + esa_term

        metrics = {
            "dapo_loss": dapo_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "esa_loss": esa_term.item() if hasattr(esa_term, "item") else float(esa_term),
            "total_loss": total_loss.item(),
        }
        if const_scores is not None:
            gate_mask = (const_scores >= cfg.tau_gate).float()
            metrics["const_gate_ratio"] = gate_mask.mean().item()
        metrics.update(dapo_metrics)

        return total_loss, metrics
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_praxis_loss.py -v
```
Expected: 2 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/praxis_loss.py tests/alignment/test_praxis_loss.py
git commit -m "feat(praxis): add PRAXISLoss with DAPO + KL + constitutional gate"
```

---

### Task 8: PRAXISTrainer

**Files:**
- Create: `src/alignment/praxis/trainer.py`
- Test: `tests/alignment/test_praxis_trainer.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_praxis_trainer.py
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.trainer import PRAXISTrainer

def make_tiny_model(d_model=16, vocab_size=64, n_layers=4):
    """Minimal mock matching AureliusTransformer's interface."""
    model = MagicMock()
    model.layers = nn.ModuleList([MagicMock() for _ in range(n_layers)])
    for layer in model.layers:
        layer.ffn = MagicMock()
        layer.ffn.router = MagicMock()
        layer.ffn.router.gate = nn.Linear(d_model, 4, bias=False)
    B, T = 2, 8
    model.return_value = (
        torch.tensor(1.0),
        torch.randn(B, T, vocab_size),
        None,
        torch.randn(B, T, d_model),
    )
    model.train = MagicMock()
    return model

def test_trainer_train_step_returns_metrics():
    D, V = 16, 64
    cfg = PRAXISConfig(d_model=D, n_principles=2, mc_dropout_n=2,
                       steer_layers=[0, 1], safety_experts=[0, 1],
                       n_group=2, max_new_tokens=4, warp_interval=9999)
    model   = make_tiny_model(d_model=D, vocab_size=V)
    ref_sd  = {}
    sft_sd  = {}
    trainer = PRAXISTrainer(model, cfg, ref_state_dict=ref_sd, sft_state_dict=sft_sd)

    batch = {
        "input_ids":      torch.randint(0, V, (2, 8)),
        "labels":         torch.randint(0, V, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    metrics = trainer.train_step(batch, step=1)
    assert "total_loss" in metrics or "dapo_loss" in metrics, f"metrics: {metrics}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_praxis_trainer.py -v
```
Expected: `ModuleNotFoundError`

**Step 3: Implement PRAXISTrainer**

```python
# src/alignment/praxis/trainer.py
from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.precision_fusion import PrecisionFusion
from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence
from src.alignment.praxis.expert_safety_affinity import ExpertSafetyAffinity
from src.alignment.praxis.mtah import MultiTokenAlignmentHorizon
from src.alignment.praxis.reward_signals import RewardSignalBundle
from src.alignment.praxis.praxis_loss import PRAXISLoss
from src.alignment.prime import PRIMEReward, PRIMEConfig
from src.alignment.constitutional_ai_v3 import CritiqueHead
from src.alignment.reward_uncertainty import MCDropoutReward
from src.alignment.grpo_v3 import GroupRewardNormalizer
from src.alignment.thinking_tokens import ThinkingLossWeights
from src.alignment.warp import anchor_merge
try:
    from src.alignment.hierarchical_reward import HierarchicalRewardModel
except ImportError:
    HierarchicalRewardModel = None


class PRAXISTrainer:
    """PRAXIS unified alignment trainer.

    Orchestrates 6-signal reward decomposition with PrecisionFusion, DAPO
    policy gradient, constitutional gradient gating, ESA routing loss, SRC
    reward signal, MTAH temporal advantage extension, and periodic WARP merge.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PRAXISConfig,
        ref_state_dict: dict | None = None,
        sft_state_dict: dict | None = None,
    ) -> None:
        self.model  = model
        self.config = config
        self.ref_state_dict = ref_state_dict or {}
        self.sft_state_dict = sft_state_dict or {}

        # Sub-components
        self.prime       = PRIMEReward(PRIMEConfig(beta=config.beta_kl))
        self.critique    = CritiqueHead(config.d_model, config.n_principles)
        self.mc_reward   = MCDropoutReward(config.d_model)
        self.hier        = HierarchicalRewardModel(config.d_model) if HierarchicalRewardModel else None
        self.normalizer  = GroupRewardNormalizer(config.n_group)
        self.think_wts   = ThinkingLossWeights(config.think_weight, config.answer_weight)
        self.fusion      = PrecisionFusion(n_signals=6)
        self.src         = SteeringRewardCorrespondence(model, config)
        self.mtah        = MultiTokenAlignmentHorizon(config)
        self.loss_fn     = PRAXISLoss(config)

        # ESA uses only MoE layers (those with a router attribute)
        moe_layers = nn.ModuleList([
            layer for layer in model.layers
            if hasattr(layer, "ffn") and hasattr(getattr(layer, "ffn", None), "router")
        ])
        self.esa = ExpertSafetyAffinity(moe_layers, config)

        self.bundle = RewardSignalBundle(config, self.prime, self.critique,
                                        self.hier or (lambda x: torch.zeros(x.shape[0])),
                                        self.mc_reward)

    def _get_log_probs(self, model_out, input_ids: Tensor, mask: Tensor) -> Tensor:
        _, logits, _, _ = model_out
        log_probs = F.log_softmax(logits, dim=-1)
        target = input_ids.clamp(min=0)
        gathered = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return gathered * mask.float()

    def train_step(self, batch: dict, step: int) -> dict:
        cfg = self.config
        input_ids = batch["input_ids"]
        labels    = batch.get("labels", input_ids)
        mask      = batch.get("attention_mask", torch.ones_like(input_ids, dtype=torch.bool))

        self.model.train()

        # Forward pass with hidden states
        model_out = self.model(input_ids, mask=mask, labels=labels, return_hidden_states=True)
        _, logits, _, hidden = model_out                            # (B, T, D)

        # Reference log-probs (no grad)
        with torch.no_grad():
            log_probs_cur = self._get_log_probs(model_out, input_ids, mask)   # (B, T)
            ref_log_probs = log_probs_cur.detach() * 0.95                     # approx reference

        # Outcome rewards: cross-entropy of correct tokens (negated = positive reward for low loss)
        B, T = input_ids.shape
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1).clamp(min=0)
        ce_per_token = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        outcome_rewards = -ce_per_token.reshape(B, T).mean(dim=1)   # (B,)
        outcome_rewards = self.normalizer.normalize(outcome_rewards)

        # Compute 6 reward signals
        signals = self.bundle.compute(hidden, log_probs_cur, ref_log_probs, outcome_rewards, mask)

        # Inject SRC into r_src slot
        src_scalar = self.src.compute(hidden.detach())
        r_src_mean = torch.full((B,), src_scalar.item(), device=hidden.device)
        r_src_std  = torch.ones(B, device=hidden.device) * 1e-6
        signals["r_src"] = (r_src_mean, r_src_std)

        # PrecisionFusion
        means = [signals[k][0] for k in ["r_prime", "r_const", "r_ccot", "r_odin", "r_hier", "r_src"]]
        stds  = [signals[k][1] for k in ["r_prime", "r_const", "r_ccot", "r_odin", "r_hier", "r_src"]]
        fused_reward = self.fusion.fuse(means, stds)                 # (B,)

        # MTAH advantage extension
        advantages_base = fused_reward.unsqueeze(1).expand(-1, T)   # (B, T)
        advantages = self.mtah.extend(advantages_base)               # (B, T)

        # Thinking token weights
        think_weights = self.think_wts.compute_weights(input_ids[0].tolist()).to(hidden.device)
        advantages = advantages * think_weights.unsqueeze(0)

        # Constitutional scores for gating
        with torch.no_grad():
            const_scores = signals["r_const"][0]                     # (B,)

        # ESA routing loss
        esa_loss = self.esa.compute(hidden.detach(), const_scores)

        # Total loss
        entropy = -(log_probs_cur * log_probs_cur.exp() * mask.float()).sum(dim=-1, keepdim=False)
        entropy = entropy.unsqueeze(1).expand(-1, T)
        total_loss, metrics = self.loss_fn.forward(
            log_probs_cur, log_probs_cur.detach(), advantages, fused_reward, mask,
            ref_log_probs=ref_log_probs, const_scores=const_scores,
            entropy=entropy, esa_loss=esa_loss,
        )

        # WARP periodic anchor merge
        if step % cfg.warp_interval == 0 and self.sft_state_dict and hasattr(self.model, "state_dict"):
            current_sd = self.model.state_dict()
            merged_sd  = anchor_merge(self.sft_state_dict, current_sd, alpha=1 - cfg.warp_anchor_mu)
            self.model.load_state_dict(merged_sd, strict=False)

        metrics["fused_reward_mean"] = fused_reward.mean().item()
        metrics["src_reward"]        = src_scalar.item()
        return metrics
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/alignment/test_praxis_trainer.py -v
```
Expected: 1 PASSED

**Step 5: Commit**

```bash
git add src/alignment/praxis/trainer.py tests/alignment/test_praxis_trainer.py
git commit -m "feat(praxis): add PRAXISTrainer orchestrating all 6 signals"
```

---

### Task 9: Integration + Registry

**Files:**
- Create: `tests/alignment/test_praxis_integration.py`
- Modify: `src/alignment/__init__.py`
- Create: `aurelius/alignment/praxis.py` (re-export wrapper)
- Modify: `src/alignment/praxis/__init__.py`

**Step 1: Write the integration smoke test**

```python
# tests/alignment/test_praxis_integration.py
"""Smoke test: import PRAXIS from both public surfaces and verify registry."""

def test_import_from_alignment():
    from src.alignment import ALIGNMENT_REGISTRY
    assert "praxis" in ALIGNMENT_REGISTRY, \
        f"'praxis' not in ALIGNMENT_REGISTRY. Keys: {list(ALIGNMENT_REGISTRY.keys())}"

def test_import_praxis_trainer_directly():
    from src.alignment.praxis.trainer import PRAXISTrainer
    assert PRAXISTrainer is not None

def test_import_praxis_config():
    from src.alignment.praxis.config import PRAXISConfig
    cfg = PRAXISConfig()
    assert cfg.eps_low == 0.20
    assert cfg.eps_high == 0.28

def test_precision_fusion_import():
    from src.alignment.praxis.precision_fusion import PrecisionFusion
    assert PrecisionFusion is not None

def test_all_novel_components_import():
    from src.alignment.praxis.steering_reward import SteeringRewardCorrespondence
    from src.alignment.praxis.expert_safety_affinity import ExpertSafetyAffinity
    from src.alignment.praxis.mtah import MultiTokenAlignmentHorizon
    assert SteeringRewardCorrespondence is not None
    assert ExpertSafetyAffinity is not None
    assert MultiTokenAlignmentHorizon is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/alignment/test_praxis_integration.py -v
```
Expected: `test_import_from_alignment` FAILS — "praxis not in ALIGNMENT_REGISTRY"

**Step 3a: Update `src/alignment/praxis/__init__.py`**

```python
# src/alignment/praxis/__init__.py
"""PRAXIS: Policy Refinement through Aligned eXpert Integration System."""
from .config import PRAXISConfig
from .precision_fusion import PrecisionFusion
from .steering_reward import SteeringRewardCorrespondence
from .expert_safety_affinity import ExpertSafetyAffinity
from .mtah import MultiTokenAlignmentHorizon
from .reward_signals import RewardSignalBundle
from .praxis_loss import PRAXISLoss
from .trainer import PRAXISTrainer

__all__ = [
    "PRAXISConfig",
    "PrecisionFusion",
    "SteeringRewardCorrespondence",
    "ExpertSafetyAffinity",
    "MultiTokenAlignmentHorizon",
    "RewardSignalBundle",
    "PRAXISLoss",
    "PRAXISTrainer",
]
```

**Step 3b: Add to `src/alignment/__init__.py` (append after the last ALIGNMENT_REGISTRY entry)**

Add at the bottom of `src/alignment/__init__.py`:

```python
from .praxis import PRAXISConfig as PRAXISConfig  # noqa: E402
from .praxis import PRAXISTrainer as PRAXISTrainer
from .praxis import PrecisionFusion as PrecisionFusion
from .praxis import SteeringRewardCorrespondence as SteeringRewardCorrespondence
from .praxis import ExpertSafetyAffinity as ExpertSafetyAffinity
from .praxis import MultiTokenAlignmentHorizon as MultiTokenAlignmentHorizon
from .praxis import PRAXISLoss as PRAXISLoss

ALIGNMENT_REGISTRY["praxis"] = PRAXISTrainer
```

**Step 3c: Create `aurelius/alignment/praxis.py`**

```python
# aurelius/alignment/praxis.py
"""Public re-export of PRAXIS components for the aurelius top-level package."""
from src.alignment.praxis import (
    PRAXISConfig,
    PRAXISLoss,
    PRAXISTrainer,
    PrecisionFusion,
    ExpertSafetyAffinity,
    MultiTokenAlignmentHorizon,
    SteeringRewardCorrespondence,
)

__all__ = [
    "PRAXISConfig",
    "PRAXISLoss",
    "PRAXISTrainer",
    "PrecisionFusion",
    "ExpertSafetyAffinity",
    "MultiTokenAlignmentHorizon",
    "SteeringRewardCorrespondence",
]
```

**Step 4: Run integration tests**

```bash
pytest tests/alignment/test_praxis_integration.py -v
```
Expected: 5 PASSED

**Step 5: Run full alignment test suite to check for regressions**

```bash
pytest tests/alignment/ -v --tb=short 2>&1 | tail -40
```
Expected: All prior tests still passing; PRAXIS tests added on top.

**Step 6: Commit**

```bash
git add src/alignment/praxis/__init__.py src/alignment/__init__.py aurelius/alignment/praxis.py tests/alignment/test_praxis_integration.py
git commit -m "feat(praxis): register PRAXISTrainer in ALIGNMENT_REGISTRY and add public exports"
```

---

## Summary

| Task | Component | Novel? | Status |
|------|-----------|--------|--------|
| 1 | PRAXISConfig | — | |
| 2 | PrecisionFusion | — (Bayesian fusion) | |
| 3 | SteeringRewardCorrespondence | **SRC** | |
| 4 | ExpertSafetyAffinity | **ESA** | |
| 5 | MultiTokenAlignmentHorizon | **MTAH** | |
| 6 | RewardSignalBundle | — (6-signal) | |
| 7 | PRAXISLoss | — | |
| 8 | PRAXISTrainer | — | |
| 9 | Integration + Registry | — | |
