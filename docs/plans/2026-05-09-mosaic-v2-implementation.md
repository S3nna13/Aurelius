# MOSAIC v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `src/alignment/mosaic_v2/` — a production-ready training system that combines every alignment module in Aurelius plus three novel architecture-aware contributions (SRC, ESA, MTAH) into a single unified trainer.

**Architecture:** `MOSAICv2Trainer.train_step()` samples N=8 completions, computes 6 precision-weighted reward signals (PRIME + Constitutional + CCoT + ODIN + Hierarchical + SRC), applies constitutional gradient gating, per-token TokenDPO credit assignment, thinking-token loss weighting, DAPO asymmetric clip, ESA routing auxiliary loss, and MTAH temporal advantage extension, then updates the policy. WARP policy merging and AbsoluteZero self-curriculum are wired in via the curriculum stage controller.

**Tech Stack:** PyTorch 2.3+, Python 3.12, existing `src/alignment/` and `src/model/latent_steering.py` modules (all already implemented — we compose, not rewrite). Note: use `model.train(False)` as the PyTorch inference-mode setter (equivalent to `.eval()`).

---

## Task 1: Package scaffold and MOSAICv2Config

**Files:**
- Create: `src/alignment/mosaic_v2/__init__.py`
- Create: `src/alignment/mosaic_v2/config.py`
- Create: `aurelius/alignment/mosaic_v2.py`
- Create: `tests/alignment/test_mosaic_v2_config.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_config.py
"""Tests for MOSAICv2Config."""
from __future__ import annotations
from aurelius.alignment.mosaic_v2 import MOSAICv2Config


def test_default_config_instantiates():
    cfg = MOSAICv2Config()
    assert cfg.n_group == 8
    assert cfg.eps_low == 0.20
    assert cfg.eps_high == 0.28
    assert cfg.beta_kl == 0.04
    assert cfg.lambda_ent == 0.001
    assert cfg.d_model == 2048
    assert cfg.n_principles == 8
    assert cfg.n_criteria == 4
    assert cfg.steer_layers == [12, 16, 20]
    assert cfg.steer_alpha == 0.3
    assert cfg.lambda_src == 0.1
    assert cfg.safety_experts == [0, 1]
    assert cfg.alpha_esa == 0.01
    assert cfg.tau_safety == 0.5
    assert cfg.tau_gate == 0.4
    assert cfg.gamma_mtah == 0.95
    assert cfg.k_mtah == 2
    assert cfg.think_weight == 0.5
    assert cfg.answer_weight == 1.0
    assert cfg.warp_interval == 50
    assert cfg.warp_anchor_mu == 0.05
    assert cfg.mc_dropout_n == 20
    assert cfg.temperature == 0.9
    assert cfg.max_new_tokens == 128


def test_config_override():
    cfg = MOSAICv2Config(n_group=4, eps_low=0.1, d_model=512)
    assert cfg.n_group == 4
    assert cfg.eps_low == 0.1
    assert cfg.d_model == 512
```

**Step 2: Run to verify it fails**

```bash
cd /Users/christienantonio/Desktop/Aurelius
python -m pytest tests/alignment/test_mosaic_v2_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'aurelius.alignment.mosaic_v2'`

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/config.py
"""MOSAICv2Config — all hyperparameters for the MOSAIC v2 trainer."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MOSAICv2Config:
    # GRPO sampling
    n_group: int = 8
    temperature: float = 0.9
    max_new_tokens: int = 128

    # Model dimensions
    d_model: int = 2048

    # DAPO clip
    eps_low: float = 0.20
    eps_high: float = 0.28

    # KL + entropy
    beta_kl: float = 0.04
    lambda_ent: float = 0.001

    # Constitutional
    n_principles: int = 8
    tau_gate: float = 0.4        # gradient gating threshold

    # Hierarchical reward
    n_criteria: int = 4

    # SRC — Steering-Reward Correspondence
    steer_layers: list[int] = field(default_factory=lambda: [12, 16, 20])
    steer_alpha: float = 0.3
    lambda_src: float = 0.1

    # ESA — Expert Safety Affinity
    safety_experts: list[int] = field(default_factory=lambda: [0, 1])
    alpha_esa: float = 0.01
    tau_safety: float = 0.5      # constitutional score below which token is flagged

    # MTAH — Multi-Token Alignment Horizon
    gamma_mtah: float = 0.95
    k_mtah: int = 2

    # Thinking tokens
    think_weight: float = 0.5
    answer_weight: float = 1.0

    # WARP merging
    warp_interval: int = 50
    warp_anchor_mu: float = 0.05

    # Uncertainty quantification
    mc_dropout_n: int = 20
```

```python
# src/alignment/mosaic_v2/__init__.py
"""MOSAIC v2 — Multi-Objective Steering Architecture with Integrated Constitutional guidance."""
from .config import MOSAICv2Config

__all__ = ["MOSAICv2Config"]
```

```python
# aurelius/alignment/mosaic_v2.py
from src.alignment.mosaic_v2 import *  # noqa: F401,F403
from src.alignment.mosaic_v2.config import MOSAICv2Config  # noqa: F401
```

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_config.py -v
```
Expected: `2 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/ aurelius/alignment/mosaic_v2.py tests/alignment/test_mosaic_v2_config.py
git commit -m "feat(mosaic-v2): scaffold package and MOSAICv2Config"
```

---

## Task 2: PrecisionFusion — Bayesian inverse-variance weighting

**Files:**
- Create: `src/alignment/mosaic_v2/precision_fusion.py`
- Modify: `src/alignment/mosaic_v2/__init__.py`
- Modify: `aurelius/alignment/mosaic_v2.py`
- Create: `tests/alignment/test_mosaic_v2_precision_fusion.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_precision_fusion.py
"""Tests for PrecisionFusion Bayesian weighting."""
from __future__ import annotations
import torch
from aurelius.alignment.mosaic_v2 import PrecisionFusion


def test_equal_uncertainty_gives_uniform_weights():
    pf = PrecisionFusion()
    values = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0]), "c": torch.tensor([5.0, 6.0])}
    stds   = {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 1.0]), "c": torch.tensor([1.0, 1.0])}
    result = pf.fuse(values, stds)
    expected = (values["a"] + values["b"] + values["c"]) / 3
    assert torch.allclose(result, expected, atol=1e-5)


def test_low_uncertainty_signal_dominates():
    pf = PrecisionFusion()
    values = {"a": torch.tensor([10.0]), "b": torch.tensor([0.0])}
    stds   = {"a": torch.tensor([1e-6]),  "b": torch.tensor([10.0])}
    result = pf.fuse(values, stds)
    assert abs(result.item() - 10.0) < 0.1


def test_fuse_returns_correct_shape():
    pf = PrecisionFusion()
    B = 8
    values = {"a": torch.randn(B), "b": torch.randn(B)}
    stds   = {"a": torch.ones(B) * 0.5, "b": torch.ones(B) * 2.0}
    result = pf.fuse(values, stds)
    assert result.shape == (B,)


def test_single_signal_passthrough():
    pf = PrecisionFusion()
    v = torch.tensor([3.14, 2.71])
    result = pf.fuse({"x": v}, {"x": torch.ones(2)})
    assert torch.allclose(result, v)
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_precision_fusion.py -v
```
Expected: `ImportError`

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/precision_fusion.py
"""Bayesian precision-weighted fusion of multiple reward signals."""
from __future__ import annotations
import torch
from torch import Tensor


class PrecisionFusion:
    """Combine reward signals by inverse-variance (precision) weighting.

    For each signal j with estimate mu_j and standard deviation sigma_j:
        w_j = (1/sigma^2_j) / sum_k (1/sigma^2_k)
        R_combined = sum_j w_j * mu_j

    Signals with smaller uncertainty receive proportionally larger weight.
    """

    def fuse(
        self,
        values: dict[str, Tensor],  # signal_name -> (B,) reward estimates
        stds:   dict[str, Tensor],  # signal_name -> (B,) uncertainty estimates
        eps: float = 1e-8,
    ) -> Tensor:
        """Compute precision-weighted combination.

        Args:
            values: Dict of reward signal values, each shape (B,).
            stds:   Dict of uncertainty estimates (std dev), same keys, same shape.
            eps:    Numerical stability floor for variances.

        Returns:
            (B,) combined reward.
        """
        assert set(values.keys()) == set(stds.keys()), "values and stds must have same keys"
        keys = list(values.keys())

        precisions = {k: 1.0 / (stds[k] ** 2 + eps) for k in keys}
        total_precision = sum(precisions[k] for k in keys)

        result = sum(precisions[k] * values[k] for k in keys) / total_precision
        return result
```

Add to `src/alignment/mosaic_v2/__init__.py`:
```python
from .precision_fusion import PrecisionFusion
__all__ = ["MOSAICv2Config", "PrecisionFusion"]
```

Add to `aurelius/alignment/mosaic_v2.py`:
```python
from src.alignment.mosaic_v2.precision_fusion import PrecisionFusion  # noqa: F401
```

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_precision_fusion.py -v
```
Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/precision_fusion.py tests/alignment/test_mosaic_v2_precision_fusion.py
git commit -m "feat(mosaic-v2): add PrecisionFusion Bayesian inverse-variance weighting"
```

---

## Task 3: SRC — Steering-Reward Correspondence (Novel Contribution 1)

**Files:**
- Create: `src/alignment/mosaic_v2/steering_reward.py`
- Create: `tests/alignment/test_mosaic_v2_src.py`

**Background:** SRC measures cosine distance between hidden states with and without a constitutional steering vector applied at designated layers. Completions already aligned with safe representations need minimal steering (near-zero distance = near-zero penalty). Uses `src/model/latent_steering.py`.

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_src.py
"""Tests for Steering-Reward Correspondence (SRC)."""
from __future__ import annotations
import torch
import torch.nn as nn
from aurelius.alignment.mosaic_v2 import SteeringRewardCorrespondence


def _tiny_model(d_model=16, n_layers=4, vocab=100):
    class Layer(nn.Module):
        def forward(self, x, *a, **kw):
            return x + torch.zeros_like(x), None, torch.tensor(0.0)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])
            self.embed = nn.Embedding(vocab, d_model)
            self.d_model = d_model

        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x, _, _ = layer(x)
            return x, x, None

    return TinyModel()


def test_src_reward_is_nonpositive():
    model = _tiny_model()
    src = SteeringRewardCorrespondence(
        model=model, steer_layers=[1, 2], steer_alpha=0.1, lambda_src=0.5,
    )
    src.set_direction(torch.randn(16))
    reward = src.compute(torch.randint(0, 100, (1, 8)))
    assert reward.item() <= 0.0


def test_src_reward_near_zero_for_zero_direction():
    model = _tiny_model()
    src = SteeringRewardCorrespondence(model=model, steer_layers=[1], steer_alpha=0.5, lambda_src=1.0)
    src.set_direction(torch.zeros(16))
    reward = src.compute(torch.randint(0, 100, (1, 4)))
    assert abs(reward.item()) < 1e-3


def test_src_returns_scalar():
    model = _tiny_model()
    src = SteeringRewardCorrespondence(model=model, steer_layers=[0], steer_alpha=0.1, lambda_src=0.2)
    src.set_direction(torch.randn(16))
    reward = src.compute(torch.randint(0, 100, (1, 6)))
    assert reward.dim() == 0
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_src.py -v
```
Expected: `ImportError`

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/steering_reward.py
"""SRC: Steering-Reward Correspondence — Novel MOSAIC v2 contribution.

Measures cosine distance between unsteered and steered hidden states.
Small distance = model already behaves safely without intervention = bonus.
Large distance = model needs external correction = penalty.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SteeringRewardCorrespondence:
    """Compute SRC reward: -lambda_src * mean_cosine_distance(h_unsteered, h_steered).

    Args:
        model: AureliusTransformer with model.layers attribute.
        steer_layers: Layer indices to hook for hidden state capture.
        steer_alpha: Additive steering coefficient.
        lambda_src: Scale for the SRC reward (positive; output is negative).
    """

    def __init__(
        self,
        model: nn.Module,
        steer_layers: list[int],
        steer_alpha: float = 0.3,
        lambda_src: float = 0.1,
    ) -> None:
        self.model = model
        self.steer_layers = steer_layers
        self.steer_alpha = steer_alpha
        self.lambda_src = lambda_src
        self._direction: Tensor | None = None

    def set_direction(self, direction: Tensor) -> None:
        """Set the constitutional steering direction vector (d_model,)."""
        self._direction = direction

    def _collect_hidden(self, input_ids: Tensor, steer: bool) -> dict[int, Tensor]:
        """Forward pass capturing hidden states at steer_layers.

        Args:
            input_ids: (B, T) token ids.
            steer: If True, inject steering vector additively into hidden states.

        Returns:
            Dict mapping layer_idx -> (B, T, d_model) hidden states.
        """
        captured: dict[int, list[Tensor]] = {li: [] for li in self.steer_layers}
        handles = []
        direction = self._direction

        for layer_idx in self.steer_layers:
            layer = self.model.layers[layer_idx]

            def make_hook(li):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, (tuple, list)) else output
                    if steer and direction is not None:
                        d_norm = direction / (direction.norm() + 1e-8)
                        h = h + self.steer_alpha * d_norm.to(h.device)
                    captured[li].append(h.detach())
                return hook_fn

            handles.append(layer.register_forward_hook(make_hook(layer_idx)))

        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            for h in handles:
                h.remove()

        return {li: captured[li][0] for li in self.steer_layers}

    def compute(self, input_ids: Tensor) -> Tensor:
        """Compute SRC reward for a completion.

        Args:
            input_ids: (1, T) or (B, T) prompt+completion token ids.

        Returns:
            Scalar tensor: -lambda_src * mean_cosine_distance (always <= 0).
        """
        h_unsteered = self._collect_hidden(input_ids, steer=False)
        h_steered   = self._collect_hidden(input_ids, steer=True)

        distances = []
        for li in self.steer_layers:
            u = h_unsteered[li].reshape(-1, h_unsteered[li].shape[-1])
            s = h_steered[li].reshape(-1, h_steered[li].shape[-1])
            cos_sim = F.cosine_similarity(u, s, dim=-1)
            distances.append((1.0 - cos_sim).mean())

        mean_dist = torch.stack(distances).mean()
        return -self.lambda_src * mean_dist
```

Update `src/alignment/mosaic_v2/__init__.py`:
```python
from .config import MOSAICv2Config
from .precision_fusion import PrecisionFusion
from .steering_reward import SteeringRewardCorrespondence

__all__ = ["MOSAICv2Config", "PrecisionFusion", "SteeringRewardCorrespondence"]
```

Update `aurelius/alignment/mosaic_v2.py`:
```python
from src.alignment.mosaic_v2 import *  # noqa: F401,F403
from src.alignment.mosaic_v2.config import MOSAICv2Config  # noqa: F401
from src.alignment.mosaic_v2.precision_fusion import PrecisionFusion  # noqa: F401
from src.alignment.mosaic_v2.steering_reward import SteeringRewardCorrespondence  # noqa: F401
```

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_src.py -v
```
Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/steering_reward.py tests/alignment/test_mosaic_v2_src.py
git commit -m "feat(mosaic-v2): add SRC (Steering-Reward Correspondence) — novel contribution 1"
```

---

## Task 4: ESA — Expert Safety Affinity (Novel Contribution 2)

**Files:**
- Create: `src/alignment/mosaic_v2/expert_safety_affinity.py`
- Create: `tests/alignment/test_mosaic_v2_esa.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_esa.py
"""Tests for Expert Safety Affinity (ESA)."""
from __future__ import annotations
import torch
from aurelius.alignment.mosaic_v2 import ExpertSafetyAffinity


def test_esa_loss_is_scalar():
    esa = ExpertSafetyAffinity(safety_experts=[0, 1], n_experts=8)
    router_logits = torch.randn(12, 8)
    const_scores  = torch.tensor([0.2, 0.8, 0.3, 0.9, 0.1, 0.7, 0.4, 0.6, 0.15, 0.85, 0.25, 0.6])
    loss = esa.compute(router_logits, const_scores, tau_safety=0.5)
    assert loss.dim() == 0


def test_esa_loss_is_finite():
    esa = ExpertSafetyAffinity(safety_experts=[0, 1], n_experts=8)
    router_logits = torch.randn(16, 8)
    const_scores  = torch.rand(16)
    loss = esa.compute(router_logits, const_scores, tau_safety=0.5)
    assert torch.isfinite(loss)


def test_esa_zero_when_no_unsafe_tokens():
    esa = ExpertSafetyAffinity(safety_experts=[0, 1], n_experts=4)
    router_logits = torch.randn(8, 4)
    const_scores  = torch.ones(8) * 0.9
    loss = esa.compute(router_logits, const_scores, tau_safety=0.5)
    assert loss.item() == 0.0


def test_esa_has_gradient():
    esa = ExpertSafetyAffinity(safety_experts=[0, 1], n_experts=4)
    router_logits = torch.randn(8, 4, requires_grad=True)
    const_scores  = torch.tensor([0.1, 0.2, 0.8, 0.3, 0.9, 0.15, 0.7, 0.05])
    loss = esa.compute(router_logits, const_scores, tau_safety=0.5)
    loss.backward()
    assert router_logits.grad is not None
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_esa.py -v
```

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/expert_safety_affinity.py
"""ESA: Expert Safety Affinity — Novel MOSAIC v2 contribution.

Pushes constitutionally-flagged tokens to route toward designated safety experts
via a cross-entropy routing auxiliary loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class ExpertSafetyAffinity:
    """Compute ESA routing auxiliary loss.

    Args:
        safety_experts: Indices of designated safety experts (e.g. [0, 1]).
        n_experts: Total number of MoE experts.
    """

    def __init__(self, safety_experts: list[int], n_experts: int) -> None:
        self.safety_experts = safety_experts
        self.n_experts = n_experts

    def compute(
        self,
        router_logits: Tensor,  # (N_tokens, n_experts) raw router logits
        const_scores: Tensor,   # (N_tokens,) constitutional score per token
        tau_safety: float = 0.5,
    ) -> Tensor:
        """Compute ESA cross-entropy routing loss.

        Args:
            router_logits: Raw pre-softmax logits from the MoE router.
            const_scores:  Per-token constitutional score in [0, 1].
            tau_safety:    Threshold below which a token is flagged as unsafe.

        Returns:
            Scalar loss (0.0 if no unsafe tokens found).
        """
        unsafe_mask = const_scores < tau_safety
        if not unsafe_mask.any():
            return router_logits.sum() * 0.0

        unsafe_logits = router_logits[unsafe_mask]

        # Soft target: uniform over safety_experts, zero elsewhere
        target = torch.zeros(
            unsafe_logits.shape[0], self.n_experts,
            dtype=unsafe_logits.dtype,
            device=unsafe_logits.device,
        )
        target[:, self.safety_experts] = 1.0 / len(self.safety_experts)

        log_probs = F.log_softmax(unsafe_logits, dim=-1)
        loss = -(target * log_probs).sum(dim=-1).mean()
        return loss
```

Update `__init__.py` and re-export wrapper to include `ExpertSafetyAffinity`.

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_esa.py -v
```
Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/expert_safety_affinity.py tests/alignment/test_mosaic_v2_esa.py
git commit -m "feat(mosaic-v2): add ESA (Expert Safety Affinity) — novel contribution 2"
```

---

## Task 5: MTAH — Multi-Token Alignment Horizon (Novel Contribution 3)

**Files:**
- Create: `src/alignment/mosaic_v2/mtah.py`
- Create: `tests/alignment/test_mosaic_v2_mtah.py`

**Background:** Extends per-token GRPO advantages temporally: `A_t = sum_{k=0}^{K} gamma^k * a_{t+k}`. Prevents myopic alignment.

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_mtah.py
"""Tests for Multi-Token Alignment Horizon (MTAH)."""
from __future__ import annotations
import torch
from aurelius.alignment.mosaic_v2 import MultiTokenAlignmentHorizon


def test_mtah_k0_is_identity():
    mtah = MultiTokenAlignmentHorizon(gamma=0.95, k=0)
    adv = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert torch.allclose(mtah.extend(adv), adv)


def test_mtah_k1_correct_values():
    mtah = MultiTokenAlignmentHorizon(gamma=1.0, k=1)
    adv = torch.tensor([[1.0, 2.0, 3.0]])
    result = mtah.extend(adv)
    # t=0: 1+2=3; t=1: 2+3=5; t=2: 3 (no future)
    expected = torch.tensor([[3.0, 5.0, 3.0]])
    assert torch.allclose(result, expected)


def test_mtah_output_shape_preserved():
    mtah = MultiTokenAlignmentHorizon(gamma=0.95, k=2)
    adv = torch.randn(8, 64)
    assert mtah.extend(adv).shape == adv.shape


def test_mtah_gamma_zero_is_identity():
    mtah = MultiTokenAlignmentHorizon(gamma=0.0, k=3)
    adv = torch.randn(4, 10)
    assert torch.allclose(mtah.extend(adv), adv)
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_mtah.py -v
```

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/mtah.py
"""MTAH: Multi-Token Alignment Horizon — Novel MOSAIC v2 contribution.

Extends per-token GRPO advantages temporally:
    A_t = sum_{k=0}^{K} gamma^k * a_{t+k}

Prevents myopic alignment by penalizing tokens that lead to unsafe continuations.
"""
from __future__ import annotations

import torch
from torch import Tensor


class MultiTokenAlignmentHorizon:
    """Extend per-token advantages over K future steps.

    Args:
        gamma: Temporal discount factor (0 <= gamma <= 1).
        k:     Number of future steps to include (k=0 => identity).
    """

    def __init__(self, gamma: float = 0.95, k: int = 2) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.gamma = gamma
        self.k = k

    def extend(self, advantages: Tensor) -> Tensor:
        """Compute temporally-extended advantages.

        Args:
            advantages: (n_group, T) per-token advantages.

        Returns:
            (n_group, T) extended advantages.
        """
        if self.k == 0 or self.gamma == 0.0:
            return advantages.clone()

        result = advantages.clone()
        T = advantages.shape[1]

        for step in range(1, self.k + 1):
            if step >= T:
                break
            # Shift advantages left by `step`, zero-padding at the right end
            future = torch.zeros_like(advantages)
            future[:, : T - step] = advantages[:, step:]
            result = result + (self.gamma ** step) * future

        return result
```

Update `__init__.py` and re-export wrapper to include `MultiTokenAlignmentHorizon`.

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_mtah.py -v
```
Expected: `4 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/mtah.py tests/alignment/test_mosaic_v2_mtah.py
git commit -m "feat(mosaic-v2): add MTAH (Multi-Token Alignment Horizon) — novel contribution 3"
```

---

## Task 6: RewardSignals — 6-signal computation bundle

**Files:**
- Create: `src/alignment/mosaic_v2/reward_signals.py`
- Create: `tests/alignment/test_mosaic_v2_reward_signals.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_reward_signals.py
"""Tests for RewardSignals bundle."""
from __future__ import annotations
import torch
from aurelius.alignment.mosaic_v2 import MOSAICv2Config, RewardSignals
from src.alignment.constitutional_ai_v3 import CritiqueHead
from src.alignment.hierarchical_reward import HierarchicalRewardModel, RewardCriterion, CriterionWeights
from src.alignment.reward_uncertainty import MCDropoutReward


def _build_rs(d_model=16, n_principles=3, n_criteria=2):
    cfg = MOSAICv2Config(d_model=d_model, n_principles=n_principles, n_criteria=n_criteria)
    crit_head = CritiqueHead(d_model=d_model, n_principles=n_principles)
    criteria  = [RewardCriterion(f"c{i}", d_model) for i in range(n_criteria)]
    hier      = HierarchicalRewardModel(
        d_model=d_model, criteria=criteria,
        weight_scheme=CriterionWeights(n_criteria, learnable=True),
    )
    mc = {k: MCDropoutReward(d_model) for k in ["quality", "const", "length", "hier", "cot"]}
    return RewardSignals(cfg=cfg, critique_head=crit_head, hier_model=hier, mc_models=mc)


def test_reward_signals_output_keys():
    rs = _build_rs()
    n_group, T, d = 4, 8, 16
    out = rs.compute(
        torch.randn(n_group, T), torch.randn(n_group, T),
        torch.ones(n_group, T), torch.randn(n_group, T, d), src_reward=None,
    )
    for k in ["quality", "const", "length", "hier", "cot",
              "quality_std", "const_std", "length_std", "hier_std", "cot_std"]:
        assert k in out, f"Missing key: {k}"


def test_reward_signals_shapes():
    rs = _build_rs()
    n_group, T, d = 4, 8, 16
    out = rs.compute(
        torch.randn(n_group, T), torch.randn(n_group, T),
        torch.ones(n_group, T), torch.randn(n_group, T, d), src_reward=None,
    )
    for k in ["quality", "const", "length", "hier", "cot"]:
        assert out[k].shape == (n_group,), f"{k} wrong shape"


def test_reward_signals_all_finite():
    rs = _build_rs()
    out = rs.compute(
        torch.randn(4, 8), torch.randn(4, 8),
        torch.ones(4, 8), torch.randn(4, 8, 16), src_reward=None,
    )
    for k in ["quality", "const", "length", "hier", "cot"]:
        assert torch.isfinite(out[k]).all(), f"{k} not finite"
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_reward_signals.py -v
```

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/reward_signals.py
"""RewardSignals: orchestrates all 6 reward signal computations for MOSAIC v2."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.alignment.prime import PRIMEReward, PRIMEConfig
from src.alignment.constitutional_ai_v3 import CritiqueHead
from src.alignment.hierarchical_reward import HierarchicalRewardModel
from src.alignment.reward_uncertainty import MCDropoutReward

from .config import MOSAICv2Config


class RewardSignals:
    """Compute all 6 MOSAIC v2 reward signals with MC-Dropout uncertainty.

    Signals:
        quality  — PRIME implicit process reward (cumsum log-ratio)
        const    — Constitutional CritiqueHead mean principle score
        length   — ODIN-style length-normalized implicit reward
        hier     — HierarchicalRewardModel total reward
        cot      — Think-span quality proxy (mean log-ratio over think tokens)
        src      — Steering-Reward Correspondence (caller provides)
    """

    def __init__(
        self,
        cfg: MOSAICv2Config,
        critique_head: CritiqueHead,
        hier_model: HierarchicalRewardModel,
        mc_models: dict[str, MCDropoutReward],
    ) -> None:
        self.cfg = cfg
        self.critique_head = critique_head
        self.hier_model = hier_model
        self.mc_models = mc_models
        self._prime_cumsum = PRIMEReward(PRIMEConfig(credit_mode="cumsum"))
        self._prime_mean   = PRIMEReward(PRIMEConfig(credit_mode="mean"))

    def compute(
        self,
        log_probs: Tensor,       # (n_group, T) policy log-probs
        ref_log_probs: Tensor,   # (n_group, T) reference log-probs
        mask: Tensor,            # (n_group, T) valid token mask
        hidden: Tensor,          # (n_group, T, d_model) final-layer hidden states
        src_reward: Tensor | None = None,  # (n_group,) or None
    ) -> dict[str, Tensor]:
        """Compute all reward signals.

        Returns:
            Dict with keys: quality, const, length, hier, cot, src (all (n_group,))
            and *_std keys for uncertainty from MC-Dropout.
        """
        n_group = log_probs.shape[0]
        last_hidden = hidden[:, -1, :]  # (n_group, d_model)

        # R_quality: PRIME cumsum (total log-ratio, length-biased)
        tok_r = self._prime_cumsum.compute_implicit_rewards(log_probs, ref_log_probs, mask)
        R_quality = self._prime_cumsum.aggregate_step_rewards(tok_r, mask)

        # R_length: PRIME mean (ODIN length-normalized, removes length bias)
        tok_r_len = self._prime_mean.compute_implicit_rewards(log_probs, ref_log_probs, mask)
        R_length  = self._prime_mean.aggregate_step_rewards(tok_r_len, mask)

        # R_const: CritiqueHead mean constitutional score
        with torch.no_grad():
            scores = self.critique_head(hidden)   # (n_group, n_principles)
            R_const = scores.mean(dim=-1)          # (n_group,)

        # R_hier: HierarchicalRewardModel
        with torch.no_grad():
            hier_out = self.hier_model(last_hidden)
            R_hier = hier_out["total_reward"]

        # R_cot: proxy for think-span quality (fallback = length-normalized reward)
        R_cot = R_length.clone()

        # R_src: provided externally
        R_src = src_reward if src_reward is not None else torch.zeros(n_group, device=log_probs.device)

        # MC-Dropout uncertainty per signal
        def mc_std(name: str) -> Tensor:
            mc = self.mc_models.get(name)
            if mc is None:
                return torch.ones(n_group, device=log_probs.device) * 0.5
            _, std = mc.predict_with_uncertainty(last_hidden.detach(), n_samples=self.cfg.mc_dropout_n)
            return std.clamp(min=1e-6)

        return {
            "quality":    R_quality,
            "const":      R_const,
            "length":     R_length,
            "hier":       R_hier,
            "cot":        R_cot,
            "src":        R_src,
            "quality_std": mc_std("quality"),
            "const_std":   mc_std("const"),
            "length_std":  mc_std("length"),
            "hier_std":    mc_std("hier"),
            "cot_std":     mc_std("cot"),
            "src_std":     torch.ones(n_group, device=log_probs.device) * 0.1,
        }
```

Update `__init__.py` and re-export wrapper to include `RewardSignals`.

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_reward_signals.py -v
```
Expected: `3 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/reward_signals.py tests/alignment/test_mosaic_v2_reward_signals.py
git commit -m "feat(mosaic-v2): add RewardSignals 6-signal computation bundle"
```

---

## Task 7: MOSAICLoss — Full combined loss

**Files:**
- Create: `src/alignment/mosaic_v2/mosaic_loss.py`
- Create: `tests/alignment/test_mosaic_v2_loss.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_loss.py
"""Tests for MOSAICLoss."""
from __future__ import annotations
import torch
from aurelius.alignment.mosaic_v2 import MOSAICv2Config, MOSAICLoss


def _inputs(n_group=4, T=8):
    return dict(
        log_probs_policy = torch.randn(n_group, T, requires_grad=True),
        log_probs_ref    = torch.randn(n_group, T),
        advantages       = torch.randn(n_group, T),
        const_scores     = torch.rand(n_group),
        router_logits    = torch.randn(n_group * T, 8),
        token_ids        = torch.zeros(n_group, T, dtype=torch.long),
    )


def test_loss_is_scalar():
    loss, _ = MOSAICLoss(MOSAICv2Config())(**_inputs())
    assert loss.dim() == 0


def test_loss_is_finite():
    loss, _ = MOSAICLoss(MOSAICv2Config())(**_inputs())
    assert torch.isfinite(loss)


def test_loss_has_gradient():
    inp = _inputs()
    loss, _ = MOSAICLoss(MOSAICv2Config())(**inp)
    loss.backward()
    assert inp["log_probs_policy"].grad is not None


def test_all_gated_still_finite():
    cfg = MOSAICv2Config(tau_gate=0.5)
    inp = _inputs(n_group=4, T=4)
    inp["const_scores"] = torch.zeros(4)  # all below tau_gate
    loss, _ = MOSAICLoss(cfg)(**inp)
    assert torch.isfinite(loss)


def test_info_keys_present():
    _, info = MOSAICLoss(MOSAICv2Config())(**_inputs())
    for k in ["policy_loss", "kl", "esa_loss", "gated_fraction", "clip_fraction"]:
        assert k in info
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_loss.py -v
```

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/mosaic_loss.py
"""MOSAICLoss: full MOSAIC v2 combined training objective."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.alignment.thinking_tokens import THINK_START_TOKEN_ID, THINK_END_TOKEN_ID

from .config import MOSAICv2Config
from .expert_safety_affinity import ExpertSafetyAffinity


class MOSAICLoss:
    """Full MOSAIC v2 loss.

        L_MOSAIC = L_policy(gated, DAPO-clipped, TokenDPO-weighted, think-split)
                 + beta_kl * KL(pi_theta || pi_ref)
                 - lambda_ent * H(pi_theta)
                 + alpha_esa * L_ESA
    """

    def __init__(self, cfg: MOSAICv2Config) -> None:
        self.cfg = cfg
        self.esa = ExpertSafetyAffinity(
            safety_experts=cfg.safety_experts,
            n_experts=8,
        )

    def __call__(
        self,
        log_probs_policy: Tensor,    # (n_group, T) requires_grad=True
        log_probs_ref: Tensor,       # (n_group, T) detached
        advantages: Tensor,          # (n_group, T) MTAH-extended
        const_scores: Tensor,        # (n_group,) constitutional score per completion
        router_logits: Tensor,       # (n_group*T, n_experts)
        token_ids: Tensor,           # (n_group, T) for thinking-token detection
        entropy: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        n_group, T = log_probs_policy.shape
        cfg = self.cfg

        # 1. Thinking-token loss weights
        think_mask = (
            (token_ids == THINK_START_TOKEN_ID).cumsum(dim=-1) >
            (token_ids == THINK_END_TOKEN_ID).cumsum(dim=-1)
        )
        loss_weights = torch.where(
            think_mask,
            torch.tensor(cfg.think_weight, device=log_probs_policy.device),
            torch.tensor(cfg.answer_weight, device=log_probs_policy.device),
        )

        # 2. TokenDPO per-token credit weights (softmax over positions)
        token_credit = F.softmax(advantages / 1.0, dim=-1)

        # 3. Constitutional gradient gate
        gate = (const_scores >= cfg.tau_gate).float()
        gated_fraction = (1.0 - gate.mean()).item()

        # 4. DAPO asymmetric clip
        log_ratio = log_probs_policy - log_probs_ref.detach()
        ratio = torch.exp(log_ratio)

        eps_upper = torch.where(
            advantages >= 0,
            torch.full_like(advantages, cfg.eps_high),
            torch.full_like(advantages, cfg.eps_low),
        )
        ratio_clipped = torch.clamp(ratio, 1.0 - cfg.eps_low, 1.0 + eps_upper)
        clip_fraction = (ratio_clipped != ratio).float().mean().item()

        policy_surr = torch.min(ratio * advantages, ratio_clipped * advantages)
        token_loss  = -policy_surr * loss_weights * token_credit
        gated_loss  = gate.unsqueeze(-1) * token_loss
        policy_loss = gated_loss.mean()

        # 5. KL penalty
        kl_loss = cfg.beta_kl * (log_probs_policy - log_probs_ref.detach()).mean()

        # 6. Entropy bonus
        entropy_term = 0.0
        if entropy is not None:
            entropy_term = -cfg.lambda_ent * entropy.mean()

        # 7. ESA auxiliary loss
        if router_logits.dim() == 3:
            router_flat = router_logits.reshape(-1, router_logits.shape[-1])
        else:
            router_flat = router_logits
        const_per_token = const_scores.unsqueeze(-1).expand(n_group, T).reshape(-1)
        esa_loss = self.esa.compute(router_flat, const_per_token, cfg.tau_safety)

        # 8. Total
        loss = policy_loss + kl_loss + entropy_term + cfg.alpha_esa * esa_loss

        info = {
            "policy_loss":    policy_loss.item(),
            "kl":             kl_loss.item(),
            "esa_loss":       esa_loss.item(),
            "gated_fraction": gated_fraction,
            "clip_fraction":  clip_fraction,
        }
        return loss, info
```

Update `__init__.py` and re-export wrapper to include `MOSAICLoss`.

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_loss.py -v
```
Expected: `5 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/mosaic_loss.py tests/alignment/test_mosaic_v2_loss.py
git commit -m "feat(mosaic-v2): add MOSAICLoss full combined training objective"
```

---

## Task 8: MOSAICv2Trainer — Main trainer

**Files:**
- Create: `src/alignment/mosaic_v2/trainer.py`
- Create: `tests/alignment/test_mosaic_v2_trainer.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_trainer.py
"""Integration test: MOSAICv2Trainer.train_step() runs end-to-end."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from aurelius.alignment.mosaic_v2 import MOSAICv2Config, MOSAICv2Trainer
from src.alignment.constitutional_ai_v3 import CritiqueHead
from src.alignment.hierarchical_reward import HierarchicalRewardModel, RewardCriterion, CriterionWeights
from src.alignment.reward_uncertainty import MCDropoutReward


def _tiny_lm(d_model=16, vocab=50, n_layers=4):
    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(d_model)
            self.fc = nn.Linear(d_model, d_model)
        def forward(self, x, *a, **kw):
            return self.ln(self.fc(x)) + x, None, torch.tensor(0.0)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed  = nn.Embedding(vocab, d_model)
            self.layers = nn.ModuleList([TinyLayer() for _ in range(n_layers)])
            self.head   = nn.Linear(d_model, vocab, bias=False)
            self.d_model = d_model
        def forward(self, input_ids):
            x = self.embed(input_ids)
            for layer in self.layers:
                x, _, _ = layer(x)
            logits = self.head(x)
            return logits.mean() * 0, logits, None

    return TinyLM()


def test_trainer_returns_expected_keys():
    d_model, vocab = 16, 50
    cfg = MOSAICv2Config(
        d_model=d_model, n_group=2, max_new_tokens=4,
        n_principles=2, n_criteria=2, mc_dropout_n=2, steer_layers=[1, 2],
    )
    policy = _tiny_lm(d_model, vocab)
    ref    = _tiny_lm(d_model, vocab)
    critique_head = CritiqueHead(d_model=d_model, n_principles=cfg.n_principles)
    hier_model    = HierarchicalRewardModel(
        d_model=d_model,
        criteria=[RewardCriterion(f"c{i}", d_model) for i in range(cfg.n_criteria)],
        weight_scheme=CriterionWeights(cfg.n_criteria, learnable=True),
    )
    mc_models = {k: MCDropoutReward(d_model) for k in ["quality", "const", "length", "hier", "cot"]}

    trainer = MOSAICv2Trainer(
        policy_model=policy, ref_model=ref,
        critique_head=critique_head, hier_model=hier_model, mc_models=mc_models,
        config=cfg, optimizer=optim.AdamW(policy.parameters(), lr=1e-4),
    )

    prompt_ids = torch.randint(0, vocab, (1, 4))
    result = trainer.train_step(prompt_ids)

    assert "loss" in result
    assert "mean_reward" in result
    assert "gated_fraction" in result
    assert torch.isfinite(torch.tensor(result["loss"]))
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_trainer.py -v
```

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/trainer.py
"""MOSAICv2Trainer: full MOSAIC v2 training step orchestration."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.alignment.constitutional_ai_v3 import CritiqueHead
from src.alignment.hierarchical_reward import HierarchicalRewardModel
from src.alignment.reward_uncertainty import MCDropoutReward

from .config import MOSAICv2Config
from .mosaic_loss import MOSAICLoss
from .mtah import MultiTokenAlignmentHorizon
from .precision_fusion import PrecisionFusion
from .reward_signals import RewardSignals
from .steering_reward import SteeringRewardCorrespondence


class MOSAICv2Trainer:
    """Full MOSAIC v2 trainer combining all alignment modules.

    Args:
        policy_model: Model being trained.
        ref_model: Frozen reference model for KL regularization.
        critique_head: Trained CritiqueHead for constitutional scoring.
        hier_model: HierarchicalRewardModel with learnable criterion weights.
        mc_models: Dict of MCDropoutReward models for uncertainty estimation.
        config: MOSAICv2Config.
        optimizer: Optimizer for policy_model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        critique_head: CritiqueHead,
        hier_model: HierarchicalRewardModel,
        mc_models: dict[str, MCDropoutReward],
        config: MOSAICv2Config,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer

        self.reward_signals  = RewardSignals(
            cfg=config, critique_head=critique_head,
            hier_model=hier_model, mc_models=mc_models,
        )
        self.precision_fusion = PrecisionFusion()
        self.mtah             = MultiTokenAlignmentHorizon(gamma=config.gamma_mtah, k=config.k_mtah)
        self.mosaic_loss      = MOSAICLoss(config)

        self.src = SteeringRewardCorrespondence(
            model=policy_model,
            steer_layers=config.steer_layers,
            steer_alpha=config.steer_alpha,
            lambda_src=config.lambda_src,
        )
        self._src_active = False
        self._step = 0

    def set_steering_direction(self, direction: Tensor) -> None:
        """Activate SRC by providing a constitutional steering direction."""
        self.src.set_direction(direction)
        self._src_active = True

    def _sample_completions(
        self, prompt_ids: Tensor
    ) -> Tensor:
        """Sample n_group completions from policy.

        Returns:
            completion_ids: (n_group, max_new_tokens)
        """
        cfg = self.config
        self.policy_model.train(False)
        all_ids = []

        with torch.no_grad():
            for _ in range(cfg.n_group):
                cur = prompt_ids.clone()
                step_ids = []
                for _ in range(cfg.max_new_tokens):
                    _, logits, _ = self.policy_model(cur)
                    next_logits = logits[:, -1, :] / cfg.temperature
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    step_ids.append(int(next_token.item()))
                    cur = torch.cat([cur, next_token], dim=1)
                all_ids.append(torch.tensor(step_ids, dtype=torch.long))

        return torch.stack(all_ids)  # (n_group, T)

    def _compute_log_probs(
        self, model: nn.Module, prompt_ids: Tensor, completion_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute per-token log-probs and last-layer hidden states.

        Returns:
            log_probs: (n_group, T)
            hidden: (n_group, T, d_model)
        """
        n_group, T = completion_ids.shape
        prompt_len = prompt_ids.shape[1]
        full_ids = torch.cat([prompt_ids.expand(n_group, -1), completion_ids], dim=1)

        _, logits, _ = model(full_ids)  # (n_group, L, V)

        log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)
        completion_lp = log_probs_all[:, prompt_len - 1: prompt_len - 1 + T, :]
        token_lp = completion_lp.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)

        # Capture last-layer hidden states
        captured: list[Tensor] = []
        handle = model.layers[-1].register_forward_hook(
            lambda m, i, o: captured.append(o[0] if isinstance(o, (tuple, list)) else o)
        )
        with torch.no_grad():
            model(full_ids)
        handle.remove()
        hidden_full = captured[0]
        hidden = hidden_full[:, prompt_len: prompt_len + T, :]

        return token_lp, hidden.detach()

    def train_step(self, prompt_ids: Tensor) -> dict:
        """Run one full MOSAIC v2 training step.

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.

        Returns:
            Dict: loss, mean_reward, reward_std, gated_fraction, policy_loss, kl, esa_loss, step.
        """
        cfg = self.config

        # Sample completions
        completion_ids = self._sample_completions(prompt_ids)  # (n_group, T)
        T = completion_ids.shape[1]
        mask = torch.ones(cfg.n_group, T)

        # Policy log-probs (with gradient)
        self.policy_model.train()
        log_probs_policy, hidden = self._compute_log_probs(
            self.policy_model, prompt_ids, completion_ids
        )

        # Reference log-probs (no gradient)
        with torch.no_grad():
            self.ref_model.train(False)
            log_probs_ref, _ = self._compute_log_probs(
                self.ref_model, prompt_ids, completion_ids
            )

        # SRC reward (if direction set)
        if self._src_active:
            src_rewards = torch.stack([
                self.src.compute(
                    torch.cat([prompt_ids.expand(1, -1), completion_ids[i:i+1]], dim=1)
                )
                for i in range(cfg.n_group)
            ])
        else:
            src_rewards = None

        # All 6 reward signals
        signals = self.reward_signals.compute(
            log_probs_policy.detach(), log_probs_ref.detach(),
            mask, hidden, src_reward=src_rewards,
        )

        # Precision-weighted fusion
        values = {k: signals[k] for k in ["quality", "const", "length", "hier", "cot", "src"]}
        stds   = {k: signals.get(f"{k}_std", torch.ones_like(signals[k]) * 0.5) for k in values}
        R_combined = self.precision_fusion.fuse(values, stds)

        # Group advantages (Dr.GRPO bias-free)
        if cfg.n_group > 1:
            adv = (R_combined - R_combined.mean()) / (R_combined.std() + 1e-8)
        else:
            adv = torch.zeros_like(R_combined)

        # Expand + MTAH extension
        adv_token    = adv.unsqueeze(-1).expand(-1, T)
        adv_extended = self.mtah.extend(adv_token)

        # Placeholder router logits (zeros; replaced with real MoE logits when active)
        router_logits = torch.zeros(cfg.n_group * T, 8)

        # MOSAIC loss
        loss, info = self.mosaic_loss(
            log_probs_policy=log_probs_policy,
            log_probs_ref=log_probs_ref.detach(),
            advantages=adv_extended,
            const_scores=signals["const"].detach(),
            router_logits=router_logits,
            token_ids=completion_ids,
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        self._step += 1
        return {
            "loss":           loss.item(),
            "mean_reward":    R_combined.mean().item(),
            "reward_std":     R_combined.std().item() if cfg.n_group > 1 else 0.0,
            "gated_fraction": info["gated_fraction"],
            "policy_loss":    info["policy_loss"],
            "kl":             info["kl"],
            "esa_loss":       info["esa_loss"],
            "step":           self._step,
        }
```

Update `__init__.py` and re-export wrapper to include `MOSAICv2Trainer`.

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_trainer.py -v
```
Expected: `1 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/trainer.py tests/alignment/test_mosaic_v2_trainer.py
git commit -m "feat(mosaic-v2): add MOSAICv2Trainer full train_step orchestration"
```

---

## Task 9: CurriculumController — Stage transitions

**Files:**
- Create: `src/alignment/mosaic_v2/curriculum.py`
- Create: `tests/alignment/test_mosaic_v2_curriculum.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_curriculum.py
"""Tests for CurriculumController stage transitions."""
from __future__ import annotations
from aurelius.alignment.mosaic_v2 import CurriculumController


def test_stage_aria_at_step_0():
    assert CurriculumController().current_stage(0) == "ARIA"

def test_stage_aurora_at_step_1500():
    assert CurriculumController().current_stage(1500) == "AURORA"

def test_stage_mosaic_at_step_5000():
    assert CurriculumController().current_stage(5000) == "MOSAIC"

def test_src_inactive_in_aria():
    assert not CurriculumController().get_stage_config(500)["src_active"]

def test_src_active_in_mosaic():
    assert CurriculumController().get_stage_config(5000)["src_active"]

def test_esa_active_in_mosaic():
    assert CurriculumController().get_stage_config(6000)["esa_active"]

def test_lambda_src_ramps_up():
    ctrl = CurriculumController()
    assert ctrl.get_stage_config(8000)["lambda_src"] >= ctrl.get_stage_config(4500)["lambda_src"]
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_curriculum.py -v
```

**Step 3: Implement**

```python
# src/alignment/mosaic_v2/curriculum.py
"""CurriculumController: manages ARIA -> AURORA -> MOSAIC stage transitions."""
from __future__ import annotations


class CurriculumController:
    """Controls active components per training stage.

    Stage boundaries:
        ARIA   : [0, 1000)    -- SFT + constitutional filtering
        AURORA : [1000, 4000) -- 4-signal GRPO
        MOSAIC : [4000, inf)  -- Full 6-signal + SRC + ESA + MTAH + WARP
    """

    ARIA_END   = 1000
    AURORA_END = 4000
    SRC_START  = 4500
    ESA_START  = 5000
    MTAH_START = 6000

    def current_stage(self, step: int) -> str:
        if step < self.ARIA_END:
            return "ARIA"
        if step < self.AURORA_END:
            return "AURORA"
        return "MOSAIC"

    def get_stage_config(self, step: int) -> dict:
        """Return active component flags and ramped hyperparams for this step."""
        stage       = self.current_stage(step)
        src_active  = step >= self.SRC_START
        esa_active  = step >= self.ESA_START
        mtah_active = step >= self.MTAH_START
        warp_active = stage == "MOSAIC"
        az_active   = step >= self.ARIA_END + 500

        if src_active:
            ramp = min(1.0, (step - self.SRC_START) / 1500.0)
            lambda_src = 0.1 + 0.2 * ramp
        else:
            lambda_src = 0.0

        if esa_active:
            ramp = min(1.0, (step - self.ESA_START) / 1000.0)
            alpha_esa = 0.005 + 0.005 * ramp
        else:
            alpha_esa = 0.0

        active_signals = ["quality", "const", "length", "hier"]
        if stage == "MOSAIC":
            active_signals.append("cot")
        if src_active:
            active_signals.append("src")

        return {
            "stage":          stage,
            "src_active":     src_active,
            "esa_active":     esa_active,
            "mtah_active":    mtah_active,
            "warp_active":    warp_active,
            "az_active":      az_active,
            "lambda_src":     lambda_src,
            "alpha_esa":      alpha_esa,
            "active_signals": active_signals,
        }
```

Update `__init__.py` and re-export wrapper to include `CurriculumController`.

**Step 4: Run to verify it passes**

```bash
python -m pytest tests/alignment/test_mosaic_v2_curriculum.py -v
```
Expected: `7 passed`

**Step 5: Commit**

```bash
git add src/alignment/mosaic_v2/curriculum.py tests/alignment/test_mosaic_v2_curriculum.py
git commit -m "feat(mosaic-v2): add CurriculumController ARIA->AURORA->MOSAIC stage transitions"
```

---

## Task 10: Registration + Integration Smoke Test

**Files:**
- Modify: `src/alignment/mosaic_v2/__init__.py` (finalize all exports)
- Modify: `aurelius/alignment/mosaic_v2.py` (finalize all re-exports)
- Modify: `src/alignment/__init__.py` (register in ALIGNMENT_REGISTRY)
- Create: `tests/alignment/test_mosaic_v2_integration.py`

**Step 1: Write the failing test**

```python
# tests/alignment/test_mosaic_v2_integration.py
"""Integration smoke test: all MOSAIC v2 exports are importable and registered."""
from __future__ import annotations


def test_all_exports_importable():
    from aurelius.alignment.mosaic_v2 import (
        MOSAICv2Config, PrecisionFusion, SteeringRewardCorrespondence,
        ExpertSafetyAffinity, MultiTokenAlignmentHorizon, RewardSignals,
        MOSAICLoss, MOSAICv2Trainer, CurriculumController,
    )
    assert MOSAICv2Config is not None


def test_registered_in_alignment_registry():
    from src.alignment import ALIGNMENT_REGISTRY
    assert "mosaic_v2" in ALIGNMENT_REGISTRY


def test_full_config_roundtrip():
    from aurelius.alignment.mosaic_v2 import MOSAICv2Config
    cfg = MOSAICv2Config(n_group=4, steer_layers=[10, 15, 20])
    assert cfg.n_group == 4
    assert cfg.steer_layers == [10, 15, 20]


def test_precision_fusion_integration():
    import torch
    from aurelius.alignment.mosaic_v2 import PrecisionFusion
    pf = PrecisionFusion()
    n = 8
    values = {k: torch.randn(n) for k in ["a", "b", "c"]}
    stds   = {k: torch.rand(n).clamp(min=0.01) for k in ["a", "b", "c"]}
    result = pf.fuse(values, stds)
    assert result.shape == (n,) and torch.isfinite(result).all()


def test_curriculum_covers_all_stages():
    from aurelius.alignment.mosaic_v2 import CurriculumController
    ctrl = CurriculumController()
    for step in [0, 500, 1000, 2000, 4000, 5500, 7000, 9000]:
        cfg = ctrl.get_stage_config(step)
        assert "stage" in cfg and "active_signals" in cfg
```

**Step 2: Run to verify it fails**

```bash
python -m pytest tests/alignment/test_mosaic_v2_integration.py -v
```

**Step 3: Finalize `src/alignment/mosaic_v2/__init__.py`**

```python
"""MOSAIC v2 — Multi-Objective Steering Architecture with Integrated Constitutional guidance."""
from .config import MOSAICv2Config
from .curriculum import CurriculumController
from .expert_safety_affinity import ExpertSafetyAffinity
from .mosaic_loss import MOSAICLoss
from .mtah import MultiTokenAlignmentHorizon
from .precision_fusion import PrecisionFusion
from .reward_signals import RewardSignals
from .steering_reward import SteeringRewardCorrespondence
from .trainer import MOSAICv2Trainer

__all__ = [
    "MOSAICv2Config",
    "CurriculumController",
    "ExpertSafetyAffinity",
    "MOSAICLoss",
    "MultiTokenAlignmentHorizon",
    "PrecisionFusion",
    "RewardSignals",
    "SteeringRewardCorrespondence",
    "MOSAICv2Trainer",
]
```

**Finalize `aurelius/alignment/mosaic_v2.py`**

```python
from src.alignment.mosaic_v2 import *  # noqa: F401,F403
from src.alignment.mosaic_v2 import (  # noqa: F401
    MOSAICv2Config, CurriculumController, ExpertSafetyAffinity,
    MOSAICLoss, MultiTokenAlignmentHorizon, PrecisionFusion,
    RewardSignals, SteeringRewardCorrespondence, MOSAICv2Trainer,
)
```

**Add to `src/alignment/__init__.py`** (near bottom, before end of file):
```python
from .mosaic_v2 import MOSAICv2Trainer as MOSAICv2Trainer  # noqa: E402
from .mosaic_v2 import MOSAICv2Config as MOSAICv2Config
from .mosaic_v2 import CurriculumController as CurriculumController
from .mosaic_v2 import ExpertSafetyAffinity as ExpertSafetyAffinity
from .mosaic_v2 import MOSAICLoss as MOSAICLoss
from .mosaic_v2 import MultiTokenAlignmentHorizon as MultiTokenAlignmentHorizon
from .mosaic_v2 import PrecisionFusion as PrecisionFusion
from .mosaic_v2 import RewardSignals as RewardSignals
from .mosaic_v2 import SteeringRewardCorrespondence as SteeringRewardCorrespondence

ALIGNMENT_REGISTRY["mosaic_v2"] = MOSAICv2Trainer
```

Also add to `__all__` in `src/alignment/__init__.py`:
```python
"CurriculumController",
"ExpertSafetyAffinity",
"MOSAICLoss",
"MOSAICv2Config",
"MOSAICv2Trainer",
"MultiTokenAlignmentHorizon",
"PrecisionFusion",
"RewardSignals",
"SteeringRewardCorrespondence",
```

**Step 4: Run all MOSAIC v2 tests**

```bash
python -m pytest tests/alignment/test_mosaic_v2_*.py -v
```
Expected: all 30+ tests pass

**Step 5: Run full alignment suite — check for regressions**

```bash
python -m pytest tests/alignment/ -q --tb=short 2>&1 | tail -20
```
Expected: existing tests continue to pass

**Step 6: Final commit**

```bash
git add src/alignment/mosaic_v2/__init__.py aurelius/alignment/mosaic_v2.py \
        src/alignment/__init__.py tests/alignment/test_mosaic_v2_integration.py
git commit -m "feat(mosaic-v2): register MOSAICv2Trainer in ALIGNMENT_REGISTRY, finalize all exports

MOSAIC v2 complete. Combines every alignment module in src/alignment/ into a
single unified trainer with three novel architecture-aware contributions:
SRC (Steering-Reward Correspondence via latent_steering.py hooks),
ESA (Expert Safety Affinity via SparseMoE routing auxiliary loss),
MTAH (Multi-Token Alignment Horizon via MTP temporal extension).
Full 6-signal Bayesian precision-weighted reward fusion with ARIA->AURORA->MOSAIC
curriculum control. 10 source files, 10 test files, 30+ tests."
```

---

## Files Created Summary

```
src/alignment/mosaic_v2/
├── __init__.py               -- All exports
├── config.py                 -- MOSAICv2Config (all hyperparams)
├── precision_fusion.py       -- PrecisionFusion (Bayesian inverse-variance)
├── steering_reward.py        -- SteeringRewardCorrespondence (SRC -- Novel 1)
├── expert_safety_affinity.py -- ExpertSafetyAffinity (ESA -- Novel 2)
├── mtah.py                   -- MultiTokenAlignmentHorizon (MTAH -- Novel 3)
├── reward_signals.py         -- RewardSignals (6-signal bundle)
├── mosaic_loss.py            -- MOSAICLoss (full combined objective)
├── trainer.py                -- MOSAICv2Trainer (train_step)
└── curriculum.py             -- CurriculumController (ARIA->AURORA->MOSAIC)

aurelius/alignment/
└── mosaic_v2.py              -- Re-export wrapper

tests/alignment/
├── test_mosaic_v2_config.py
├── test_mosaic_v2_precision_fusion.py
├── test_mosaic_v2_src.py
├── test_mosaic_v2_esa.py
├── test_mosaic_v2_mtah.py
├── test_mosaic_v2_reward_signals.py
├── test_mosaic_v2_loss.py
├── test_mosaic_v2_trainer.py
├── test_mosaic_v2_curriculum.py
└── test_mosaic_v2_integration.py
```
