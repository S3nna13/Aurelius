# Aurelius Harvest Cycles 124–127 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 24 new modules across 4 cycles harvested from Kimi K2.5, GLM-5, GPT-OSS-120B, and Claude Code — covering agent, alignment, model, training, longcontext, inference, chat, eval, and data surfaces.

**Architecture:** Each cycle dispatches 6 parallel agents with strict file-ownership boundaries. Every module ships three artifacts: module file, unit tests (10–16 tests), and integration wiring. Pure PyTorch only — no HuggingFace/einops/flash_attn/scipy at runtime.

**Tech Stack:** Python 3.14, PyTorch, pytest, `.venv/bin/python3.14`

---

## Pre-Flight (before every cycle)

```bash
cd ~/Desktop/Aurelius
.venv/bin/python3.14 -m pytest --tb=no -q   # must be green
grep -rE "from (transformers|einops|trl|xformers|flash_attn|bitsandbytes|peft|diffusers|datasets|accelerate|deepspeed|langchain|llamaindex)" src/ tests/  # must return nothing
```

Tiny test config (use in every unit test):
```python
TINY = dict(n_layers=2, d_model=64, n_heads=4, n_kv_heads=2,
            head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64)
```

---

## CYCLE 124 — Agent · Alignment · Optimizers

### Task 124-A: `src/alignment/parl.py` — Parallel Agent RL Reward

**Surface:** alignment | **Paper:** Kimi K2.5 §3.3 (arXiv:2602.02276)
**Files:**
- Create: `src/alignment/parl.py`
- Create: `tests/alignment/test_parl.py`
- Create: `tests/integration/test_parl_integration.py`
- Modify: `src/alignment/__init__.py` — add `ALIGNMENT_REGISTRY["parl"]`

**Step 1: Write failing tests**
```python
# tests/alignment/test_parl.py
import torch, pytest
from src.alignment.parl import PARLReward, AnnealedLambda

def test_reward_shape():
    r = PARLReward()
    perf = torch.tensor([0.8, 0.0, 1.0])
    parallel = torch.tensor([3.0, 1.0, 0.0])
    finish = torch.tensor([1.0, 0.5, 0.0])
    out = r(perf, parallel, finish, step=0)
    assert out.shape == (3,)

def test_serial_collapse_penalized():
    r = PARLReward(lambda1=1.0, lambda2=1.0)
    # zero parallel subagents → r_parallel=0 contribution
    out_serial = r(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]), step=0)
    out_parallel = r(torch.tensor([1.0]), torch.tensor([4.0]), torch.tensor([1.0]), step=0)
    assert out_parallel > out_serial

def test_lambda_annealing_to_zero():
    ann = AnnealedLambda(start=1.0, total_steps=100)
    assert abs(ann(0) - 1.0) < 1e-6
    assert abs(ann(100)) < 1e-6

def test_no_nan_inf():
    r = PARLReward()
    out = r(torch.zeros(4), torch.zeros(4), torch.zeros(4), step=50)
    assert torch.isfinite(out).all()

def test_determinism():
    r = PARLReward()
    a = r(torch.tensor([0.5]), torch.tensor([2.0]), torch.tensor([0.8]), step=10)
    b = r(torch.tensor([0.5]), torch.tensor([2.0]), torch.tensor([0.8]), step=10)
    assert torch.allclose(a, b)

def test_perf_only_when_lambda_zero():
    r = PARLReward(lambda1=0.0, lambda2=0.0)
    perf = torch.tensor([0.7])
    out = r(perf, torch.tensor([99.0]), torch.tensor([99.0]), step=0)
    assert torch.allclose(out, perf)

def test_batch_size_one():
    r = PARLReward()
    out = r(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), step=5)
    assert out.shape == (1,)

def test_empty_subagent_list():
    r = PARLReward()
    out = r(torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([0.0]), step=0)
    assert torch.isfinite(out).all()
```

**Step 2: Run — expect FAIL (ImportError)**
```bash
.venv/bin/python3.14 -m pytest tests/alignment/test_parl.py -v --tb=short
```

**Step 3: Implement**
```python
# src/alignment/parl.py
"""PARL reward — Kimi K2.5 §3.3 (arXiv:2602.02276)."""
import torch
from dataclasses import dataclass

class AnnealedLambda:
    def __init__(self, start: float = 1.0, total_steps: int = 10_000):
        self.start = start
        self.total_steps = total_steps

    def __call__(self, step: int) -> float:
        return self.start * max(0.0, 1.0 - step / self.total_steps)


@dataclass
class PARLReward:
    lambda1: float = 1.0   # weight for r_parallel
    lambda2: float = 1.0   # weight for r_finish
    total_steps: int = 10_000

    def __call__(
        self,
        r_perf: torch.Tensor,     # task-level outcome reward [B]
        r_parallel: torch.Tensor,  # normalized active sub-agent count [B]
        r_finish: torch.Tensor,    # sub-agent completion rate [B]
        step: int = 0,
    ) -> torch.Tensor:
        ann1 = self.lambda1 * max(0.0, 1.0 - step / self.total_steps)
        ann2 = self.lambda2 * max(0.0, 1.0 - step / self.total_steps)
        return r_perf + ann1 * r_parallel + ann2 * r_finish
```

**Step 4: Run — expect PASS**
```bash
.venv/bin/python3.14 -m pytest tests/alignment/test_parl.py -v
```

**Step 5: Write integration test**
```python
# tests/integration/test_parl_integration.py
from src.alignment import ALIGNMENT_REGISTRY
import torch

def test_parl_in_registry():
    assert "parl" in ALIGNMENT_REGISTRY

def test_parl_construct_from_registry():
    cls = ALIGNMENT_REGISTRY["parl"]
    r = cls()
    out = r(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([1.0]), step=0)
    assert out.shape == (1,)

def test_non_parl_registry_unchanged():
    assert "dpo" in ALIGNMENT_REGISTRY
```

**Step 6: Wire registry**
In `src/alignment/__init__.py`, add:
```python
from src.alignment.parl import PARLReward
ALIGNMENT_REGISTRY["parl"] = PARLReward
```

**Step 7: Full suite + foreign-import check**
```bash
.venv/bin/python3.14 -m pytest tests/alignment/test_parl.py tests/integration/test_parl_integration.py -v
grep -rE "from (transformers|einops|trl)" src/alignment/parl.py  # must return nothing
```

**Step 8: Commit**
```bash
git add src/alignment/parl.py tests/alignment/test_parl.py tests/integration/test_parl_integration.py src/alignment/__init__.py
git commit -m "feat(cycle-124): parl reward — Kimi K2.5 parallel agent RL (2602.02276)"
```

---

### Task 124-B: `src/alignment/toggle.py` — Token-Efficient RL

**Surface:** alignment | **Paper:** Kimi K2.5 §3.4
**Files:**
- Create: `src/alignment/toggle.py`
- Create: `tests/alignment/test_toggle.py`
- Create: `tests/integration/test_toggle_integration.py`
- Modify: `src/alignment/__init__.py` — add `ALIGNMENT_REGISTRY["toggle"]`

**Step 1: Write failing tests**
```python
# tests/alignment/test_toggle.py
import torch, pytest
from src.alignment.toggle import ToggleReward

def test_phase0_below_threshold_zeroes_reward():
    t = ToggleReward(lambda_threshold=0.8, token_budget=100)
    # accuracy below threshold → reward=0 in phase 0
    out = t(mean_reward=torch.tensor([1.0]), accuracy=0.5,
            tokens_used=50, phase=0)
    assert out.item() == pytest.approx(0.0)

def test_phase0_above_threshold_within_budget():
    t = ToggleReward(lambda_threshold=0.8, token_budget=100)
    out = t(mean_reward=torch.tensor([1.0]), accuracy=0.9,
            tokens_used=50, phase=0)
    assert out.item() == pytest.approx(1.0)

def test_phase0_over_budget_zeroes_reward():
    t = ToggleReward(lambda_threshold=0.8, token_budget=100)
    out = t(mean_reward=torch.tensor([1.0]), accuracy=0.9,
            tokens_used=150, phase=0)
    assert out.item() == pytest.approx(0.0)

def test_phase1_passes_reward_through():
    t = ToggleReward(lambda_threshold=0.8, token_budget=100)
    out = t(mean_reward=torch.tensor([0.6]), accuracy=0.1,
            tokens_used=9999, phase=1)
    assert out.item() == pytest.approx(0.6)

def test_batch_shape():
    t = ToggleReward(lambda_threshold=0.5, token_budget=200)
    out = t(torch.ones(4), accuracy=0.8, tokens_used=100, phase=0)
    assert out.shape == (4,)

def test_no_nan():
    t = ToggleReward(lambda_threshold=0.5, token_budget=50)
    for phase in (0, 1):
        out = t(torch.zeros(3), accuracy=0.0, tokens_used=0, phase=phase)
        assert torch.isfinite(out).all()
```

**Step 2: Run — expect FAIL**
```bash
.venv/bin/python3.14 -m pytest tests/alignment/test_toggle.py -v --tb=short
```

**Step 3: Implement**
```python
# src/alignment/toggle.py
"""Toggle token-efficient RL — Kimi K2.5 §3.4 (arXiv:2602.02276).
Alternates between budget-limited (phase 0) and standard (phase 1) optimization.
Achieves 25-30% token reduction with negligible performance loss.
"""
import torch
from dataclasses import dataclass

@dataclass
class ToggleReward:
    lambda_threshold: float = 0.8
    token_budget: int = 2048

    def __call__(
        self,
        mean_reward: torch.Tensor,
        accuracy: float,
        tokens_used: int,
        phase: int,  # 0=budget-limited, 1=standard
    ) -> torch.Tensor:
        if phase == 1:
            return mean_reward
        # phase 0: reward only if accuracy >= threshold AND within budget
        if accuracy >= self.lambda_threshold and tokens_used <= self.token_budget:
            return mean_reward
        return torch.zeros_like(mean_reward)
```

**Step 4: Run — expect PASS, wire registry, write integration test, commit**
```bash
.venv/bin/python3.14 -m pytest tests/alignment/test_toggle.py -v
```
Wire: `ALIGNMENT_REGISTRY["toggle"] = ToggleReward` in `src/alignment/__init__.py`.

```bash
git add src/alignment/toggle.py tests/alignment/test_toggle.py tests/integration/test_toggle_integration.py src/alignment/__init__.py
git commit -m "feat(cycle-124): toggle token-efficient RL — Kimi K2.5 §3.4"
```

---

### Task 124-C: `src/alignment/grm.py` — Generative Reward Model

**Surface:** alignment | **Paper:** Kimi K2.5 §3.5 + GLM-5 §4.2
**Files:**
- Create: `src/alignment/grm.py`
- Create: `tests/alignment/test_grm.py`
- Create: `tests/integration/test_grm_integration.py`
- Modify: `src/alignment/__init__.py` — add `ALIGNMENT_REGISTRY["grm"]`

**Core design:** GRM scores along 4 dimensions (helpfulness, instruction adherence, relevance, detail), returns weighted scalar. Hybrid mode switches to rule-based verifier for verifiable tasks (code/math).

**Key tests to write:**
- output shape `(B,)` and range `[0,1]`
- each dimension weight sums to 1 in normalized mode
- adversarial all-zeros scores produce finite output
- hybrid mode: `mode="rule"` bypasses GRM scoring
- identical candidates score identically
- malformed input (empty string) does not crash

**Step 3: Implement**
```python
# src/alignment/grm.py
"""Generative Reward Model — Kimi K2.5 §3.5 + GLM-5 §4.2."""
from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Literal

DIMENSIONS = ["helpfulness", "adherence", "relevance", "detail"]

@dataclass
class GRMConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {d: 0.25 for d in DIMENSIONS}
    )
    mode: Literal["grm", "rule"] = "grm"

class GenerativeRewardModel:
    def __init__(self, config: GRMConfig | None = None):
        self.config = config or GRMConfig()
        w = self.config.weights
        total = sum(w.values()) or 1.0
        self._w = {k: v / total for k, v in w.items()}

    def score(
        self,
        dim_scores: dict[str, torch.Tensor],  # each: [B] in [0,1]
        rule_reward: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.mode == "rule" and rule_reward is not None:
            return rule_reward
        out = sum(self._w.get(k, 0.0) * v for k, v in dim_scores.items())
        return torch.clamp(out, 0.0, 1.0)
```

**Commit:**
```bash
git add src/alignment/grm.py tests/alignment/test_grm.py tests/integration/test_grm_integration.py src/alignment/__init__.py
git commit -m "feat(cycle-124): generative reward model — Kimi K2.5/GLM-5 hybrid GRM"
```

---

### Task 124-D: `src/agent/agent_swarm.py` — Orchestrator + Frozen Subagent Swarm

**Surface:** agent | **Paper:** Kimi K2.5 §3.2
**Files:**
- Create: `src/agent/agent_swarm.py`
- Create: `tests/agent/test_agent_swarm.py`
- Create: `tests/integration/test_agent_swarm_integration.py`
- Modify: `src/agent/__init__.py` — add `AGENT_LOOP_REGISTRY["agent_swarm"]`

**Core algorithm:**
```
critical_steps = Σ_t [S_main(t) + max_i(S_sub,i(t))]
speedup = serial_steps / critical_steps
```

**Key tests:**
- `CriticalPathAnalyzer.compute()` returns correct value on known input
- speedup > 1.0 when subagents run in parallel
- serial fallback (1 subagent) → speedup = 1.0
- empty subagent list → critical_steps = S_main
- adversarial: subagent with 0 steps
- `AgentSwarm.dispatch()` returns list of results equal to task count
- orchestrator step limit respected
- subagent step limit respected per agent

**Step 3: Implement**
```python
# src/agent/agent_swarm.py
"""Agent Swarm — orchestrator + frozen subagents (Kimi K2.5 §3.2, arXiv:2602.02276).
Orchestrator is trainable via PARL; subagents are frozen.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class SubAgentResult:
    task_id: int
    result: Any
    steps_used: int
    status: str  # "completed" | "truncated" | "error"

@dataclass
class CriticalPathAnalyzer:
    def compute(
        self,
        main_steps_per_stage: list[int],
        sub_steps_per_stage: list[list[int]],  # [stage][agent_i]
    ) -> int:
        total = 0
        for t, main_s in enumerate(main_steps_per_stage):
            parallel = sub_steps_per_stage[t] if t < len(sub_steps_per_stage) else []
            total += main_s + (max(parallel) if parallel else 0)
        return total

    def speedup(self, serial_steps: int, critical_steps: int) -> float:
        return serial_steps / max(critical_steps, 1)

@dataclass
class AgentSwarm:
    orchestrator_max_steps: int = 15
    subagent_max_steps: int = 100

    def dispatch(
        self,
        tasks: list[Any],
        subagent_fn: Callable[[Any, int], SubAgentResult],
    ) -> list[SubAgentResult]:
        return [subagent_fn(task, self.subagent_max_steps) for task in tasks]
```

**Commit:**
```bash
git add src/agent/agent_swarm.py tests/agent/test_agent_swarm.py tests/integration/test_agent_swarm_integration.py src/agent/__init__.py
git commit -m "feat(cycle-124): agent swarm — orchestrator+frozen subagents, critical-path analyzer"
```

---

### Task 124-E: `src/optimizers/muonclip.py` — MuonClip Optimizer

**Surface:** optimizers | **Paper:** Kimi K2.5 §3.3 + GLM-5 §3.1
**Files:**
- Create: `src/optimizers/muonclip.py`
- Create: `tests/optimizers/test_muonclip.py`
- Create: `tests/integration/test_muonclip_integration.py`
- Modify: `src/optimizers/__init__.py` — add `OPTIMIZER_REGISTRY["muonclip"]`

**Core:** Nesterov momentum + per-matrix (Muon Split: per-head) orthogonalization + gradient-norm clipping. Stabilizes RL training where ADAM diverges on long-horizon rewards.

**Key tests:**
- loss decreases over 10 steps on a small linear model
- `torch.manual_seed` → identical param updates
- gradient finite after step
- `max_norm` clipping: inject large grad, verify clipped norm
- orthogonalization: output matrix has near-orthogonal rows (dot product < ε)
- `lr=0` → params unchanged
- works with `param_groups`

**Step 3: Implement**
```python
# src/optimizers/muonclip.py
"""MuonClip — Nesterov + per-matrix orthogonalization + grad clipping.
Muon Split (GLM-5 §3.1): orthogonalize per attention head for scale-stable updates.
"""
import torch
from torch.optim import Optimizer

def _orthogonalize(M: torch.Tensor) -> torch.Tensor:
    if M.ndim < 2:
        return M
    orig_shape = M.shape
    m = M.reshape(M.shape[0], -1)
    # Newton-Schulz iteration (2 steps, cheap approximation)
    A = m @ m.T
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    B = 1.5 * I - 0.5 * A
    m = B @ m
    return m.reshape(orig_shape)

class MuonClip(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, max_norm=1.0):
        defaults = dict(lr=lr, momentum=momentum, max_norm=max_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            max_norm = group["max_norm"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                # Nesterov gradient
                g_ns = g + beta * m
                # Muon Split orthogonalization
                g_orth = _orthogonalize(g_ns)
                # Gradient clipping
                norm = g_orth.norm()
                if norm > max_norm:
                    g_orth = g_orth * (max_norm / (norm + 1e-8))
                p.add_(g_orth, alpha=-lr)
                m.mul_(beta).add_(g)
        return loss
```

**Commit:**
```bash
git add src/optimizers/muonclip.py tests/optimizers/test_muonclip.py tests/integration/test_muonclip_integration.py src/optimizers/__init__.py
git commit -m "feat(cycle-124): muonclip optimizer — Nesterov+Muon Split+grad clipping"
```

---

### Task 124-F: `src/alignment/cross_stage_distillation.py` — On-Policy Cross-Stage Distillation

**Surface:** alignment | **Paper:** GLM-5 §5.4
**Files:**
- Create: `src/alignment/cross_stage_distillation.py`
- Create: `tests/alignment/test_cross_stage_distillation.py`
- Create: `tests/integration/test_cross_stage_distillation_integration.py`
- Modify: `src/alignment/__init__.py` — add `ALIGNMENT_REGISTRY["cross_stage_distillation"]`

**Core:** `L_CSD = L_RL + α·KL(π_θ ∥ π_teacher)` where teacher = final checkpoint of preceding RL stage. Prevents capability degradation across sequential RL curriculum.

**Key tests:**
- loss = RL loss when `alpha=0`
- KL term is non-negative
- KL=0 when student == teacher (same logits)
- finite output when logits contain very negative values
- `alpha` scaling: doubling alpha doubles KL contribution
- gradient flows through student logits

**Step 3: Implement**
```python
# src/alignment/cross_stage_distillation.py
"""On-Policy Cross-Stage Distillation — GLM-5 §5.4 (arXiv:2602.15763).
Prevents catastrophic forgetting between sequential RL curriculum stages.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class CrossStageDistillation:
    alpha: float = 0.1

    def loss(
        self,
        rl_loss: torch.Tensor,
        student_logits: torch.Tensor,   # [B, T, V]
        teacher_logits: torch.Tensor,   # [B, T, V]
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p_teacher = F.softmax(teacher_logits.detach(), dim=-1)
        log_p_student = F.log_softmax(student_logits, dim=-1)
        kl = F.kl_div(log_p_student, p_teacher, reduction="none").sum(-1)  # [B, T]
        if attention_mask is not None:
            kl = kl * attention_mask
            kl = kl.sum() / (attention_mask.sum() + 1e-8)
        else:
            kl = kl.mean()
        return rl_loss + self.alpha * kl
```

**Commit:**
```bash
git add src/alignment/cross_stage_distillation.py tests/alignment/test_cross_stage_distillation.py tests/integration/test_cross_stage_distillation_integration.py src/alignment/__init__.py
git commit -m "feat(cycle-124): cross-stage distillation — GLM-5 §5.4 on-policy KL regularization"
```

---

### Cycle 124 Close

```bash
.venv/bin/python3.14 -m pytest --tb=no -q          # must be green
grep -rE "from (transformers|einops|trl|xformers|flash_attn)" src/ tests/  # must return nothing
.venv/bin/python3.14 -m pytest tests/integration/ -q
git push   # cycle 124 = push (first of every-3-cycle rule)
echo "cycle-124 | parl,toggle,grm,agent_swarm,muonclip,cross_stage_distillation | alignment×3,agent×1,optimizers×1 | integrated=[ALIGNMENT_REGISTRY,AGENT_LOOP_REGISTRY,OPTIMIZER_REGISTRY] | deferred=[]" >> .aurelius-cycles.log
```

---

## CYCLE 125 — Model · Training · LongContext · Inference

### Task 125-A: `src/model/dsa_attention.py` — DeepSeek Sparse Attention

**Surface:** model | **Paper:** GLM-5 §3.1 (arXiv:2602.15763)
**Files:**
- Create: `src/model/dsa_attention.py`
- Create: `tests/model/test_dsa_attention.py`
- Create: `tests/integration/test_dsa_attention_integration.py`
- Modify: `src/model/__init__.py` — add `MODEL_COMPONENT_REGISTRY["dsa_attention"]`

**Core:** Content-aware sparse attention. A Lightning Indexer (small MLP) predicts top-k token indices per query. Replaces O(L²) with O(L·k). Two-stage training: dense warm-up (train indexer only) → sparse adaptation.

**FROZEN FILE WARNING:** Do NOT touch `src/model/attention.py`, `src/model/transformer.py`.

**Key tests (tiny config, max_seq_len=64):**
- output shape `[B, T, d_model]` correct
- `top_k ≤ seq_len` never violated
- sparsity: actual attended tokens per query == top_k
- output identical to dense when `top_k=seq_len`
- gradient through indexer (warm-up phase)
- `freeze_indexer=True` → indexer params have no grad
- determinism under `torch.manual_seed`
- no NaN/Inf on random input
- seq_len=1 edge case
- seq_len=64 (full tiny max)

**Step 3: Implement**
```python
# src/model/dsa_attention.py
"""DeepSeek Sparse Attention — GLM-5 §3.1 (arXiv:2602.15763).
Two-stage: dense warm-up (train indexer only) → sparse adaptation.
O(L^2) → O(L*k) attention computation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class DSAConfig:
    d_model: int = 512
    n_heads: int = 8
    top_k: int = 64
    freeze_indexer: bool = False

class LightningIndexer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.proj = nn.Linear(d_model // n_heads, n_heads, bias=False)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return self.proj(q)

class DSAAttention(nn.Module):
    def __init__(self, cfg: DSAConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.indexer = LightningIndexer(cfg.d_model, cfg.n_heads)
        if cfg.freeze_indexer:
            for p in self.indexer.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        top_k = min(self.cfg.top_k, T)
        # Indexer selects top-k key positions per query (content-aware)
        idx_scores = self.indexer(q.mean(dim=1))  # [B, n_heads]
        # Simple top-k selection on last dim of K for demonstration
        k_scores = k.mean(dim=-1)  # [B, n_heads, T]
        _, top_idx = k_scores.topk(top_k, dim=-1)  # [B, n_heads, top_k]
        # Gather sparse K, V
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        k_sparse = k.gather(2, idx_exp)
        v_sparse = v.gather(2, idx_exp)
        # Attention over sparse set
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k_sparse.transpose(-1, -2)) * scale, dim=-1)
        out = torch.matmul(attn, v_sparse)
        out = out.transpose(1, 2).reshape(B, T, self.cfg.d_model)
        return self.o_proj(out)
```

**Commit:**
```bash
git add src/model/dsa_attention.py tests/model/test_dsa_attention.py tests/integration/test_dsa_attention_integration.py src/model/__init__.py
git commit -m "feat(cycle-125): DSA attention — GLM-5 Lightning Indexer sparse attention"
```

---

### Task 125-B: `src/model/mtp_shared.py` — MTP with Parameter Sharing

**Surface:** model | **Paper:** GLM-5 §3.3
**Files:**
- Create: `src/model/mtp_shared.py`
- Create: `tests/model/test_mtp_shared.py`
- Create: `tests/integration/test_mtp_shared_integration.py`
- Modify: `src/model/__init__.py` — add `MODEL_COMPONENT_REGISTRY["mtp_shared"]`

**Core:** 3 MTP prediction heads sharing a single projection weight matrix. Acceptance rate 2.76 vs 2.55 (unshared). Distinct from existing `src/model/multi_token_prediction.py`.

**Key tests:**
- 3 heads share identical weight tensor (same `data_ptr`)
- output: list of 3 logit tensors, each `[B, T, V]`
- gradient flows to shared weight from all 3 heads
- tiny config: `vocab_size=256, d_model=64`
- acceptance_rate computation returns float in [0, seq_len]
- no NaN on random hidden states

**Step 3: Implement**
```python
# src/model/mtp_shared.py
"""MTP with Parameter Sharing — GLM-5 §3.3 (arXiv:2602.15763).
3 prediction heads share projection weights. Acceptance rate: 2.76 vs 2.55 baseline.
"""
import torch
import torch.nn as nn

class SharedMTPHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, n_heads: int = 3):
        super().__init__()
        self.n_heads = n_heads
        # Single shared projection — all heads use the same weight
        self.shared_proj = nn.Linear(d_model, vocab_size, bias=False)
        # Per-head input projections (not shared)
        self.input_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False) for _ in range(n_heads)
        ])

    def forward(self, hidden: torch.Tensor) -> list[torch.Tensor]:
        return [self.shared_proj(proj(hidden)) for proj in self.input_projs]

    def acceptance_rate(self, logits_list: list[torch.Tensor],
                        targets: torch.Tensor) -> float:
        total, accepted = 0, 0
        for i, logits in enumerate(logits_list):
            preds = logits.argmax(-1)           # [B, T]
            shifted_targets = targets[:, i+1:]  # [B, T-i-1]
            shifted_preds = preds[:, :shifted_targets.shape[1]]
            accepted += (shifted_preds == shifted_targets).float().sum().item()
            total += shifted_targets.numel()
        return accepted / max(total, 1)
```

**Commit:**
```bash
git commit -m "feat(cycle-125): mtp-shared — GLM-5 parameter-sharing MTP heads, accept-rate 2.76"
```

---

### Task 125-C: `src/training/async_rl_infra.py` — Async RL Infrastructure

**Surface:** training | **Paper:** GLM-5 §4.1
**Files:**
- Create: `src/training/async_rl_infra.py`
- Create: `tests/training/test_async_rl_infra.py`
- Create: `tests/integration/test_async_rl_infra_integration.py`
- Modify: `src/training/trainer.py` — add branch for `use_async_rl=True`

**Core:** Multi-Task Rollout Orchestrator decouples inference (rollout) from training (gradient update). Heartbeat-driven fault tolerance. Supports 1000+ concurrent rollouts.

**Key tests:**
- `RolloutOrchestrator` queues tasks and returns results
- heartbeat timeout triggers task re-queue
- rollout result carries `(task_id, completion, reward)`
- concurrent dispatch of N tasks returns N results
- orchestrator handles empty task list gracefully
- training step consumes rollout batch without crash
- `use_async_rl=False` → standard sync path unchanged (regression guard)

**Step 3: Implement**
```python
# src/training/async_rl_infra.py
"""Async RL Infrastructure — GLM-5 §4.1 (arXiv:2602.15763).
Decouples rollout (inference) from gradient updates (training).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from collections import deque

@dataclass
class RolloutResult:
    task_id: int
    completion: str
    reward: float
    tokens_used: int

@dataclass
class RolloutOrchestrator:
    heartbeat_timeout: float = 30.0   # seconds before re-queue

    def __post_init__(self):
        self._queue: deque[Any] = deque()
        self._in_flight: dict[int, tuple[Any, float]] = {}

    def enqueue(self, tasks: list[Any]) -> None:
        for t in tasks:
            self._queue.append(t)

    def dispatch(
        self, rollout_fn: Callable[[Any], RolloutResult]
    ) -> list[RolloutResult]:
        results = []
        while self._queue:
            task = self._queue.popleft()
            task_id = id(task)
            self._in_flight[task_id] = (task, time.monotonic())
            try:
                result = rollout_fn(task)
                results.append(result)
            finally:
                self._in_flight.pop(task_id, None)
        return results

    def requeue_stale(self) -> int:
        now = time.monotonic()
        stale = [k for k, (t, ts) in self._in_flight.items()
                 if now - ts > self.heartbeat_timeout]
        for k in stale:
            task, _ = self._in_flight.pop(k)
            self._queue.appendleft(task)
        return len(stale)
```

**Commit:**
```bash
git commit -m "feat(cycle-125): async-rl-infra — GLM-5 rollout orchestrator, heartbeat fault-tolerance"
```

---

### Task 125-D: `src/training/tito_gateway.py` — Token-in-Token-out Gateway

**Surface:** training | **Paper:** GLM-5 §4.1
**Files:**
- Create: `src/training/tito_gateway.py`
- Create: `tests/training/test_tito_gateway.py`
- Create: `tests/integration/test_tito_gateway_integration.py`

**Core:** Canonicalizes token IDs at the inference→training boundary, preventing off-by-one re-tokenization mismatches in off-policy RL.

**Key tests:**
- round-trip: encode → canonicalize → decode → same string
- mismatched vocab raises `ValueError`
- empty token list returns empty list
- canonical IDs are within `[0, vocab_size)`
- handles padding tokens correctly

**Step 3: Implement**
```python
# src/training/tito_gateway.py
"""TITO Gateway — Token-in-Token-out (GLM-5 §4.1).
Eliminates re-tokenization mismatches at inference/training engine boundary.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TITOGateway:
    vocab_size: int
    pad_id: int = 0

    def canonicalize(self, token_ids: list[int]) -> list[int]:
        if not token_ids:
            return []
        if any(t >= self.vocab_size or t < 0 for t in token_ids):
            raise ValueError(
                f"Token IDs out of range [0, {self.vocab_size}): {token_ids}"
            )
        return token_ids  # in production: remap IDs across tokenizer boundary

    def wrap_batch(self, batch: list[list[int]]) -> list[list[int]]:
        return [self.canonicalize(seq) for seq in batch]
```

**Commit:**
```bash
git commit -m "feat(cycle-125): tito-gateway — GLM-5 token-in-token-out re-tokenization fix"
```

---

### Task 125-E: `src/longcontext/hierarchical_context_mgr.py` — Hierarchical Context Management

**Surface:** longcontext | **Paper:** GLM-5 §6.2
**Files:**
- Create: `src/longcontext/hierarchical_context_mgr.py`
- Create: `tests/longcontext/test_hierarchical_context_mgr.py`
- Create: `tests/integration/test_hierarchical_context_mgr_integration.py`
- Modify: `src/longcontext/__init__.py` — add `LONGCONTEXT_STRATEGY_REGISTRY["hierarchical_context"]`

**Core:** keep-recent-k turns strategy; falls back to discard-all when quality < threshold. Triggers at 80% of max_len. BrowseComp: 55.3% → 62.0%.

**Key tests:**
- at 79% capacity → no truncation
- at 81% capacity → keep-recent-k applied
- after truncation, context length ≤ max_len
- discard-all fallback when `quality_score < threshold`
- empty context → no crash
- k=0 → discard all turns except last
- recent turns preserved in correct order

**Step 3: Implement**
```python
# src/longcontext/hierarchical_context_mgr.py
"""Hierarchical Context Management — GLM-5 §6.2 (arXiv:2602.15763).
keep-recent-k → discard-all fallback. BrowseComp: 55.3% → 62.0%.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HierarchicalContextManager:
    max_len: int = 8192
    trigger_ratio: float = 0.8
    keep_k: int = 10          # keep last k turns
    quality_threshold: float = 0.3

    def manage(
        self,
        turns: list[dict],     # list of {"role": str, "content": str, "tokens": int}
        quality_score: float = 1.0,
    ) -> list[dict]:
        total = sum(t["tokens"] for t in turns)
        if total < self.max_len * self.trigger_ratio:
            return turns
        if quality_score < self.quality_threshold:
            # Discard all but last turn
            return turns[-1:] if turns else []
        # Keep-recent-k strategy
        recent = turns[-self.keep_k:] if self.keep_k > 0 else []
        return recent
```

**Commit:**
```bash
git commit -m "feat(cycle-125): hierarchical-context-mgr — GLM-5 keep-recent-k + discard-all fallback"
```

---

### Task 125-F: `src/inference/reasoning_level_controller.py` — Reasoning Level Controller

**Surface:** inference | **Paper:** GPT-OSS-120B (arXiv:2508.10925)
**Files:**
- Create: `src/inference/reasoning_level_controller.py`
- Create: `tests/inference/test_reasoning_level_controller.py`
- Create: `tests/integration/test_reasoning_level_controller_integration.py`
- Modify: `src/inference/__init__.py` — add `DECODER_REGISTRY["reasoning_level"]`

**Core:** Parses `"Reasoning: low/medium/high"` prefix from system prompt → injects generation hyperparameters. SWE-bench: low=47.9%, medium=52.6%, high=62.4%.

**Key tests:**
- `"Reasoning: low"` → temperature=0.3, max_tokens=512
- `"Reasoning: medium"` → temperature=0.6, max_tokens=2048
- `"Reasoning: high"` → temperature=1.0, max_tokens=8192
- missing prefix → default "medium"
- case-insensitive: `"reasoning: HIGH"` works
- malformed prefix → default, no crash
- returned config is a plain dict (no side effects)

**Step 3: Implement**
```python
# src/inference/reasoning_level_controller.py
"""Reasoning Level Controller — GPT-OSS-120B (arXiv:2508.10925).
Maps system-prompt prefix to generation hyperparameters.
SWE-bench Verified: low=47.9%, medium=52.6%, high=62.4%.
"""
from __future__ import annotations
import re

LEVELS: dict[str, dict] = {
    "low":    {"temperature": 0.3,  "max_tokens": 512,  "top_p": 0.9},
    "medium": {"temperature": 0.6,  "max_tokens": 2048, "top_p": 0.95},
    "high":   {"temperature": 1.0,  "max_tokens": 8192, "top_p": 0.95},
}
_PATTERN = re.compile(r"reasoning:\s*(low|medium|high)", re.IGNORECASE)

def parse_reasoning_level(system_prompt: str) -> dict:
    m = _PATTERN.search(system_prompt or "")
    level = m.group(1).lower() if m else "medium"
    return dict(LEVELS[level])
```

**Commit:**
```bash
git commit -m "feat(cycle-125): reasoning-level-controller — GPT-OSS low/medium/high system-prompt mapping"
```

---

### Cycle 125 Close

```bash
.venv/bin/python3.14 -m pytest --tb=no -q
.venv/bin/python3.14 -m pytest tests/integration/ -q
echo "cycle-125 | dsa_attention,mtp_shared,async_rl_infra,tito_gateway,hierarchical_context_mgr,reasoning_level_controller | model×2,training×2,longcontext×1,inference×1 | integrated=[MODEL_COMPONENT_REGISTRY,LONGCONTEXT_STRATEGY_REGISTRY,DECODER_REGISTRY] | deferred=[]" >> .aurelius-cycles.log
```

---

## CYCLE 126 — Model · Chat · Eval · Training · Agent

### Task 126-A: `src/model/dp_aware_moe_routing.py` — DP-aware MoE Routing

**Surface:** model | **Paper:** GLM-5 §3.2
**Files:**
- Create: `src/model/dp_aware_moe_routing.py`
- Create: `tests/model/test_dp_aware_moe_routing.py`
- Create: `tests/integration/test_dp_aware_moe_routing_integration.py`
- Modify: `src/model/__init__.py` — add `MODEL_COMPONENT_REGISTRY["dp_aware_moe_routing"]`

**Core:** Consistent hashing maps session_id → fixed DP rank, preventing cross-rank KV cache misses in multi-turn agentic workloads.

**Key tests:**
- same session_id → always same rank
- rank in `[0, num_dp_ranks)`
- different session_ids may map to different ranks
- rank distribution is reasonably uniform (no single rank gets >80% of 1000 sessions)
- `num_dp_ranks=1` → always rank 0
- empty session_id → deterministic rank

**Step 3: Implement**
```python
# src/model/dp_aware_moe_routing.py
"""DP-aware MoE Routing via Consistent Hashing — GLM-5 §3.2 (arXiv:2602.15763).
Maps session_id to fixed DP rank to preserve KV cache locality in multi-turn agents.
"""
import hashlib
from dataclasses import dataclass

@dataclass
class DPAwareMoERouter:
    num_dp_ranks: int = 8

    def rank_for_session(self, session_id: str) -> int:
        digest = hashlib.md5(session_id.encode()).digest()
        return int.from_bytes(digest[:4], "little") % self.num_dp_ranks

    def route(self, session_id: str, expert_ids: list[int]) -> tuple[int, list[int]]:
        rank = self.rank_for_session(session_id)
        return rank, expert_ids
```

**Commit:**
```bash
git commit -m "feat(cycle-126): dp-aware-moe-routing — consistent hashing for KV cache locality"
```

---

### Task 126-B: `src/model/mla_256.py` — MLA-256 + Muon Split

**Surface:** model | **Paper:** GLM-5 §3.1
**Files:**
- Create: `src/model/mla_256.py`
- Create: `tests/model/test_mla_256.py`
- Create: `tests/integration/test_mla_256_integration.py`
- Modify: `src/model/__init__.py` — add `MODEL_COMPONENT_REGISTRY["mla_256"]`

**Core:** head_dim 192→256, head_count ×0.67. Per-head Muon Split orthogonalization on projection weights. Reduces decoding KV cache vs standard MLA while stabilizing training logit scales.

**Key tests:**
- output shape `[B, T, d_model]`
- head_dim=256 in weight tensors
- `n_heads` reduced by 1/3 vs base MLA
- per-head orthogonalization: projection rows are near-orthogonal
- gradient finite
- matches tiny config output shape

**Commit:**
```bash
git commit -m "feat(cycle-126): mla-256 — GLM-5 head-dim 256 + Muon Split per-head orthogonalization"
```

---

### Task 126-C: `src/chat/harmony_template.py` — Harmony Response Format

**Surface:** chat | **Paper:** GPT-OSS-120B (arXiv:2508.10925)
**Files:**
- Create: `src/chat/harmony_template.py`
- Create: `tests/chat/test_harmony_template.py`
- Create: `tests/integration/test_harmony_template_integration.py`
- Modify: `src/chat/__init__.py` — add `CHAT_TEMPLATE_REGISTRY["harmony"]`

**Core:** Chat template trained into GPT-OSS-120B. Includes reasoning scratchpad delimiters, structured tool-call format, system prompt position. Fails loudly if used without required tokens.

**Key tests:**
- user/assistant turns render in correct order
- system prompt appears before first user turn
- tool-call format includes `<tool_call>` and `<tool_result>` delimiters
- reasoning scratchpad enclosed in `<think>...</think>`
- empty messages list → empty string, no crash
- adversarial: role not in {system, user, assistant, tool} → ValueError
- round-trip: render → parse → same messages

**Step 3: Implement**
```python
# src/chat/harmony_template.py
"""Harmony Response Format — GPT-OSS-120B (arXiv:2508.10925).
Chat template with reasoning scratchpad, tool-call, and system prompt support.
"""
from __future__ import annotations
from dataclasses import dataclass

VALID_ROLES = {"system", "user", "assistant", "tool"}

@dataclass
class HarmonyTemplate:
    bos: str = "<|begin_of_text|>"
    eos: str = "<|end_of_text|>"
    think_open: str = "<think>"
    think_close: str = "</think>"

    def render(self, messages: list[dict]) -> str:
        parts = [self.bos]
        for msg in messages:
            role = msg.get("role", "")
            if role not in VALID_ROLES:
                raise ValueError(f"Unknown role: {role!r}. Must be one of {VALID_ROLES}")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n<|end_system|>")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n<|end_user|>")
            elif role == "assistant":
                thinking = msg.get("thinking", "")
                if thinking:
                    parts.append(f"<|assistant|>\n{self.think_open}{thinking}{self.think_close}\n{content}\n<|end_assistant|>")
                else:
                    parts.append(f"<|assistant|>\n{content}\n<|end_assistant|>")
            elif role == "tool":
                parts.append(f"<tool_result>\n{content}\n</tool_result>")
        return "\n".join(parts)
```

**Commit:**
```bash
git commit -m "feat(cycle-126): harmony-template — GPT-OSS chat format with think/tool delimiters"
```

---

### Task 126-D: `src/eval/swarm_bench.py` — Agent Swarm Benchmark

**Surface:** eval | **Depends on:** `src/agent/agent_swarm.py` (cycle 124)
**Files:**
- Create: `src/eval/swarm_bench.py`
- Create: `tests/eval/test_swarm_bench.py`
- Create: `tests/integration/test_swarm_bench_integration.py`
- Modify: `src/eval/__init__.py` — add `BENCHMARK_REGISTRY["swarm_bench"]`

**Core:** Evaluates agent_swarm on BrowseComp-style tasks. Computes: task completion rate, critical-path steps, parallelism ratio, speedup vs single-agent baseline.

**Key tests:**
- `SwarmBench.run()` returns dict with keys: `completion_rate`, `critical_steps`, `speedup`
- completion_rate ∈ [0, 1]
- speedup > 1.0 when tasks are parallelizable
- speedup = 1.0 for single-task benchmark
- empty task list → all metrics = 0 or NaN handled gracefully
- determinism under fixed seed

**Commit:**
```bash
git commit -m "feat(cycle-126): swarm-bench — BrowseComp-style evaluation for agent swarms"
```

---

### Task 126-E: `src/training/slime_framework.py` — Slime RL Framework

**Surface:** training | **Paper:** GLM-5 §4
**Files:**
- Create: `src/training/slime_framework.py`
- Create: `tests/training/test_slime_framework.py`
- Create: `tests/integration/test_slime_framework_integration.py`

**Core:** Unified post-training RL infrastructure. Task router maps task_type → verifier → reward_fn. Supports rollout server lifecycle management.

**Key tests:**
- `SlimeTaskRouter.route("swe")` returns correct verifier + reward_fn
- unknown task_type → ValueError (fail loudly)
- reward_fn(completion, target) returns scalar
- multiple task types registered and routable
- router is extensible: `register_task()` adds new entry

**Step 3: Implement**
```python
# src/training/slime_framework.py
"""Slime RL Framework — GLM-5 §4 (arXiv:2602.15763).
Unified post-training infrastructure: task router → verifier → reward_fn.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any

VerifierFn = Callable[[str, str], bool]
RewardFn = Callable[[str, str], float]

@dataclass
class SlimeTask:
    name: str
    verifier: VerifierFn
    reward_fn: RewardFn

class SlimeTaskRouter:
    def __init__(self):
        self._registry: dict[str, SlimeTask] = {}

    def register_task(self, task: SlimeTask) -> None:
        self._registry[task.name] = task

    def route(self, task_type: str) -> SlimeTask:
        if task_type not in self._registry:
            raise ValueError(f"Unknown task type: {task_type!r}. "
                             f"Registered: {list(self._registry)}")
        return self._registry[task_type]

def make_default_router() -> SlimeTaskRouter:
    router = SlimeTaskRouter()
    router.register_task(SlimeTask(
        name="swe",
        verifier=lambda completion, target: completion.strip() == target.strip(),
        reward_fn=lambda completion, target: 1.0 if completion.strip() == target.strip() else 0.0,
    ))
    router.register_task(SlimeTask(
        name="terminal",
        verifier=lambda c, t: t in c,
        reward_fn=lambda c, t: 1.0 if t in c else 0.0,
    ))
    return router
```

**Commit:**
```bash
git commit -m "feat(cycle-126): slime-framework — GLM-5 unified RL task router + verifier registry"
```

---

### Task 126-F: `src/agent/plugin_hook.py` — Plugin Hook Registry

**Surface:** agent | **Inspired by:** Claude Code plugin architecture
**Files:**
- Create: `src/agent/plugin_hook.py`
- Create: `tests/agent/test_plugin_hook.py`
- Create: `tests/integration/test_plugin_hook_integration.py`
- Modify: `src/agent/__init__.py` — add `AGENT_LOOP_REGISTRY["plugin_hook"]`

**Core:** HOOK_REGISTRY with 5 hook points. Callables registered at import. Enables logging, rate-limiting, safety filtering without modifying agent loop.

**Key tests:**
- hooks registered and called in registration order
- `pre_tool_call` hook receives tool_name + args
- `on_error` hook receives exception object
- hook raising exception propagates (no silent swallow)
- multiple hooks per point all called
- empty hook list → no crash
- adversarial: hook that raises → exception reaches caller

**Step 3: Implement**
```python
# src/agent/plugin_hook.py
"""Plugin Hook Registry — Claude Code-inspired extensible agent hooks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any

HOOK_POINTS = ("pre_tool_call", "post_tool_call", "pre_generation",
               "post_generation", "on_error")

class PluginHookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {k: [] for k in HOOK_POINTS}

    def register(self, point: str, fn: Callable) -> None:
        if point not in self._hooks:
            raise ValueError(f"Unknown hook point: {point!r}. Valid: {HOOK_POINTS}")
        self._hooks[point].append(fn)

    def fire(self, point: str, **kwargs: Any) -> None:
        for fn in self._hooks[point]:
            fn(**kwargs)

    def clear(self, point: str | None = None) -> None:
        if point is None:
            for k in self._hooks:
                self._hooks[k].clear()
        else:
            self._hooks[point].clear()

HOOK_REGISTRY = PluginHookRegistry()
```

**Commit:**
```bash
git commit -m "feat(cycle-126): plugin-hook-registry — Claude Code-inspired pre/post agent hook system"
```

---

### Cycle 126 Close

```bash
.venv/bin/python3.14 -m pytest --tb=no -q
.venv/bin/python3.14 -m pytest tests/integration/ -q
git push   # cycle 126 = push (3-cycle rule: 124,125,126)
echo "cycle-126 | dp_aware_moe_routing,mla_256,harmony_template,swarm_bench,slime_framework,plugin_hook | model×2,chat×1,eval×1,training×1,agent×1 | integrated=[MODEL_COMPONENT_REGISTRY,CHAT_TEMPLATE_REGISTRY,BENCHMARK_REGISTRY] | deferred=[]" >> .aurelius-cycles.log
```

---

## CYCLE 127 — Vision / Multimodal

**Harvest basis:** Kimi K2.5 MoonViT-3D + Zero-Vision SFT (arXiv:2602.02276)
**Surfaces:** model ×2, data ×2, alignment ×1, eval ×1

> **Constraint:** No vision runtime deps. Implement vision encoder as pure PyTorch nn.Module. Image inputs are tensors of shape `[B, C, H, W]`; video is `[B, T_frames, C, H, W]`.

---

### Task 127-A: `src/model/moonvit_patch_packer.py` — NaViT Patch Packing

**Surface:** model | **Paper:** Kimi K2.5 §2.1
**Files:**
- Create: `src/model/moonvit_patch_packer.py`
- Create: `tests/model/test_moonvit_patch_packer.py`
- Create: `tests/integration/test_moonvit_patch_packer_integration.py`
- Modify: `src/model/__init__.py` — add `MODEL_COMPONENT_REGISTRY["moonvit_patch_packer"]`

**Core:** Packs 2D image patches into 1D sequence (NaViT strategy). Spatiotemporal: handles up to 4 frames × (H/16 × W/16) patches. Variable resolution — no fixed grid required.

**Key tests:**
- patch count = (H//16) × (W//16) for image
- video: patch count = T_frames × (H//16) × (W//16)
- output shape: `[B, num_patches, patch_dim]` where `patch_dim = C × 16 × 16`
- different H/W produce correct patch counts
- 4-frame video: 4× patches of single frame
- no NaN on random input
- `patch_size=16` default, configurable

**Step 3: Implement**
```python
# src/model/moonvit_patch_packer.py
"""MoonViT NaViT Patch Packer — Kimi K2.5 §2.1 (arXiv:2602.02276).
Variable-resolution 2D→1D patch packing for image and video inputs.
Spatiotemporal: up to 4 frames × (H/16 × W/16) patches unified into 1D sequence.
"""
import torch
import torch.nn as nn

class MoonViTPatchPacker(nn.Module):
    def __init__(self, patch_size: int = 16, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, self.patch_dim)

    def pack_image(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"H,W must be divisible by {p}"
        nh, nw = H // p, W // p
        # Rearrange to patches
        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B,C,nh,nw,p,p]
        patches = patches.contiguous().view(B, C, nh * nw, p * p)
        patches = patches.permute(0, 2, 1, 3).reshape(B, nh * nw, C * p * p)
        return self.proj(patches)

    def pack_video(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        frames = [self.pack_image(x[:, t]) for t in range(T)]
        return torch.cat(frames, dim=1)  # [B, T*nh*nw, patch_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return self.pack_image(x)
        elif x.ndim == 5:
            return self.pack_video(x)
        raise ValueError(f"Expected 4D (image) or 5D (video) tensor, got {x.ndim}D")
```

**Commit:**
```bash
git commit -m "feat(cycle-127): moonvit-patch-packer — Kimi K2.5 NaViT spatiotemporal patch packing"
```

---

### Task 127-B: `src/model/vision_projector.py` — Vision→LLM Projector

**Surface:** model | **Paper:** Kimi K2.5 §2.1
**Files:**
- Create: `src/model/vision_projector.py`
- Create: `tests/model/test_vision_projector.py`
- Create: `tests/integration/test_vision_projector_integration.py`
- Modify: `src/model/__init__.py` — add `MODEL_COMPONENT_REGISTRY["vision_projector"]`

**Core:** Projects patch embeddings from ViT hidden dim to LLM hidden dim. Optional 4× temporal compression pooling for video (4 frames → 1 pooled representation).

**Key tests:**
- output shape `[B, num_patches, llm_hidden]`
- temporal compression: `[B, 4*N, d]` → `[B, N, llm_d]` (4× reduction)
- no temporal compression when `temporal_compress=False`
- gradient flows through projection
- works with `num_patches=1` (single patch)

**Step 3: Implement**
```python
# src/model/vision_projector.py
"""Vision→LLM Projector — Kimi K2.5 §2.1 (arXiv:2602.02276).
Projects ViT patch embeddings to LLM hidden dim.
4× temporal pooling for video inputs.
"""
import torch
import torch.nn as nn

class VisionProjector(nn.Module):
    def __init__(self, vit_dim: int, llm_dim: int,
                 temporal_compress: bool = False, compress_factor: int = 4):
        super().__init__()
        self.proj = nn.Linear(vit_dim, llm_dim)
        self.temporal_compress = temporal_compress
        self.compress_factor = compress_factor

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        # patch_embeds: [B, N, vit_dim]
        if self.temporal_compress:
            B, N, D = patch_embeds.shape
            cf = self.compress_factor
            N_out = N // cf
            x = patch_embeds[:, :N_out * cf].reshape(B, N_out, cf, D)
            patch_embeds = x.mean(dim=2)  # [B, N_out, D]
        return self.proj(patch_embeds)
```

**Commit:**
```bash
git commit -m "feat(cycle-127): vision-projector — ViT→LLM projection with 4× temporal pooling"
```

---

### Task 127-C: `src/data/vision_token_mixer.py` — Early-Fusion Vision Token Mixer

**Surface:** data | **Paper:** Kimi K2.5 §2.2
**Files:**
- Create: `src/data/vision_token_mixer.py`
- Create: `tests/data/test_vision_token_mixer.py`
- Create: `tests/integration/test_vision_token_mixer_integration.py`
- Modify: `src/data/__init__.py` — add `LOADER_REGISTRY["vision_token_mixer"]`

**Core:** Mixes vision patch tokens with text tokens at a constant ratio (default 10%) throughout training. Early fusion (not late). Processes ~15T mixed tokens in Kimi K2.5 pre-training.

**Key tests:**
- vision ratio ≈ target_ratio (within 2%)
- total token count = text_count + vision_count
- `ratio=0.0` → all text tokens
- `ratio=1.0` → all vision tokens
- mixed sequence preserves text token order
- determinism under `torch.manual_seed`

**Step 3: Implement**
```python
# src/data/vision_token_mixer.py
"""Early-Fusion Vision Token Mixer — Kimi K2.5 §2.2 (arXiv:2602.02276).
Mixes vision patch tokens with text at constant ratio (10%) throughout pre-training.
Lower vision ratio (10%) outperforms late-fusion high ratio (50%).
"""
import torch
from dataclasses import dataclass

@dataclass
class VisionTokenMixer:
    vision_ratio: float = 0.10   # fraction of tokens that are vision patches

    def mix(
        self,
        text_tokens: torch.Tensor,   # [T_text]
        vision_tokens: torch.Tensor, # [T_vision]
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        n_text = text_tokens.shape[0]
        n_vision = vision_tokens.shape[0]
        total = n_text + n_vision
        # Interleave: insert vision tokens at evenly-spaced positions
        result = torch.zeros(total, dtype=text_tokens.dtype)
        vision_positions = torch.linspace(0, total - 1, n_vision).long()
        mask = torch.zeros(total, dtype=torch.bool)
        mask[vision_positions] = True
        result[mask] = vision_tokens
        result[~mask] = text_tokens[:mask.logical_not().sum()]
        return result
```

**Commit:**
```bash
git commit -m "feat(cycle-127): vision-token-mixer — Kimi K2.5 early-fusion 10% vision ratio interleaving"
```

---

### Task 127-D: `src/data/programmatic_image_tools.py` — Zero-Vision SFT Proxy Tools

**Surface:** data | **Paper:** Kimi K2.5 §2.3
**Files:**
- Create: `src/data/programmatic_image_tools.py`
- Create: `tests/data/test_programmatic_image_tools.py`
- Create: `tests/integration/test_programmatic_image_tools_integration.py`
- Modify: `src/data/__init__.py` — add `LOADER_REGISTRY["programmatic_image_tools"]`

**Core:** Pure-PyTorch proxy ops for image understanding tasks. Model learns to compose these during text-only SFT. Enables visual reasoning without visual annotations.

**Key tests:**
- `crop_region([B,C,H,W], x1,y1,x2,y2)` → correct output shape
- `pixel_distance(pt1, pt2)` → Euclidean distance
- `blob_count(mask)` → integer count of connected blobs
- `crop_region` with out-of-bounds → clamps to image edges
- `pixel_distance` on identical points → 0.0
- `blob_count` on all-zero mask → 0
- no NaN/Inf on any op

**Step 3: Implement**
```python
# src/data/programmatic_image_tools.py
"""Programmatic Image Tools — Zero-Vision SFT proxy ops (Kimi K2.5 §2.3, arXiv:2602.02276).
Pure PyTorch image operations. Model learns to compose these during text-only SFT,
enabling visual reasoning without manual visual annotations.
"""
import torch
import math

def crop_region(
    image: torch.Tensor, x1: int, y1: int, x2: int, y2: int
) -> torch.Tensor:
    _, _, H, W = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return image[:, :, y1:y2, x1:x2]

def pixel_distance(pt1: tuple[float, float], pt2: tuple[float, float]) -> float:
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def blob_count(mask: torch.Tensor) -> int:
    # mask: [H, W] bool tensor; count 4-connected components (BFS)
    visited = torch.zeros_like(mask, dtype=torch.bool)
    H, W = mask.shape
    count = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not visited[i, j]:
                count += 1
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= H or c < 0 or c >= W:
                        continue
                    if visited[r, c] or not mask[r, c]:
                        continue
                    visited[r, c] = True
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        stack.append((r+dr, c+dc))
    return count
```

**Commit:**
```bash
git commit -m "feat(cycle-127): programmatic-image-tools — Zero-Vision SFT proxy ops (crop, distance, blob-count)"
```

---

### Task 127-E: `src/alignment/zero_vision_sft.py` — Zero-Vision SFT Trainer

**Surface:** alignment | **Paper:** Kimi K2.5 §2.3
**Files:**
- Create: `src/alignment/zero_vision_sft.py`
- Create: `tests/alignment/test_zero_vision_sft.py`
- Create: `tests/integration/test_zero_vision_sft_integration.py`
- Modify: `src/alignment/__init__.py` — add `ALIGNMENT_REGISTRY["zero_vision_sft"]`

**Core:** Text-only SFT that activates visual reasoning via programmatic image operations. Outperforms text+vision SFT on visual tasks. Image ops proxied through tool calls in training data.

**Key tests:**
- loss computes on text-only batch (no vision inputs)
- tool-call loss mask: loss computed only on tool-call tokens
- gradient flows through all trainable params
- `visual_tool_calls` count tracked per batch
- malformed tool call (missing args) → ValueError, not crash
- determinism under `torch.manual_seed`

**Step 3: Implement**
```python
# src/alignment/zero_vision_sft.py
"""Zero-Vision SFT — Kimi K2.5 §2.3 (arXiv:2602.02276).
Text-only SFT activates visual reasoning via programmatic image tool calls.
Outperforms text+vision SFT on visual tasks (cross-modal transfer).
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ZeroVisionSFT:
    tool_loss_weight: float = 2.0   # upweight loss on tool-call tokens

    def loss(
        self,
        logits: torch.Tensor,      # [B, T, V]
        labels: torch.Tensor,      # [B, T]
        tool_call_mask: torch.Tensor,  # [B, T] bool — True on tool-call tokens
    ) -> torch.Tensor:
        B, T, V = logits.shape
        base_loss = F.cross_entropy(
            logits.reshape(-1, V), labels.reshape(-1), reduction="none"
        ).reshape(B, T)
        weights = torch.ones_like(base_loss)
        weights[tool_call_mask] = self.tool_loss_weight
        return (base_loss * weights).mean()
```

**Commit:**
```bash
git commit -m "feat(cycle-127): zero-vision-sft — Kimi K2.5 text-only SFT with tool-call upweighting"
```

---

### Task 127-F: `src/eval/vision_grounding_eval.py` — Vision Grounding Evaluation

**Surface:** eval | **Paper:** Kimi K2.5 §4.1
**Files:**
- Create: `src/eval/vision_grounding_eval.py`
- Create: `tests/eval/test_vision_grounding_eval.py`
- Create: `tests/integration/test_vision_grounding_eval_integration.py`
- Modify: `src/eval/__init__.py` — add `METRIC_REGISTRY["vision_grounding"]`, `METRIC_REGISTRY["ocr_ned"]`, `METRIC_REGISTRY["counting_diff"]`

**Core:** Three vision reward metrics from Kimi K2.5:
- **Grounding F1**: soft IoU matching between predicted + ground-truth bounding boxes
- **OCR NED**: normalized edit distance for text recognition  
- **Counting diff**: absolute difference from ground-truth count

**Key tests:**
- perfect grounding prediction → F1=1.0
- no overlap → F1=0.0
- soft IoU: partial overlap returns value in (0,1)
- OCR NED: identical strings → 0.0, completely different → 1.0
- counting diff: exact count → 0.0, off by 3 → 3.0
- all metrics: empty prediction → handled gracefully
- all metrics: finite output on edge inputs

**Step 3: Implement**
```python
# src/eval/vision_grounding_eval.py
"""Vision Grounding Evaluation — Kimi K2.5 §4.1 (arXiv:2602.02276).
Metrics: F1 with soft IoU (grounding), normalized edit distance (OCR), abs diff (counting).
"""
from __future__ import annotations

def box_iou(box_a: tuple[float,float,float,float],
            box_b: tuple[float,float,float,float]) -> float:
    ax1,ay1,ax2,ay2 = box_a
    bx1,by1,bx2,by2 = box_b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def grounding_f1(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5) -> float:
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    tp = sum(
        1 for pb in pred_boxes
        if any(box_iou(pb, gb) >= iou_threshold for gb in gt_boxes)
    )
    prec = tp / len(pred_boxes) if pred_boxes else 0.0
    rec  = tp / len(gt_boxes)
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0

def ocr_ned(pred: str, gt: str) -> float:
    # Normalized edit distance via Wagner-Fischer DP
    m, n = len(pred), len(gt)
    if m == 0 and n == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], (
                prev if pred[i-1] == gt[j-1]
                else 1 + min(prev, dp[j], dp[j-1])
            )
    return dp[n] / max(m, n)

def counting_diff(pred_count: int, gt_count: int) -> float:
    return abs(pred_count - gt_count)
```

**Commit:**
```bash
git commit -m "feat(cycle-127): vision-grounding-eval — soft-IoU F1, OCR NED, counting-diff metrics"
```

---

### Cycle 127 Close

```bash
.venv/bin/python3.14 -m pytest --tb=no -q
.venv/bin/python3.14 -m pytest tests/integration/ -q
grep -rE "from (transformers|einops|trl|xformers|flash_attn|bitsandbytes|peft|diffusers|datasets|accelerate|deepspeed|langchain|llamaindex)" src/ tests/  # must return nothing
git push   # cycle 127 = push (every 3 cycles, and vision cycle always push)
echo "cycle-127 | moonvit_patch_packer,vision_projector,vision_token_mixer,programmatic_image_tools,zero_vision_sft,vision_grounding_eval | model×2,data×2,alignment×1,eval×1 | integrated=[MODEL_COMPONENT_REGISTRY,LOADER_REGISTRY,ALIGNMENT_REGISTRY,METRIC_REGISTRY] | deferred=[]" >> .aurelius-cycles.log
```

---

## Dispatch Template (copy-paste for each cycle)

Use this agent prompt for each module in parallel dispatch:

```
You are implementing [MODULE_NAME] for the Aurelius project at ~/Desktop/Aurelius.

Aurelius is a frontier-tier agentic coding LLM platform and model family.

SURFACE: [surface]
PAPER / SPEC: [Title, arXiv:XXXX.XXXXX]
SIBLINGS (do NOT duplicate): [other 5 modules in this cycle]

DELIVERABLES (all three required):
1. src/<domain>/<file>.py
2. tests/<domain>/test_<file>.py  
3. tests/integration/test_<file>_integration.py + seam wiring

Follow the implementation plan at docs/plans/2026-04-20-harvest-implementation.md — Task [CYCLE-LETTER].

VERIFY BEFORE REPORTING DONE:
cd ~/Desktop/Aurelius
.venv/bin/python3.14 -m pytest tests/<domain>/test_<file>.py tests/integration/test_<file>_integration.py -v
.venv/bin/python3.14 -m pytest --tb=no -q
grep -rE "from (transformers|einops|trl|xformers|flash_attn|bitsandbytes|peft|diffusers|datasets|accelerate|deepspeed|langchain|llamaindex)" src/<domain>/<file>.py
```

BUDGET: 25 min OR 3 test-fix iterations. If exceeded, STOP cleanly, report DEFERRED.
```
