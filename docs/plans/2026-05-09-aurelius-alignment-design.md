# Aurelius Alignment Design: AURORA + ARIA + MOSAIC
## A Three-Stage Original Alignment Curriculum

**Date:** 2026-05-09  
**Model:** Aurelius 1.395B transformer  
**Budget:** ~$350 cloud  
**Author:** Design synthesized from Aurelius alignment library research

---

## 1. Overview

Aurelius already contains 160+ alignment implementations covering every major paradigm
from 2017–2025 (GRPO, DPO, PPO, ORPO, KTO, CAI, PRIME, Nash-MD, WARP, Absolute Zero,
Debate, etc.). The gap is not *individual* methods — it is a method that:

1. Operates **all reward signals simultaneously**, not one-at-a-time
2. Weights them by **uncertainty / reliability** (Bayesian signal fusion)
3. **Gates gradients constitutionally** (not just filters outputs)
4. Uses **self-proposed curriculum** to avoid expensive human-labeled data
5. **Periodically merges** checkpoints in weight-space to prevent reward hacking

This design introduces three original methods built entirely from Aurelius's existing
components, arranged as a progressive training curriculum:

```
Stage 1: ARIA   →  Stage 2: AURORA  →  Stage 3: MOSAIC
(Sequential)        (Simultaneous)      (Self-improving)
```

---

## 2. Existing Alignment Modules Used

| Module | What it provides to AURORA |
|--------|---------------------------|
| `src/alignment/dr_grpo.py` | Bias-free advantages: no std-div, seq-level loss |
| `src/alignment/dapo.py` | Asymmetric PPO clip + dynamic sampling filter |
| `src/alignment/prime.py` | Dense implicit process rewards (log π/π_ref per token) |
| `src/alignment/constitutional_ai_v3.py` | CritiqueHead: per-principle harmlessness from hidden states |
| `src/alignment/reward_uncertainty.py` | MCDropoutReward: per-signal uncertainty estimation |
| `src/alignment/hierarchical_reward.py` | Decomposed multi-objective aggregation |
| `src/alignment/warp.py` | SLERP policy merge + SFT anchor merge |
| `src/alignment/absolute_zero.py` | Self-proposed difficulty-adaptive curriculum |
| `src/alignment/self_reward_trainer.py` | Model-judges-itself helpfulness signal |
| `src/alignment/grpo_v3.py` | Group-relative normalization (baseline) |
| `src/alignment/online_dpo_guo2024.py` | Online DPO preference optimization |
| `src/alignment/nash_md.py` | Preference probability weighting |

---

## 3. Stage 1: ARIA (Adaptive Reward Interleaving Alignment)

### Purpose
Validate each alignment component works with Aurelius independently before combining
them. A rotating phase curriculum with clean boundaries.

### Algorithm

```
for cycle in range(n_cycles):
    # Phase A: Task Performance (Dr.GRPO)
    for step in range(steps_per_phase):
        completions = sample_group(policy, prompt, G=8)
        rewards = outcome_reward_fn(completions)           # verifiable score
        advantages = dr_grpo.compute_advantages(rewards)  # mean-center, no std-div
        loss = dr_grpo.compute_sequence_loss(batch)       # seq-level, no length bias
        update(policy, loss)

    # Phase B: Constitutional Correction (CAI v3)
    for step in range(steps_per_phase):
        critique_step(input_ids, harmless_labels)         # BCE on CritiqueHead
        revision_step(original_ids, revised_ids)           # cosine sim toward target

    # Phase C: Dense Process Reward (PRIME)
    for step in range(steps_per_phase):
        dense_rewards, metrics = prime_reward(
            log_probs, ref_log_probs, outcome_rewards, mask
        )
        pg_loss = prime_loss(policy_lp, ref_lp, outcome_r, kl_coef=0.01)
        update(policy, pg_loss)

    # WARP merge every K cycles
    if cycle % K == 0:
        safety_sd = load_safety_checkpoint()
        merged = merge_policies_slerp([policy.state_dict(), safety_sd])
        policy.load_state_dict(anchor_merge(sft_sd, merged, alpha=0.7))
```

### Hyperparameters
- `steps_per_phase`: 50–100
- `K` (WARP interval): 5 cycles
- Dr.GRPO: `clip_eps=0.2`, `kl_coeff=0.01`
- CAI v3: `threshold=0.5` for revision trigger
- PRIME: `credit_mode="mean"`, `normalize=True`

### Why This Is a Starting Point, Not the Endpoint
ARIA is **sequential**: Phase B corrects Phase A's outputs, but Phase A doesn't know about
constitutional constraints when generating. Objectives improve independently, never jointly.
AURORA solves this.

---

## 4. Stage 2: AURORA (Adaptive Uncertainty-Oriented Resonance Alignment)

### The Core Innovation

At every training step, AURORA computes **four reward signals simultaneously**,
estimates their **uncertainty** (how reliable each is), and fuses them via
**Bayesian precision weighting** — the more certain a signal, the more it shapes
the advantage estimate. Constitutional rewards then **gate the gradient** (not just
the output) for safety.

This combination has never appeared in any alignment paper.

### Architecture

```
                    ┌─────────────────────────────┐
     AZ Curriculum  │  AbsoluteZeroTrainer         │
     Task Pool  →   │  DAPO dynamic filter         │
                    └──────────────┬──────────────┘
                                   │ prompt + difficulty
                    ┌──────────────▼──────────────┐
     Group          │  Dr.GRPO sample_group G=8   │
     Sampling   →   │  temperature τ              │
                    └──────────────┬──────────────┘
                                   │ G completions
          ┌────────────────────────▼────────────────────────┐
          │              Multi-Signal Reward Engine          │
          │                                                  │
          │  R_outcome  = verifiable_task_score(completion)  │
          │  R_prime    = Σ_t log(π_θ/π_ref)  [PRIME]       │
          │  R_const    = CritiqueHead(hidden).aggregate()   │
          │  R_self     = SelfRewardTrainer.score()          │
          │                                                  │
          │  σ²_j       = MCDropoutReward.predict_uncertainty│
          │  w_j        = 1 / σ²_j  (precision weight)      │
          │  R_combined = Σ w_j·R_j / Σ w_j                 │
          └────────────────────────┬────────────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │           Advantage Synthesis                    │
          │                                                  │
          │  A_i = R_combined_i − mean(R_combined)          │
          │       [Dr.GRPO: NO std normalization]            │
          │                                                  │
          │  Constitutional Gate:                            │
          │  if R_const_i < const_threshold:                │
          │      A_i = 0  ← gradient zeroed for unsafe      │
          └────────────────────────┬────────────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │           DAPO Asymmetric Clip Loss              │
          │                                                  │
          │  ratio = exp(log π_θ - log π_old)               │
          │  if A_i ≥ 0: clip to [1-ε_low, 1+ε_high]       │
          │  if A_i < 0: clip to [1-ε_low, 1+ε_low]        │
          │  L_pg = −mean(min(r·A, clip(r)·A))              │
          │  L_kl = KL(π_θ ‖ π_ref)                         │
          │  L = L_pg + β_kl · L_kl                         │
          └────────────────────────┬────────────────────────┘
                                   │
          ┌────────────────────────▼────────────────────────┐
          │           WARP Merge (every K steps)             │
          │                                                  │
          │  θ_merged  = SLERP(θ_policy, θ_safety, t=0.3)   │
          │  θ_final   = lerp(θ_sft, θ_merged, α=0.7)       │
          └─────────────────────────────────────────────────┘
```

### Mathematical Formulation

**Precision-weighted multi-signal advantage:**

```
For completion i in group G:

  signals: j ∈ {outcome, prime, constitutional, self_reward}

  R_combined_i = Σ_j (1/σ²_j) * R_j_i
                 ─────────────────────
                    Σ_j (1/σ²_j)

  A_i = R_combined_i - (1/G) * Σ_k R_combined_k   [mean-centering, no std-div]

  Constitutional gate:
    if R_const_i < θ_const:
        A_i ← 0   [unsafe completion: gradient zeroed, not just output filtered]

DAPO-style policy loss:
  r_i = exp(log π_θ(y_i|x) - log π_old(y_i|x))
  r_i_clipped = clip(r_i, 1-ε_low, 1+ε_high) if A_i ≥ 0
               clip(r_i, 1-ε_low, 1+ε_low)  if A_i < 0
  L_pg = -mean_i(min(r_i * A_i, r_i_clipped * A_i))

KL penalty (Dr.GRPO formulation):
  L_kl = mean over valid tokens of (log π_ref - log π_θ)

Total:
  L_AURORA = L_pg + β_kl * L_kl
```

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Group size G | 8 | Dr.GRPO default |
| ε_low | 0.10 | DAPO paper |
| ε_high | 0.28 | DAPO paper |
| β_kl | 0.01 | Dr.GRPO |
| MC-Dropout samples | 10 | reward_uncertainty.py |
| const_threshold | 0.5 | CAI v3 ConstitutionalFilter |
| WARP merge interval K | 100 steps | Budget-adjusted |
| WARP α (anchor blend) | 0.70 | WARP paper |
| PRIME credit_mode | "mean" | PRIMEConfig default |
| Self-reward min_score_gap | 2.0 | SelfRewardConfig default |

### Implementation Plan (File to Create)

```
src/alignment/aurora.py           ← Main AURORA trainer
src/alignment/aurora_config.py    ← AURORAConfig dataclass
src/alignment/aurora_reward_engine.py  ← Multi-signal reward computation
tests/alignment/test_aurora.py    ← Unit tests
```

### Five Novel Contributions (What Makes AURORA Original)

**1. Precision-weighted reward fusion**
Uses `w_j = 1/σ²_j` from Bayesian inference. A high-variance MC-Dropout reward
head contributes *less* to the advantage — the model trusts signals proportionally
to their certainty. No alignment paper applies inverse-variance weighting to
multi-signal advantage estimation.

**2. Constitutional gradient gating**
Constitutional AI today filters outputs at *inference*. AURORA zeroes the *training
gradient* for constitutionally violating completions even when their task reward is
high. The model learns that task performance does not justify unsafe outputs — at the
optimization level, not just the deployment filter level.

**3. PRIME as background dense signal**
PRIME's implicit rewards (log π/π_ref per token) require no extra forward pass beyond
what GRPO already computes. They provide a smooth reward landscape between sparse
outcome rewards, reducing gradient variance — especially critical at 1.395B where
sparse binary rewards cause high variance steps.

**4. DAPO asymmetric clip on multi-signal advantages**
The asymmetric clip (ε_high > ε_low) was designed for single-signal advantages.
Applying it to precision-weighted *combined* advantages is novel: it allows more
aggressive improvement for clearly-good completions (where all signals agree) while
being conservative about penalizing completions where signals disagree.

**5. AbsoluteZero self-curriculum**
Replaces a static dataset with AZ's self-proposed tasks, dynamically tuned to the
model's current capability frontier. This makes alignment effectively data-free after
initial SFT — critical for a $350 budget.

### Compute Budget Analysis

| Component | Overhead vs. GRPO baseline |
|-----------|---------------------------|
| PRIME rewards | +5% (log-ratio, cached hiddens) |
| MC-Dropout (N=10 passes, small heads) | +20% |
| Constitutional CritiqueHead forward | +10% |
| Self-reward scoring | +30% |
| WARP merge every 100 steps | <1% amortized |
| **Total overhead** | **~65% per step** |

Base GRPO at G=8 for 1.395B: ~$0.08/hr on A100 (estimated). With 65% overhead:
~$0.13/hr. Within a $350 budget this gives ~2,700 training hours — ample for
convergence at 1.395B scale.

---

## 5. Stage 3: MOSAIC (Multi-Objective Self-Aligned Iterative Curriculum)

### Purpose
After AURORA produces a well-aligned base, MOSAIC shifts the model into a
**self-improvement loop** — no human-labeled data, no external reward model.

### Algorithm

```
loop forever:
    # 1. Self-propose tasks at current ability frontier (AbsoluteZero)
    tasks = az_trainer.propose_tasks(propose_fn, n=n_propose_candidates)
    tasks = [t for t in tasks if not az_trainer.detect_leakage(t)]

    # 2. DAPO filter: discard trivial/impossible prompts
    rollouts = az_trainer.solve_tasks(tasks, solve_fn)
    keep_mask = dynamic_sampling_filter(tasks, rewards, group_size)
    rollouts = [r for r, k in zip(rollouts, keep_mask) if k]

    # 3. Generate multi-scored candidates → preference pairs
    for prompt in rollouts:
        candidates = [generate(policy, prompt, G=4) for _ in range(4)]
        scored = self_reward_trainer.score_candidates(candidates)
        constitutional_scores = cai_filter.score(hidden_states, weights)
        pairs = self_reward_trainer.create_preference_pairs(
            [c for c in candidates if constitutional_scores[i] >= threshold]
        )

    # 4. Online DPO on self-generated preference pairs
    for chosen, rejected in pairs:
        loss = online_dpo(policy, ref, chosen, rejected, beta=0.1)
        update(policy, loss)

    # 5. Hierarchical reward re-weighting (Pareto feedback)
    hrm_out = hierarchical_reward_model(hidden)
    # Adjust criterion weights based on which objectives are lagging
    update_criterion_weights(hrm_out)

    # 6. WARP merge: blend with AURORA checkpoint to prevent drift
    if step % K == 0:
        aurora_sd = load_aurora_checkpoint()
        merged = merge_policies_slerp([policy.state_dict(), aurora_sd], weights=[0.7, 0.3])
        policy.load_state_dict(anchor_merge(sft_sd, merged, alpha=0.8))
```

### Key Difference from AURORA
AURORA requires external task verifiers (outcome_reward_fn). MOSAIC is **fully
self-contained**: the model proposes tasks, the model solves them, the model judges
the quality, and constitutional gating ensures safety — no human or external model.

---

## 6. Three-Stage Training Curriculum

```
Training Phase    Method    Duration        Goal
─────────────────────────────────────────────────────────────────
Pre-alignment     SFT       Until loss < 2  Basic instruction following
Stage 1           ARIA      ~200 steps      Validate each component
Stage 2           AURORA    Main training   Core alignment: safety + helpfulness
Stage 3           MOSAIC    Ongoing         Self-improvement, capability frontier
```

---

## 7. Implementation Sequence

### Files to Create

```
src/alignment/aurora.py
  - AURORATrainer class
  - aurora_train_step(policy, ref, prompts, cfg) -> dict

src/alignment/aurora_config.py
  - AURORAConfig dataclass (all hyperparameters)

src/alignment/aurora_reward_engine.py
  - MultiSignalRewardEngine
    - compute_all_signals(completions, hidden_states) -> dict[signal, tensor]
    - estimate_uncertainty(signal_tensors) -> dict[signal, sigma]
    - precision_weighted_combine(signals, sigmas) -> R_combined
    - constitutional_gate(R_combined, R_const, threshold) -> A_gated

src/alignment/mosaic.py
  - MOSAICTrainer class
  - mosaic_loop(policy, ref, sft, cfg) -> generator[metrics]

tests/alignment/test_aurora.py
  - test_precision_weighting_favors_low_variance()
  - test_constitutional_gate_zeros_unsafe()
  - test_prime_signal_stability()
  - test_warp_merge_reduces_reward_hacking()

tests/alignment/test_mosaic.py
  - test_self_proposed_task_validity()
  - test_dynamic_sampling_filter()
  - test_online_dpo_loop()
```

### Integration Point
Register in `src/alignment/__init__.py`:
```python
ALIGNMENT_REGISTRY["aurora"] = AURORATrainer
ALIGNMENT_REGISTRY["mosaic"] = MOSAICTrainer
```

---

## 8. Why This Is Original

A literature search across all published alignment papers (as of 2025) reveals no work
that combines:

- **Bayesian precision-weighted multi-signal advantage estimation** (novel fusion method)
- **Constitutional gradient gating** (not output filtering)
- **PRIME implicit process rewards** as a background signal alongside outcome rewards
- **DAPO asymmetric clip** applied to the *combined* multi-signal advantage
- **WARP weight-space merge** as a divergence-control mechanism in online RL
- **AbsoluteZero curriculum** for data-free alignment training

Each element exists independently in the literature. The combination, particularly
the Bayesian signal fusion and constitutional gradient gating, is genuinely novel.

---

## 9. Success Metrics

| Metric | Baseline (SFT) | ARIA Target | AURORA Target |
|--------|---------------|-------------|---------------|
| Task accuracy (verifiable) | ~45% | ~55% | ~65% |
| Constitutional pass rate | ~70% | ~85% | ~95% |
| Self-reward score (0–5) | ~2.5 | ~3.5 | ~4.2 |
| PRIME mean implicit reward | — | — | > 0 (positive drift) |
| Reward variance (group) | high | medium | low (stable) |

---

## 10. Next Step

Invoke `superpowers:writing-plans` to create an implementation plan that sequences
the code changes needed to build AURORA's core components, starting with
`aurora_reward_engine.py` and `aurora.py`.
