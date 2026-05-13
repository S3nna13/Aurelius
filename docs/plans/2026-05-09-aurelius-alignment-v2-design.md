# MOSAIC v2: Multi-Objective Steering Architecture with Integrated Constitutional Guidance

**Date:** 2026-05-09  
**Status:** Design — supersedes `2026-05-09-aurelius-alignment-design.md`  
**Model:** AureliusTransformer 1.395B (d_model=2048, n_layers=24, n_heads=16/8 GQA, d_ff=5632, vocab=128K)

---

## 1. Executive Summary

MOSAIC v2 is the second-generation alignment architecture for Aurelius. It synthesizes every alignment module in `src/alignment/` into a single cohesive training system, then extends it with three novel contributions that exploit Aurelius's specific architectural primitives in ways no published alignment method currently does:

1. **Steering-Reward Correspondence (SRC)** — a self-referential reward signal derived from how much latent-space steering is required to keep a completion constitutionally safe. Models that have already internalized safe behavior need less steering; less steering = higher reward.
2. **Expert Safety Affinity (ESA)** — an auxiliary routing loss that designates SparseMoE experts {0, 1} as "safety experts" and pushes constitutionally unsafe tokens to route to them during training, creating interpretable structural specialization.
3. **Multi-Token Alignment Horizon (MTAH)** — extends the alignment advantage signal temporally over MTP-predicted future tokens, preventing myopic alignment where the current token is safe but the continuation is not.

Combined with precision-weighted Bayesian signal fusion (from Reward Uncertainty + PRIME + Hierarchical Reward), constitutional gradient gating (CAI v3), thinking-token-separated objectives, WARP policy merging, and AbsoluteZero self-curriculum, MOSAIC v2 produces a system with 9 interacting alignment mechanisms and 3 novel architecture-aware innovations.

---

## 2. Aurelius Architecture Exploitable Properties

These are the Aurelius-specific features MOSAIC v2 uniquely leverages. Generic alignment papers cannot exploit them because they are Aurelius-specific.

| Feature | Location | How MOSAIC v2 Uses It |
|---------|----------|----------------------|
| SparseMoE (8 experts, top-2) | `src/model/sparse_moe.py` | ESA: designate experts 0,1 as safety experts via auxiliary routing loss |
| MoD token routing (50% capacity) | `src/model/mod.py` | Constitutionally unsafe token features get lower compute budget in subsequent layers |
| MTP head (k=2 future tokens) | `src/model/mtp.py` | MTAH: extend advantage signal over 2 future token predictions |
| Latent Steering | `src/model/latent_steering.py` | SRC: measure cosine distance between steered/unsteered hidden states as reward |
| Hybrid Attention (GQA/CSA/HCA) | `src/model/transformer.py` | SRC steering is applied at CSA/HCA layers (12, 16, 20) — deeper representation |
| Thinking Tokens (200001, 200002) | `src/alignment/thinking_tokens.py` | Split alignment objectives: CoT quality for think tokens, constitutional+helpfulness for answer tokens |
| NeuralBrain reflect loop | `src/agent/neural_brain.py` | Structured self-critique provides richer reward signal than scalar LLM-as-judge |
| ValueHead | `src/model/value_head.py` | Provides per-token value estimates for GAE-style baseline in GRPO |

---

## 3. Complete Alignment Module Synthesis

Every alignment module used and the role it plays:

### Signal Generation Layer
| Module | File | Role in MOSAIC v2 |
|--------|------|-------------------|
| GRPO v3 / Dr.GRPO | `grpo_v3.py`, `dr_grpo.py` | Group sampling (N=8), bias-free advantage (no std-div), sequence-length-normalized loss |
| DAPO | `dapo.py` | Asymmetric clip (ε_low=0.2, ε_high=0.28), entropy bonus (λ_ent=0.001), dynamic group filter |
| PRIME | `prime.py` | Dense per-token process reward from log(π_θ/π_ref) — no extra forward pass, background signal |
| Constitutional AI v3 | `constitutional_ai_v3.py` | CritiqueHead (n_principles=8) scores on safety/harmlessness/helpfulness |
| Contrastive CoT | `contrastive_cot.py` | 5th signal: rewards correct reasoning chains over incorrect ones (process quality) |
| ODIN | `odin.py` | Length-bias disentanglement via per-token normalization: removes spurious length-reward correlation |
| Hierarchical Reward | `hierarchical_reward.py` | Learned per-criterion weights (softmax-normalized, diversity regularized) |
| Reward Uncertainty | `reward_uncertainty.py` | MC-Dropout (N=20) per signal → σ²_j for Bayesian precision weighting |

### Optimization Layer
| Module | File | Role in MOSAIC v2 |
|--------|------|-------------------|
| Token DPO | `token_dpo.py` | Per-token credit assignment via softmax-normalized advantage weights |
| Thinking Tokens | `thinking_tokens.py` | Split loss weights: think_weight=0.5 (CoT quality), answer_weight=1.0 (helpfulness+safety) |
| Activation Steering | `activation_steering.py` + `model/latent_steering.py` | Steer residual streams during GRPO sampling; measure steering cost as SRC reward |
| Constitutional Filter | `constitutional_filter.py` | Gradient gating: zero gradient for completions scoring below τ_gate=0.4 |
| WARP | `warp.py` | Periodic SLERP ensemble merge + linear SFT anchor (every Δ=50 steps) |
| AbsoluteZero | `absolute_zero.py` | Self-play task curriculum: deduction/abduction/induction with leakage detection |
| Self-Reward Trainer | `self_reward_trainer.py` | Iterative DPO on self-scored preference pairs (bootstrapped at Stage 1) |
| STILL | `still.py` | Generate→verify→filter→SFT loop for bootstrapping Stage 1 seed data |
| SimPO | `simpo.py` | Reference-free DPO fallback when π_ref KL diverges too far (Stage 3 safety net) |

### Architecture-Coupled Layer (Novel — MOSAIC v2 Only)
| Component | Exploits | Description |
|-----------|---------|-------------|
| SRC (Steering-Reward Correspondence) | Latent Steering + Hybrid Attention | Self-referential safety metric from steering magnitude |
| ESA (Expert Safety Affinity) | SparseMoE routing | Structural safety specialization in designated experts |
| MTAH (Multi-Token Alignment Horizon) | MTP head | Temporally extended alignment advantage signal |

---

## 4. Mathematical Formulation

### 4.1 Reward Signal Decomposition

For a completion y_i sampled from prompt x, the MOSAIC v2 reward decomposes as:

```
R_total(y_i) = Σ_j w_j · R_j(y_i)
```

where the precision weights are:

```
w_j = (1/σ²_j) / Σ_k (1/σ²_k)
```

and σ²_j = MC-Dropout variance of R_j over N=20 stochastic forward passes.

**R_quality (PRIME process reward):**
```
R_quality(y_i) = (1/T_i) · Σ_t [log π_θ(y_{i,t}|y_{i,<t},x) - log π_ref(y_{i,t}|y_{i,<t},x)]
```
This is the PRIME implicit process reward — dense, per-token, computed from the same forward pass used for the policy gradient. No additional model call needed.

**R_const (constitutional):**
```
R_const(y_i) = (1/n_principles) · Σ_p σ(CritiqueHead_p(pool(h_L(y_i))))
```
where CritiqueHead is an MLP with n_principles=8 output heads on the pooled final-layer hidden state.

**R_cot (contrastive chain-of-thought):**
```
R_cot(y_i) = exp(log π_SFT(y_think_correct)) / [exp(log π_SFT(y_think_correct)) + exp(log π_SFT(y_think_wrong))]
```
Applied to the `<think>...</think>` span only. Rewards completions whose reasoning chain matches verified-correct CoT samples.

**R_length (ODIN length-disentangled):**
```
R_length(y_i) = [Σ_t (log π_θ(y_{i,t}) - log π_ref(y_{i,t})) · mask_t] / (|y_i| · λ_len)
```
where λ_len=1.0 and |y_i| is the number of non-padding tokens. This normalizes the implicit reward by sequence length, preventing the model from learning to pad for reward.

**R_hier (hierarchical multi-criterion):**
```
R_hier(y_i) = Σ_c softmax(w_c) · CriterionHead_c(pool(h_L(y_i)))
```
where the n_criteria=4 criterion heads (factuality, coherence, instruction-following, safety) have learnable weights updated via diversity regularization.

**R_SRC (Steering-Reward Correspondence — NOVEL):**
```
R_SRC(y_i) = -λ_SRC · (1/|L_steer|) · Σ_{l ∈ L_steer} cosine_distance(h_l^unsteered(y_i), h_l^steered(y_i))
```
where L_steer = {12, 16, 20} (CSA/HCA layers), h_l^steered is the hidden state under constitutional steering vector (α=0.3, method="add"), and h_l^unsteered is the vanilla forward pass.

**Interpretation:** A model that needs steering to stay safe has large cosine distance → negative SRC reward. A model that has internalized safe behavior has near-zero distance → near-zero SRC penalty. Over training, this signal pushes the model toward inherently safe completions that don't require external intervention.

### 4.2 Combined Reward and Group Advantage

After computing all 6 reward signals per completion:

```
R_combined_i = Σ_j w_j · R_j(y_i)         # Bayesian precision-weighted fusion
```

DAPO dynamic filter — keep the group only if:
```
0 < mean(R_combined) < 1                    # non-trivial, non-saturated group
```

Dr.GRPO group advantage (bias-free: normalize only if n_group > 1):
```
ā_i = (R_combined_i - mean_g) / (std_g + ε)    ε=1e-8
```

### 4.3 Per-Token Advantage with MTAH Extension

**Base per-token expansion (Dr.GRPO):**
```
ā_{i,t} = ā_i     (same advantage for all tokens in completion i)
```

**MTAH extension (when MTP is active, mtp_n_predict=2):**
```
Ā_{i,t} = Σ_{k=0}^{K} γ^k · ā_{i,t+k}      K=2, γ=0.95
```
where ā_{i,t+k} is the GRPO advantage of the k-th future token as predicted by the MTP head. This extends the alignment horizon so the policy is penalized for safe current tokens followed by unsafe continuations.

**TokenDPO per-token credit weights:**
```
c_{i,t} = softmax(Ā_{i,t} / τ_token)          τ_token=1.0, over positions t
```

**Thinking-token split:**
```
loss_weight_{i,t} = think_weight=0.5   if token_is_think
                    answer_weight=1.0  if token_is_answer
```

### 4.4 MOSAIC v2 Policy Loss

**Importance ratio:**
```
r_{i,t} = π_θ(y_{i,t}|y_{i,<t},x) / π_ref(y_{i,t}|y_{i,<t},x)
```

**DAPO asymmetric clip:**
```
ε_upper = ε_high=0.28  if Ā_{i,t} > 0 else ε_low=0.20
clipped_{i,t} = clip(r_{i,t}, 1-ε_low, 1+ε_upper)
```

**Full MOSAIC v2 loss:**
```
L_MOSAIC = -1/n_group · Σ_i [I_{gate_i}/T_i · Σ_t loss_weight_{i,t} · c_{i,t} · min(r_{i,t}·Ā_{i,t}, clipped_{i,t}·Ā_{i,t})]
           + β_KL · KL(π_θ || π_ref)
           + λ_ent · H_entropy_bonus
           + α_ESA · L_ESA
```

where:
- `I_{gate_i} = 1` if R_const_i ≥ τ_gate=0.4 else 0 (constitutional gradient gating — zeroes gradient for unsafe completions entirely)
- β_KL=0.04 (KL penalty to reference)
- λ_ent=0.001 (entropy bonus for exploration)
- α_ESA=0.01 (ESA auxiliary loss weight)

### 4.5 Expert Safety Affinity Loss (ESA — Novel)

During GRPO training, for each forward pass through MoE layers:

1. **Identify safety-sensitive tokens:** tokens where R_const < τ_safety=0.5 at their position
2. **Compute ESA loss:**
```
L_ESA = -(1/|T_safety|) · Σ_{t ∈ T_safety} log[Σ_{e ∈ {0,1}} p_route(e|h_t)]
```
where p_route(e|h_t) is the softmax router probability for expert e given token hidden state h_t.

**Effect:** Constitutionally-flagged tokens are incrementally pushed toward safety experts 0,1 via cross-entropy routing supervision. Over thousands of steps, safety-relevant computation concentrates in interpretable model components, making alignment more robust to downstream fine-tuning.

### 4.6 WARP Policy Merging

Every Δ=50 optimization steps:

1. Collect K=4 recent policy snapshots: {θ_1, θ_2, θ_3, θ_4}
2. Pairwise SLERP merge:
```
θ_A = SLERP(θ_1, θ_2, t=0.5)
θ_B = SLERP(θ_3, θ_4, t=0.5)
θ_merged = SLERP(θ_A, θ_B, t=0.5)
```
3. Linear anchor toward SFT reference (prevents reward hacking drift):
```
θ_new = (1-μ)·θ_merged + μ·θ_SFT      μ=0.05
```

### 4.7 Constitutional Gradient Gating

For each completion in the GRPO group, before computing gradients:
```
if R_const(y_i) < τ_gate:
    ∂L/∂θ from completion i ← 0
```

This is gradient-level filtering, not output filtering. Constitutionally unsafe completions contribute no gradient signal — the policy doesn't learn from them at all.

---

## 5. Three-Stage Training Curriculum

### Stage 1: ARIA (Alignment via Representation-Informed Adaptation)
**Goal:** Seed the model with constitutional behavior and thinking-token structure before RL.
**Duration:** 1,000 steps  
**Methods:** STILL (generate→verify→filter→SFT) + Self-Reward bootstrapping + constitutional filtering

**ARIA loss:**
```
L_ARIA = L_SFT(correct_answers) + λ_think·L_CoT(correct_reasoning) + L_const_filter
```

Key setup:
- Run STILL for N_iter=3 iterations, N_samples=8 per prompt, keep top-k=2 verified completions
- Bootstrap Self-Reward preference pairs where score_gap ≥ 2.0
- Apply CritiqueHead constitutional filter (reject completions with R_const < 0.5)
- Use ThinkingLossWeights to give think tokens 0.5× weight (they're seed data, not yet high quality)

### Stage 2: AURORA (Adaptive Uncertainty-Regularized Online Reinforcement Alignment)
**Goal:** Online GRPO with 4-signal fusion and constitutional gating. No SRC/ESA/MTAH yet (MTP not active).
**Duration:** 3,000 steps  
**Methods:** GRPO v3 + DAPO filter + PRIME + Constitutional + ODIN + Hierarchical + Reward Uncertainty + Gradient Gating

**AURORA uses 4 reward signals** (not SRC/MTAH — those require the model to be more stable first):
```
R_AURORA = w_quality·R_quality + w_const·R_const + w_length·R_length + w_hier·R_hier
```
with precision-weighted fusion from MC-Dropout σ²_j estimates.

**AbsoluteZero** self-curriculum activates at Step 500 of AURORA: the model begins proposing its own deduction/abduction/induction tasks.

### Stage 3: MOSAIC v2 (Full System)
**Goal:** Full 6-signal alignment with all novel contributions active.
**Duration:** 5,000 steps  
**Methods:** All of AURORA + SRC + ESA + MTAH + ContrastiveCoT + WARP

**Activation schedule:**
- Steps 0–500: AURORA config (warm-up, MTP not yet active)
- Steps 500–1000: Activate SRC (α=0.1 initially), CCoT as 5th signal
- Steps 1000–2000: Activate ESA (α_ESA=0.005), ramp SRC to α=0.3
- Steps 2000+: Activate MTAH (requires stable MTP predictions), full WARP merging

---

## 6. Hyperparameter Table

### Reward Signal Weights (initial; updated by precision weighting)
| Signal | Symbol | Initial w_j | Source Module |
|--------|---------|------------|---------------|
| PRIME quality | R_quality | 0.3 | `prime.py` |
| Constitutional | R_const | 0.3 | `constitutional_ai_v3.py` |
| Contrastive CoT | R_cot | 0.15 | `contrastive_cot.py` |
| ODIN length | R_length | 0.1 | `odin.py` |
| Hierarchical | R_hier | 0.1 | `hierarchical_reward.py` |
| SRC steering | R_SRC | 0.05 | `activation_steering.py` + `latent_steering.py` |

### GRPO / Optimization
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_group | 8 | Matches DAPO default; enough diversity for group advantages |
| ε_low | 0.20 | PPO lower clip ratio |
| ε_high | 0.28 | DAPO higher clip for positive advantages |
| β_KL | 0.04 | KL regularization to reference |
| λ_ent | 0.001 | Entropy bonus for exploration |
| temperature | 0.9 | GRPO sampling temperature |
| max_new_tokens | 64 (Stage 1), 128 (Stage 2–3) | Curriculum length |

### Novel Contribution Parameters
| Parameter | Value | Component |
|-----------|-------|-----------|
| L_steer layers | {12, 16, 20} | SRC (CSA/HCA layers) |
| α (steering) | 0.3 | SRC steering coefficient |
| λ_SRC | 0.1→0.3 | SRC reward scale (ramped) |
| Safety experts | {0, 1} | ESA |
| α_ESA | 0.005→0.01 | ESA loss weight (ramped) |
| τ_safety | 0.5 | ESA token flagging threshold |
| γ (MTAH discount) | 0.95 | MTAH temporal discount |
| K (MTAH horizon) | 2 | MTP head look-ahead |

### WARP / Curriculum
| Parameter | Value | Component |
|-----------|-------|-----------|
| Merge interval Δ | 50 steps | WARP |
| SFT anchor μ | 0.05 | WARP linear anchor |
| K checkpoints | 4 | WARP ensemble size |
| Leakage threshold | cos_sim > 0.9 | AbsoluteZero |
| Leakage penalty | -0.5 | AbsoluteZero |
| Constitutional gate τ | 0.4 | Gradient gating |
| think_weight | 0.5 | Thinking Tokens |
| answer_weight | 1.0 | Thinking Tokens |
| MC-Dropout passes | 20 | Reward Uncertainty |

---

## 7. Implementation Architecture

### File Structure
```
src/alignment/
├── mosaic_v2/
│   ├── __init__.py
│   ├── config.py              # MOSAICv2Config dataclass (all hyperparams)
│   ├── reward_signals.py      # All 6 reward signal computations
│   ├── steering_reward.py     # SRC: Steering-Reward Correspondence
│   ├── expert_safety_affinity.py  # ESA: routing auxiliary loss
│   ├── mtah.py                # MTAH: multi-token alignment horizon
│   ├── precision_fusion.py    # Bayesian inverse-variance weighting
│   ├── mosaic_loss.py         # Full L_MOSAIC with all components
│   ├── trainer.py             # MOSAICv2Trainer.train_step()
│   └── curriculum.py          # ARIA→AURORA→MOSAIC stage controller
```

### Key Dependencies (already implemented in Aurelius)
- `src/alignment/prime.py` → PRIMEReward
- `src/alignment/constitutional_ai_v3.py` → CritiqueHead, ConstitutionalFilter
- `src/alignment/reward_uncertainty.py` → MCDropoutReward, DeepEnsembleReward
- `src/alignment/hierarchical_reward.py` → HierarchicalRewardModel
- `src/alignment/dapo.py` → DAPOFilter, DAPOLoss, dynamic_sampling_filter
- `src/alignment/dr_grpo.py` → DrGRPOTrainer, DrGRPOConfig
- `src/alignment/contrastive_cot.py` → CCoTConfig, compute_ccot_loss
- `src/alignment/odin.py` → ODINLoss
- `src/alignment/token_dpo.py` → TokenDPOTrainer, compute_per_token_advantages
- `src/alignment/thinking_tokens.py` → ThinkingLossWeights, THINK_START_TOKEN_ID
- `src/alignment/activation_steering.py` → SteeringHook, ActivationSteerer
- `src/model/latent_steering.py` → extract_hidden_states, SteeringConfig
- `src/alignment/warp.py` → WARPTrainer, slerp_two, anchor_merge
- `src/alignment/absolute_zero.py` → AbsoluteZeroTrainer
- `src/alignment/still.py` → STILLTrainer
- `src/alignment/self_reward_trainer.py` → SelfRewardTrainer

### train_step() Pseudocode
```python
def train_step(prompt_ids: Tensor) -> dict:
    # 1. Sample completions with activation steering
    with SteeringHook(model, layers=[12,16,20], coeff=0.3, direction=safety_vec):
        group_ids, _ = sample_group(policy, prompt_ids, n_group=8, ...)
        h_steered = collect_hidden_states(layers=[12,16,20])

    # Unsteered hidden states for SRC
    h_unsteered = collect_hidden_states_no_hook(layers=[12,16,20])

    # 2. Compute all 6 reward signals
    R_quality   = prime_reward.compute(group_ids, prompt_ids)
    R_const     = critique_head.forward(h_unsteered)
    R_cot       = ccot_scorer.score(group_ids)
    R_length    = odin_loss.normalize(group_ids, prompt_ids)
    R_hier      = hierarchical_reward.forward(h_unsteered)
    R_SRC       = -lambda_SRC * cosine_dist(h_steered, h_unsteered).mean()

    # 3. Uncertainty quantification
    sigma_sq = {j: mc_dropout_reward.predict_uncertainty(R_j) for j in signals}
    weights  = {j: 1/sigma_sq[j] / sum(1/sigma_sq.values()) for j in signals}

    # 4. Precision-weighted fusion
    R_combined = sum(weights[j] * R_j for j, R_j in signals.items())

    # 5. DAPO group filter
    if not (0 < R_combined.mean() < 1):
        return None  # skip non-informative group

    # 6. Group advantages (Dr.GRPO bias-free)
    adv = (R_combined - R_combined.mean()) / (R_combined.std() + 1e-8)

    # 7. MTAH extension
    adv_extended = mtah.extend(adv, mtp_predictions, gamma=0.95, K=2)

    # 8. TokenDPO per-token credit weights
    token_weights = softmax(adv_extended / tau_token)

    # 9. Thinking-token split weights
    loss_weights = thinking_weights.compute(group_ids)  # 0.5 think / 1.0 answer

    # 10. Policy log probs (with gradient)
    log_probs_policy = compute_token_log_probs(policy, prompt_ids, group_ids)
    log_probs_ref    = compute_token_log_probs(ref, prompt_ids, group_ids).detach()

    # 11. Constitutional gradient gating
    gate_mask = (R_const >= tau_gate).float()  # [n_group]

    # 12. DAPO asymmetric clip
    ratio = exp(log_probs_policy - log_probs_ref)
    epsilon_upper = where(adv_extended > 0, epsilon_high, epsilon_low)
    clipped = clamp(ratio, 1 - epsilon_low, 1 + epsilon_upper)

    # 13. MOSAIC v2 policy loss
    raw_loss = -min(ratio * adv_extended, clipped * adv_extended)
    gated = gate_mask.unsqueeze(-1) * raw_loss * loss_weights * token_weights
    L_policy = gated.sum() / (n_group * max_new_tokens)

    # 14. ESA auxiliary loss
    L_ESA = esa.compute(h_unsteered, R_const, tau_safety=0.5)

    # 15. KL + entropy
    L_KL = beta_KL * (log_probs_policy - log_probs_ref).mean()
    L_ent = -lambda_ent * entropy(policy_logits)

    # 16. Total loss
    loss = L_policy + L_KL + L_ent + alpha_ESA * L_ESA

    # 17. Optimize
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    # 18. WARP merge every 50 steps
    if step % 50 == 0:
        warp.merge_and_anchor(checkpoints, ref_model)

    return {...}
```

---

## 8. Novel Contribution Details

### 8.1 Steering-Reward Correspondence (SRC)

**Motivation:** Constitutional alignment methods (CAI, RLHF with safety reward) train the model to produce safe outputs, but don't measure whether the model has *internalized* the safety behavior or is just surface-level complying. A model that only avoids unsafe outputs because the reward penalizes them will diverge from safety when the reward signal is not present (e.g., after fine-tuning on a new task).

**Mechanism:** During GRPO sampling, we run two parallel forward passes — one with a constitutional steering vector (safety direction computed via mean-difference between safe/unsafe activation clusters), and one without. The cosine distance between the two hidden state trajectories measures how much the model's representations need to be adjusted to stay constitutionally safe.

```python
# Compute safety steering vector (once per curriculum stage)
safety_direction = extract_mean_diff_vector(
    model=policy,
    safe_activations=collect_activations(safe_prompts, layer=12),
    unsafe_activations=collect_activations(unsafe_prompts, layer=12),
)

# Per completion: SRC reward
def src_reward(y_i, prompt_ids, model, layers, safety_direction, alpha=0.3, lambda_SRC=0.1):
    h_unsteered = collect_hidden_states(model, prompt_ids, y_i, layers)
    with SteeringHook(model, layers, coeff=alpha, direction=safety_direction):
        h_steered = collect_hidden_states(model, prompt_ids, y_i, layers)
    dist = mean([cosine_dist(h_steered[l], h_unsteered[l]) for l in layers])
    return -lambda_SRC * dist
```

**Convergence property:** As training progresses and the policy internalizes safe behavior, `h_unsteered → h_steered` for safe completions, so R_SRC → 0 (no penalty). The signal becomes a gentle regularizer rather than a dominant loss term — exactly what we want in later training stages.

### 8.2 Expert Safety Affinity (ESA)

**Motivation:** Sparse MoE models have interpretable expert specialization. We can steer this specialization during alignment training to create designated "safety experts" that handle constitutionally-sensitive computations. This makes alignment more robust: even after further fine-tuning, tokens that activate safety concepts will preferentially route to safety-specialized experts, providing a structural safety buffer.

**Mechanism:** 
1. Run constitutional scorer on training completions to identify "safety-sensitive" tokens (tokens at positions where R_const drops below τ_safety)
2. Add cross-entropy routing supervision for those tokens, targeting experts {0, 1}
3. The ESA loss is small (α_ESA=0.01) so it nudges routing without overriding task performance

**Long-term effect:** After 5,000 MOSAIC steps, experts 0 and 1 will have learned representations that specialize in safety-relevant token processing. This creates a natural "safety module" within the MoE architecture.

### 8.3 Multi-Token Alignment Horizon (MTAH)

**Motivation:** Standard GRPO assigns a single sequence-level advantage score to every token in a completion. This is *myopic*: a completion could have a safe first token followed by an unsafe chain of reasoning, and the per-token advantage would be uniformly high (if the full sequence reward is high, which it may not be if rewards are sequence-level). MTAH uses Aurelius's MTP head to look 2 tokens ahead during advantage computation.

**Mechanism:**
```python
# Advantage at token t using MTP look-ahead
def mtah_advantage(adv_sequence: Tensor, mtp_adv: Tensor, gamma: float = 0.95, K: int = 2) -> Tensor:
    # adv_sequence: [n_group, T] — standard GRPO per-token advantage
    # mtp_adv: [n_group, T, K] — advantage of k-th future token per position
    extended = adv_sequence.clone()
    for k in range(1, K + 1):
        extended = extended + (gamma ** k) * mtp_adv[:, :, k-1]
    return extended
```

**Effect:** A token that is safe but leads to unsafe continuations (mtp_adv is negative for t+1, t+2) gets a lower extended advantage, discouraging that token even though the immediate constitutional score is acceptable. This enforces alignment over the full generation trajectory, not just token by token.

---

## 9. What MOSAIC v2 Does That v1 Didn't

| Dimension | AURORA v1 | MOSAIC v2 |
|-----------|-----------|-----------|
| Reward signals | 4 (quality, const, length, hier) | 6 (+ CCoT, + SRC) |
| Novel contributions | 2 (precision-weighting, gradient gating) | 5 (+ SRC, ESA, MTAH) |
| Architecture coupling | None | SRC uses latent_steering; ESA uses MoE routing; MTAH uses MTP |
| Reasoning alignment | None | CCoT + thinking-token split loss |
| Token granularity | Sequence-level advantage | Per-token credit via TokenDPO weights |
| Temporal horizon | 1 token (current) | 3 tokens (current + 2 MTP future) |
| Structural alignment | Gradient-level only | Gradient + routing specialization (ESA) |
| Self-curriculum | AbsoluteZero (deduction only) | AbsoluteZero + STILL bootstrapping + Self-Reward |
| Policy merging | WARP every 50 steps | WARP + AZ self-tasks + self-play filtering |

---

## 10. Expected Training Dynamics

**Stage 1 (ARIA, steps 0–1000):**
- SFT quality improves, constitutional violations drop from ~40% to ~15% of completions
- Thinking-token structure established; model learns to use `<think>` prefix correctly
- STILL provides high-quality bootstrap seed data from 3 iterations

**Stage 2 (AURORA, steps 1000–4000):**
- GRPO group advantages stabilize as DAPO filter removes degenerate groups
- Reward uncertainty σ²_j converges → precision weights stabilize by step 2000
- Constitutional gate fires on ~20% of completions initially, dropping to ~5% by end
- AbsoluteZero self-curriculum begins generating novel task types at step 1500

**Stage 3 (MOSAIC v2, steps 4000–9000):**
- SRC reward increases from -0.15 → -0.05 as model internalizes safe behaviors
- ESA routing specialization visible in expert activation histograms by step 6000
- MTAH advantage extends look-ahead: multi-step unsafe reasoning chains increasingly penalized
- WARP merges prevent reward hacking drift; SFT anchor μ=0.05 keeps model coherent
- Expected: eval_loss improves from 5.573 → ~4.8; constitutional violation rate < 3%

---

## 11. Budget Analysis

| Stage | Steps | Est. Hours (A100) | Est. Cost |
|-------|-------|-------------------|-----------|
| ARIA (SFT+STILL) | 1,000 | ~4h | ~$16 |
| AURORA (4-signal GRPO) | 3,000 | ~12h | ~$48 |
| MOSAIC v2 (full 6-signal + novel) | 5,000 | ~24h | ~$96 |
| **Total** | **9,000** | **~40h** | **~$160** |

Fits comfortably within the $350 total cloud budget (with $190 headroom for experiments and ablations).

---

## 12. Ablation Priority

To validate the novel contributions, run these ablations in order of cost:

1. **AURORA vs MOSAIC v2 (full)** — validates the 3 novel contributions together
2. **MOSAIC v2 - SRC** — isolates SRC contribution
3. **MOSAIC v2 - ESA** — isolates expert specialization contribution
4. **MOSAIC v2 - MTAH** — isolates temporal horizon contribution
5. **5-signal vs 6-signal** (with/without CCoT) — validates CoT as alignment signal

**Metric:** Held-out constitutional violation rate + STILL verifier score + mean sequence reward.

---

## 13. Integration Points

### Registration (add to `src/alignment/__init__.py`)
```python
from .mosaic_v2 import MOSAICv2Trainer, MOSAICv2Config
ALIGNMENT_REGISTRY["mosaic_v2"] = MOSAICv2Trainer
```

### Config activation flags (in `AureliusConfig`)
- `mtp_enabled=True, mtp_n_predict=2` — for MTAH
- `moe_enabled=True, moe_num_experts=8` — for ESA  
- `hybrid_attention_enabled=True` — for SRC at CSA/HCA layers

---

*Authored 2026-05-09. Implementation plan: see writing-plans output.*
