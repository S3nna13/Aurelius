# Seed Insights: Training & Alignment Architectures

## 1. Direct Preference Optimization (DPO) — arxiv:2305.18290

### Core Mechanism
DPO eliminates the separate reward model + RL loop by reparameterizing the RLHF objective. The key insight: the optimal policy under KL-constrained reward maximization has a closed form, and the Bradley-Terry preference model can be expressed directly in terms of the policy `π_θ` and reference policy `π_ref`. This maps preference learning onto a binary classification loss over policy log-probability ratios.

### Loss Function
```
ℒ_DPO(π_θ; π_ref) = -E[ log σ( β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x) ) ]
```
The implicit reward is `r̂_θ(x,y) = β log π_θ(y|x)/π_ref(y|x)`. The gradient dynamically weights each example by `σ(r̂_θ(x,y_l) - r̂_θ(x,y_w))` — higher weight when the implicit reward ranks the dispreferred response higher.

### Implementation Features for Neural Training Module

1. **Implicit Reward from Policy Logits** — During training, maintain both the active policy `π_θ` and a frozen reference `π_ref`. The reward signal is computed as `β * (logits_θ - logits_ref)`, avoiding a separate reward model forward pass.

2. **Per-Example Dynamic Weighting** — The sigmoid weight `σ(r̂_l - r̂_w)` acts as an adaptive importance scalar. Implement as a stop-gradient tensor that gates the gradient update, preventing degeneration from naive unlikelihood.

3. **KL-Constraint via Reference Anchor** — The `β` parameter directly controls the reverse KL divergence from the reference model. Unlike PPO's clipped surrogate, this is built into the loss algebraically — no separate KL penalty term or value function needed.

4. **Offline Preference Dataset Training** — DPO trains entirely offline on static preference pairs `(x, y_w, y_l)`. No online sampling from the LM during training. This enables batching over pre-collected human preferences without an RL rollout loop.

5. **Bradley-Terry / Plackett-Luce Extensibility** — The loss generalizes from pairwise comparisons to ranked lists via the Plackett-Luce model. Implement a multi-item variant for k-wise comparisons: `ℒ = -E[ log( exp(β log π_θ(y_1|x)/π_ref(y_1|x)) / Σ_j exp(β log π_θ(y_j|x)/π_ref(y_j|x)) ) ]`.

---

## 2. Constitutional AI (CAI) — arxiv:2212.08073

### Core Mechanism
Two-phase training (SL + RL) using a constitution — a short list of principles/rules — instead of human harmlessness labels. Phase 1 (Supervised): sample from model, generate self-critique and revision guided by constitutional principles, then supervised fine-tune on revised responses. Phase 2 (RL / RLAIF): generate two responses from fine-tuned model, use a model to evaluate which better adheres to constitution, train a preference model from AI-generated preferences, then RL with that preference model as reward.

### Loss & Reward Structure
- **SL Phase**: Standard cross-entropy loss on revised (constitutionally-aligned) responses. No explicit reward.
- **RL Phase (RLAIF)**: Standard preference model loss: `ℒ_R = -E[log σ(r_φ(x,y_w) - r_φ(x,y_l))]` where preferences are AI-labeled using constitutional principles, then PPO on `r_φ` with KL penalty.
- **Chain-of-Thought Augmentation**: Both phases can be augmented with chain-of-thought reasoning that explains why a response violates principles before critiquing or scoring.

### Implementation Features for Neural Training Module

1. **Constitutional Principle Embeddings Bank** — Store a learnable embedding for each constitutional principle. During self-critique, attend over principle embeddings to produce a principle-conditioned critique vector. This makes the constitution differentiable and allows soft weighting of violations.

2. **Self-Critique with Rejection Sampling** — After generating an initial response, run a critique LM conditioned on `(response, principle_embedding)` to produce a natural language critique. Use rejection sampling: keep the critique + revision pair only if the critique model's confidence exceeds a threshold. The revised response becomes the SL target.

3. **Dual-Phase Training Controller** — Loss function automatically switches between SL (cross-entropy on revisions) and RL (PPO on preference model). Phase transition occurs when SL loss plateaus or after a fixed number of constitution-violation-reduction evaluations.

4. **AI Feedback Scorer Module** — A separate scorer model (typically 1-2 orders of magnitude smaller) that evaluates `(response, constitution_principle)` pairs and outputs a harmlessness score. This replaces the need for human preference labels in the RL phase. Train this scorer via classification on constitution-adherence judgments.

5. **Chain-of-Thought Latent Variable Integration** — During the RL phase, sample a CoT trajectory from the policy that explains the harmlessness evaluation before the response. Backprop through this CoT path allows the model to learn reasoning over constitutional constraints jointly with response generation.

---

## 3. InstructGPT (RLHF) — arxiv:2203.02155

### Core Mechanism
Three-phase pipeline: (1) Supervised fine-tuning (SFT) on human-written demonstrations. (2) Reward model (RM) training using human comparisons of model outputs under the Bradley-Terry preference model. (3) PPO-based RL fine-tuning to maximize RM reward while constraining KL divergence from the SFT model. The reward function used during PPO is: `r(x,y) = r_φ(x,y) - β(log π_θ(y|x) - log π_ref(y|x))`.

### Loss & Reward Structure
- **SFT**: `ℒ_SFT = -E[log π_SFT(y_demo|x)]` — cross-entropy on human demonstrations.
- **RM**: `ℒ_R(r_φ) = -E[log σ(r_φ(x,y_w) - r_φ(x,y_l))]` — Bradley-Terry preference loss.
- **PPO-RL**: Maximize `E[r(x,y)] - β * KL(π_θ || π_ref)` with clipped surrogate objective.

### Implementation Features for Neural Training Module

1. **Shared Transformer Backbone with Reward Head** — Initialize reward model from the SFT model by adding a linear projection head on top of the final hidden state. The backbone is frozen during RM training; only the head and optionally the last transformer layer are fine-tuned to prevent catastrophic forgetting.

2. **PPO with Adaptive KL Controller** — Implement an adaptive `β` coefficient that adjusts based on observed KL: if `KL > KL_target * 1.5`, increase `β`; if `KL < KL_target / 1.5`, decrease `β`. This stabilizes training across diverse prompts by dynamically tightening the constraint.

3. **Reward Normalization with Moving Statistics** — Maintain running mean/std of reward values across the batch. Normalize rewards before computing PPO advantage: `r_norm = (r - μ_r) / σ_r`. This ensures consistent gradient scales when the reward distribution shifts during training.

4. **Per-Token Value Function (Critic)** — Instead of a single sequence-level value, compute a per-token value head emitting a scalar for each position. The advantage is computed as `A_t = R_t - V_t` where `R_t` is the Monte Carlo return from position t. This gives the actor finer-grained gradient signals.

5. **Mixed-Preference Reward Ensemble** — Train an ensemble of 3-5 reward models from different initializations. During PPO, use the minimum reward across the ensemble as the reward signal. This prevents the policy from over-optimizing spurious features present in a single reward model (reward hacking mitigation).

---

## Comparison Table: Training Loop Integration with Reasoning

| Aspect | DPO | CAI | InstructGPT |
|---|---|---|---|
| **Training loop** | Single-phase classification | Two-phase (SL → RL) | Three-phase (SFT → RM → RL) |
| **Reward source** | Implicit from log-ratio | AI-labeled preferences | Human-labeled preferences |
| **RL required?** | No | Yes (PPO + RLAIF) | Yes (PPO) |
| **KL mechanism** | Built into loss via β | External KL penalty | Adaptive β controller |
| **CoT integration** | None needed | CoT for critique & revision | None |
| **Sample complexity** | Lowest (offline only) | Medium (self-generated) | Highest (human labels) |
| **Differentiability** | Fully differentiable | Partially (RL samples) | Partially (RL samples) |
