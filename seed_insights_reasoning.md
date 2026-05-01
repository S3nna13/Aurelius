# Seed Insights: Reasoning Papers → Neural Brain Module

---

## Paper 1: Chain-of-Thought Prompting (Wei et al., 2022)

### Core Algorithm

Generate intermediate reasoning steps sequentially before producing the final answer. Each step decomposes the problem into a smaller sub-problem, and the reasoning chain is autoregressively generated. This elicits complex multi-step reasoning in language models without explicit training — the ability emerges at sufficient scale.

### Neural Module Implementation

Replace prompting with a dedicated **Latent Reasoning Chunking Layer** that sits between encoder and decoder. Rather than generating text tokens as intermediate steps, the network produces a sequence of latent reasoning vectors — each one a compressed sub-goal representation that refines the query-to-answer mapping.

### Concrete Implementation Features

1. **Step-wise Latent Decomposition Head**: A learned module that takes the query encoding and emits N latent sub-goal vectors autoregressively. Each vector is produced by a small Transformer block that conditions on the query and all prior sub-goal vectors, then feeds into the next step. This is not text — it is a learned latent decomposition.

2. **Reasoning Path Gating**: A binary gate per step that determines whether to emit a latent reasoning vector or to jump directly to answer generation. The gate is trained via a learned auxiliary loss that maximizes final-correctness while minimizing unnecessary steps. Emergent behavior: harder queries get more steps.

3. **Cross-Step Attention Bridge**: Each reasoning step attends to the original query AND all prior steps via causal attention. This prevents the vanishing-gradient problem of long reasoning chains. Implement as a set of cross-attention heads in the reasoning chunker that are distinct from the decoder cross-attention.

4. **Verification Skip-Connection**: After the reasoning chain, a parallel verification pathway computes a confidence score. If confidence is below threshold, the network loops back and regenerates the reasoning chain from a different latent initialization (beam-search in latent space).

5. **Progressive Answer Refinement**: The decoder receives both the query and the full reasoning chain. Each step of reasoning injects a learned bias term into the decoder's layer norms, allowing the decoder to shift its decision boundary as reasoning progresses.

### Training Methodology

- Train end-to-end with maximum likelihood on problems that have ground-truth step-by-step solutions (e.g., math word problems with solution traces, multi-hop QA with evidence chains).
- Add a **step-count regularization loss** that penalizes unnecessarily long chains (L1 penalty on the number of reasoning steps taken).
- For problems without step annotations, use the verification skip-connection as a self-supervised signal: generate reasoning chain, compute answer confidence, and backprop through successful chains only (reward-conditioned training).
- Curriculum: start with 1-step problems, progressively increase required steps.

---

## Paper 2: ReAct — Synergizing Reasoning and Acting (Yao et al., 2022)

### Core Algorithm

Interleave reasoning traces with environment actions. At each step, the agent can either emit a reasoning thought (internal cognition) or an action (external interaction). Actions query external sources (e.g., Wikipedia API) and the results feed back into the reasoning loop. This prevents hallucination by grounding reasoning in observed facts, and prevents aimless exploration by using reasoning to plan actions.

### Neural Module Implementation

Build a **Think-Act Dual-Stream Architecture** where two parallel pathways share a common encoder but route through different decoders. A learned router decides which stream activates at each timestep, and the observation stream injects external data back into the shared latent state.

### Concrete Implementation Features

1. **Dynamic Router Network**: A lightweight feed-forward classifier that takes the current hidden state and decides among three actions: (a) emit a reasoning step, (b) emit an environment action, or (c) read an observation. Trained via a combination of supervised learning (on trajectories with annotated "think" vs "act" steps) and reinforcement learning (reward from task completion).

2. **Action Projection Layer**: A learnable linear projection that maps latent states into a structured action space (e.g., API call templates). The output is a discrete token sequence representing the action, but the projection ensures the action is syntactically valid. This replaces hand-written parsing.

3. **Observation Injection Gate**: After an environment action, the observation result is encoded through a small observation encoder and injected into the hidden state via a learned gating mechanism (similar to a GRU reset gate). This lets the network decide how much of the observation to absorb vs. ignore.

4. **Belief State Correction Module**: When an observation contradicts the current reasoning trace (e.g., agent thought an object was in location A but observation says it is not), a dedicated correction sub-network adjusts all prior reasoning step representations via a backward pass through the reasoning chain. Implemented as a lightweight gradient-correction step applied at inference time.

5. **Interleaved Trajectory Memory**: A sliding-window attention mechanism that spans the full think-act-obs sequence. Unlike standard causal attention, this uses relative position encodings that distinguish thought tokens from action tokens from observation tokens, allowing the network to learn different attention patterns for each modality.

### Training Methodology

- Collect or synthetically generate trajectories with interleaved thought/action/observation steps. Annotate each step with its type (think/act/obs).
- Multi-task loss: (a) next-token prediction loss, (b) router classification loss (supervised on step types), (c) action validity loss (encourage syntactically valid actions), (d) environment reward (RL loss on task completion).
- Use behavior cloning from expert trajectories, then fine-tune with policy gradient (REINFORCE) on environment reward.
- Curriculum: first train on short 2-3 step think-act-obs cycles, then extend to long-horizon tasks.

---

## Paper 3: Reflexion — Verbal Reinforcement Learning (Shinn et al., 2023)

### Core Algorithm

After each trial, the agent reflects on its trajectory and environmental feedback, generating a natural-language summary of what went wrong and how to improve. This summary is stored in an episodic memory buffer. On subsequent trials, the agent conditions its decisions on both the current state and all prior reflections. This provides a semantic gradient signal without weight updates.

### Neural Module Implementation

Implement an **Episodic Self-Reflection Loop** with three specialized sub-networks: Actor (policy), Evaluator (critic), and Reflector (memory encoder). The Reflector converts trajectory outcomes into compact reflection vectors stored in a differentiable episodic memory bank, which the Actor queries during inference.

### Concrete Implementation Features

1. **Reflection Encoder**: A Transformer encoder that takes the full trajectory embedding plus the evaluator score and produces a fixed-dimension **reflection vector**. This vector encodes the key lessons: what mistake was made, at which step, and what corrective action should be taken. The encoder is trained to maximize the mutual information between the reflection vector and the improvement in the next trial's reward.

2. **Differentiable Episodic Memory Bank**: A set of learned memory slots (e.g., 128 vectors of dimension d) that store reflection vectors. A **memory controller** writes new reflections via a gated write mechanism and reads relevant memories via attention over slots (keys = query encoding, values = reflection vectors). Memory slots are updated via gradient descent during training, not just appended.

3. **Evaluator Network**: A learned reward model that takes the trajectory and outputs both a scalar reward estimate and a structured error vector (which issues were encountered). The error vector is concatenated with the trajectory encoding before being fed to the Reflection Encoder. Train the evaluator via contrastive learning: trajectories that lead to success vs. failure.

4. **Reflection-Conditioned Policy Head**: The Actor's policy head receives the current state encoding plus the weighted sum of retrieved memory vectors. This conditions action selection on accumulated experience. The memory attention weights are learned jointly with the policy — the network learns to ignore irrelevant reflections and focus on useful ones.

5. **Trial-Level Termination Gate**: A learned binary classifier that predicts whether the current trajectory quality is sufficient to stop trying. If the Evaluator's confidence exceeds a learned threshold, the network exits the trial loop early. This provides computational efficiency while preserving the iterative improvement mechanism.

### Training Methodology

- Outer loop: train all modules jointly via a variant of actor-critic where the "critic" is the Evaluator and the "experience replay" is the episodic memory bank.
- Inner loop (inference-time): run N trials per problem. Between trials, the Reflection Encoder writes to memory, and the Actor re-conditions on updated memory.
- Loss functions: (a) Actor: policy gradient on task reward, (b) Evaluator: MSE between predicted and actual reward, (c) Reflection Encoder: contrastive loss that maximizes the improvement in reward from trial t to trial t+1 conditioned on the reflection, (d) Memory controller: gating sparsity loss to encourage compact memory usage.
- Interleave training between single-trial episodes (for standard supervised learning) and multi-trial episodes (for the Reflexion loop).
- Hard-negative mining: during training, curate cases where the first trial fails but the second succeeds — these provide the strongest learning signal for the Reflection Encoder.

---

## Cross-Cutting Synthesis: Combined Module Architecture

These three mechanisms can be composed into a unified reasoning module:

```
Input Query
    |
    v
[CoT Latent Decomposition] -- produces N latent reasoning steps
    |
    v
[ReAct Think-Act Router] -- interleaves reasoning steps with environment actions/observations
    |
    v
[Reflexion Episodic Loop] -- iterates across trials, storing and retrieving reflections
    |
    v
Final Output
```

The router from ReAct gates whether the current step is internal reasoning (CoT pathway) or external action (API call). After each full trajectory, the Reflexion loop evaluates quality, writes reflections to episodic memory, and re-runs with updated context. The entire module is trained end-to-end with the composite loss: decomposition correctness + routing accuracy + trial improvement.
