# Paper Ideas for Reasoning — Neural Modules, Losses & Architectures

---

## 1. STaR: Bootstrapping Reasoning With Reasoning (2203.14465)

**Key idea:** A self-taught loop where a model generates rationales for questions, retries with ground-truth hints when wrong, then fine-tunes only on rationales that led to correct answers, iteratively bootstrapping its own reasoning ability.

### Ideas

**Idea 1.1 — Rationale-Conditioned Answer Head with Filtration Gate**

- **What:** A module that learns to predict whether a generated rationale will lead to a correct answer before seeing the answer.
- **Neural mechanism:** A small binary classifier (rationale filter) that takes the hidden state at the final rationale token and predicts `p(correct | rationale)`. During training, rationales with `p < threshold` are discarded before they influence the answer head.
- **PyTorch:** `nn.Linear(d_model, 1)` with sigmoid, applied to `last_hidden_state[:, -1, :]` of the rationale segment. Loss: `BCEWithLogitsLoss` against ground-truth correctness labels.
- **Integration:** Wraps any decoder-only LM. Runs after rationale generation, gates whether the rationale passes to the answer decoder. Connected to the STaR retry loop — when the filter predicts wrong, skip to the "hint mode" immediately.

**Idea 1.2 — Hint-Conditioned Rationale Regenerator**

- **What:** A separate adapter module that re-encodes the correct answer as a "hint" and re-generates a rationale conditioned on both the question and the answer.
- **Neural mechanism:** A cross-attention layer between the answer embedding and the question's hidden states. Concatenate the answer embedding into the decoder input at a special `<hint>` token position.
- **PyTorch:** `nn.MultiheadAttention(cross_attn)` in a small transformer block inserted after the main LM's last layer. The query is the question representation, key/value are the answer embedding.
- **Loss:** Standard language modeling loss (CrossEntropy) on the regenerated rationale tokens, but only backpropagated through the adapter (frozen LM backbone).
- **Integration:** Activated only during the retry phase (STaR's "hint mode"). The regenerated rationale replaces the original for the fine-tuning step.

**Idea 1.3 — Rationale-Utility Weighted Fine-Tuning**

- **What:** Instead of binary keep/discard, assign a continuous weight to each rationale proportional to how much it improved answer correctness over the direct-answer baseline.
- **Neural mechanism:** A utility scorer that compares the log-probability of the correct answer under the rationale-conditioned model vs. the no-rationale baseline. This score becomes a per-sample loss weight.
- **PyTorch:** No extra parameters. Compute `weight = exp(log_p_correct_with_rationale - log_p_correct_without)`. Then `loss = weight * CrossEntropy(rationale_tokens, ground_truth_rationale)`.
- **Loss:** Weighted CrossEntropy, where weight is clamped to `[0.1, 5.0]` to prevent outliers.
- **Integration:** Applied during the fine-tuning phase of STaR. The baseline log-probs come from a frozen copy of the model before the STaR iteration.

---

## 2. Tree of Thoughts (2305.10601)

**Key idea:** Instead of a single chain of reasoning, the model explores a tree of multiple candidate "thoughts" at each step, uses self-evaluation to prune/select branches, and uses BFS/DFS for deliberate search.

### Ideas

**Idea 2.1 — Thought-State Value Head (Critic Module)**

- **What:** A learned evaluator that scores the promise of a partial reasoning state (thought), replacing the paper's LM-prompted self-evaluation.
- **Neural mechanism:** A transformer encoder head that takes the sequence of thought tokens so far and outputs a scalar `V(thought_state)` estimating the probability this branch leads to a correct solution.
- **PyTorch:** `nn.TransformerEncoderLayer` stack (2-4 layers) + `nn.Linear(d_model, 1)`. Input: embedded thought tokens. Output: scalar value.
- **Loss:** MSE against the final outcome (1.0 if branch ultimately yields correct answer, 0.0 otherwise). Train using Monte Carlo rollouts.
- **Integration:** Wraps the LM. At each tree node, the LM generates K candidate thoughts. The value head scores all K. Only top-M by value are expanded. Enables BFS/DFS pruning without calling the LLM for self-evaluation.

**Idea 2.2 — Differentiable Tree Expansion with Gumbel-Softmax Branch Selection**

- **What:** Make the tree expansion step differentiable by using Gumbel-Softmax to select which branches to explore, allowing end-to-end training of the thought generator.
- **Neural mechanism:** The thought generator produces K candidate next-thought embeddings. A learnable router (linear projection + Gumbel-Softmax) selects a weighted combination of candidates (top-2), which is fed into the next step's decoder.
- **PyTorch:** `F.gumbel_softmax(logits, tau=1.0, hard=False)` on router logits. Weighted sum of candidate embeddings: `selected = sum(weights[i] * candidate_embeds[i] for i in top_2)`.
- **Loss:** CrossEntropy on final answer + auxiliary loss that penalizes the router for collapsing to a single branch (entropy regularization).
- **Integration:** Replaces the discrete BFS/DFS with a soft, differentiable tree. The Gumbel temperature is annealed during training toward hard selection.

**Idea 2.3 — Backtracking Trigger Network**

- **What:** A lightweight binary classifier that decides when to backtrack (discard current branch and revisit a previous node), automating the paper's manual backtracking.
- **Neural mechanism:** Takes the last N thoughts' hidden states, aggregates via mean pooling, and feeds into an MLP with sigmoid output. When the signal crosses a threshold, the tree search pops back to the previous node.
- **PyTorch:** `nn.Linear(d_model * N, 256) -> nn.ReLU() -> nn.Linear(256, 1) -> nn.Sigmoid()`. Trained with positive labels from rollout data where backtracking preceded a successful solution.
- **Loss:** BCEWithLogitsLoss.
- **Integration:** Connected to a search manager that maintains a stack of tree nodes. When trigger fires, the LM's hidden state is reverted to the parent node's state.

---

## 3. Program of Thoughts (2211.12588)

**Key idea:** Instead of using the LM for both reasoning and computation, express reasoning as executable Python code and offload computation to an external runtime, achieving near-perfect arithmetic accuracy.

### Ideas

**Idea 3.1 — Neural Code-Generation Head with Variable Trace**

- **What:** A decoder module that generates Python code step-by-step while maintaining a latent "variable trace" — a learned embedding of all intermediate variable values.
- **Neural mechanism:** Augment each token's hidden state with a differentiable memory of past variable assignments. After each line of generated code, an execution simulator updates the variable trace via a learned write operation.
- **PyTorch:** `nn.Linear(d_model, d_vocab)` for code tokens. A separate `VariableTrace` module: `nn.GRUCell(d_model, d_trace)` that ingests the line's hidden state and updates a trace vector. The trace vector is appended to the next token's embedding.
- **Loss:** CrossEntropy on generated code tokens + auxiliary loss predicting the output of each line (supervised by executing the code externally).
- **Integration:** The trace vector feeds as a prefix to each generated code line. During inference, the actual Python executor replaces the trace module.

**Idea 3.2 — Differentiable Python Interpreter Proxy**

- **What:** A neural network that approximates the execution of simple arithmetic and string operations so gradient can flow through the computation, enabling end-to-end training.
- **Neural mechanism:** A small graph neural network where each operation (+, -, *, /, len, etc.) is a learned "operation module" that takes input tensors and produces an output tensor of the same dimensionality.
- **PyTorch:** A dictionary of `nn.Module` entries, one per supported operation. `ops['+'] = AddModule(nn.Linear(d, d))`, `ops['*'] = MulModule(nn.Linear(d, d))`, etc. The code is parsed into an AST, and the GNN executes it token by token.
- **Loss:** MSE between the proxy's output and the actual Python output.
- **Integration:** During training, the proxy stands in for the external Python runtime so gradients flow from answer loss back through the generated code. At inference, the proxy is discarded and real Python is used.

**Idea 3.3 — Code-Rationale Dual Output Head**

- **What:** A single model that produces both a natural-language rationale and the equivalent Python program simultaneously, sharing the backbone but using separate heads.
- **Neural mechanism:** Two parallel LM heads after the shared transformer backbone. One head produces text tokens (rationale), the other produces code tokens (program). A learned gating mechanism decides per reasoning step which head to use.
- **PyTorch:** Two `nn.Linear(d_model, vocab_size)` heads. A gate: `nn.Linear(d_model, 2)` with softmax. The final token distribution is `gate_text * text_logits + gate_code * code_logits`.
- **Loss:** Combined CrossEntropy on both rationale and program tokens.
- **Integration:** The dual output allows the model to "think" in natural language when reasoning is qualitative and switch to code when arithmetic precision is needed.

---

## 4. Voyager: Open-Ended Embodied Agent (2305.16291)

**Key idea:** A lifelong agent in Minecraft that uses GPT-4 to propose exploration goals, generates/grows a skill library of executable code via iterative refinement, and self-verifies through environment feedback.

### Ideas

**Idea 4.1 — Differentiable Skill Library with Retrieval-Augmented Skill Encoder**

- **What:** A vector store of skill embeddings where each skill is a sequence of embedding vectors (from a learned skill encoder). Skills are retrieved via attention over the current observation.
- **Neural mechanism:** A dual-encoder setup: (a) a "skill encoder" that encodes each skill's code tokens into a single embedding via mean pooling over a small transformer, (b) an "observation encoder" that encodes the current state. Retrieved skills are concatenated as cross-attention context.
- **PyTorch:** `skill_encoder = nn.TransformerEncoder(...)`; `obs_encoder = nn.TransformerEncoder(...)`; similarity = `torch.matmul(obs_embed, skill_matrix.T)`. Top-K skill embeddings attend into the action decoder via `nn.MultiheadAttention`.
- **Loss:** Contrastive loss (InfoNCE): positive = skill that was actually used, negatives = random skills from library.
- **Integration:** The action decoder (another transformer) receives [obs_embed; retrieved_skill_embeds] as context. The skill library grows dynamically — new skills are encoded and added to the matrix.

**Idea 4.2 — Iterative Self-Correction Loop with Execution Trace Critic**

- **What:** A module that consumes environment feedback (execution errors, observation deltas) and proposes edits to the last generated program, replacing Voyager's GPT-4 blackbox queries with a learned corrector.
- **Neural mechanism:** A "diff" transformer that takes (incorrect_program_embedding, error_message_embedding, observation_before, observation_after) and generates a sequence of edit operations (keep, delete, insert, replace) applied to the program tokens.
- **PyTorch:** A sequence-to-sequence transformer. Input: concatenation of error embed + program embed + obs diff. Output: edit operations per position. Edit ops are produced as discrete tokens via `nn.Linear(d_model, 4)` (4 edit types).
- **Loss:** CrossEntropy on edit operation tokens + auxiliary loss predicting which line contains the error (binary classification per line).
- **Integration:** After each program execution attempt, the critic compares expected vs actual observation. If mismatch, the corrector edits the program and re-executes. This loop repeats until success or max iterations.

**Idea 4.3 — Exploration Curriculum via Intrinsic Motivation Module**

- **What:** A neural module that computes an exploration bonus for each possible next task, driving the agent toward novel states, replacing Voyager's hand-crafted curriculum.
- **Neural mechanism:** A random network distillation (RND) approach: two networks — a fixed random target `f(obs)` and a trainable predictor `g(obs)`. The exploration bonus is `||f(obs) - g(obs)||^2`, which is high for novel states.
- **PyTorch:** `target = nn.Sequential(nn.Linear(d_obs, 512), nn.ReLU(), nn.Linear(512, 512))` (frozen). `predictor = nn.Sequential(nn.Linear(d_obs, 512), nn.ReLU(), nn.Linear(512, 512))` (trainable).
- **Loss:** MSE between predictor and target on visited states. Bonus = sqrt(MSE).
- **Integration:** The task proposer (a small LM) receives a prompt that includes the current exploration bonus for candidate tasks. Tasks with high bonus are prioritized. The predictor is trained online as the agent explores.

---

## 5. Quiet-STaR (2403.09629)

**Key idea:** A generalization of STaR where the LM learns to generate internal "thoughts" (rationales) at every token position during pretraining on raw text, improving next-token prediction and zero-shot reasoning without task-specific fine-tuning.

### Ideas

**Idea 5.1 — Learnable Thought Tokens (<start_thought> and <end_thought>)**

- **What:** Two special learnable embeddings that the model inserts into its own sequence to mark where thinking begins and ends. The model learns when to start and stop thinking.
- **Neural mechanism:** Free-standing embedding vectors of dimension `d_model`. During forward pass, at each token position, a binary "think/no-think" gating network decides whether to insert a thought. The gate is trained with REINFORCE.
- **PyTorch:** `start_thought = nn.Parameter(torch.randn(d_model))`; `end_thought = nn.Parameter(torch.randn(d_model))`. Gate: `nn.Linear(d_model, 2)` with softmax and Gumbel-Softmax sampling.
- **Loss:** REINFORCE with reward = reduction in next-token perplexity when thought tokens are included. Baseline: perplexity without thoughts. The thought tokens themselves are trained with the LM's autoregressive loss.
- **Integration:** Inserted at the sequence level before the LM forward pass. The rest of the model sees these as normal tokens.

**Idea 5.2 — Tokenwise Parallel Thought Sampler**

- **What:** A parallel sampling algorithm that generates K thought continuations at each token position simultaneously, then selects the best one, avoiding the sequential bottleneck of generating one thought at a time.
- **Neural mechanism:** Expand the batch dimension by a factor of K at each thought-start position. Generate K independent thought continuations of length L in parallel. Score each with the value head (2.1) and select the top-1.
- **PyTorch:** At thought-start token, repeat the hidden state K times: `hidden = hidden.unsqueeze(1).expand(-1, K, -1)`. Run the transformer with batch dimension `B*K`. After L tokens, compute scores and pick: `best_idx = scores.argmax(dim=1)`, then gather.
- **Loss:** Same as Quiet-STaR (REINFORCE + LM loss). The parallel selection uses the value head scores as the REINFORCE reward.
- **Integration:** A wrapper around the main transformer `forward()`. When `do_sample_thoughts=True`, it executes the parallel expansion. Memory usage increases by K× during thought generation.

**Idea 5.3 — Extended Teacher Forcing for Rationale-Augmented Text**

- **What:** A training technique where the model sees the original text plus inserted thoughts during training, but the loss is only computed on the original text tokens (not the thought tokens), forcing the thoughts to earn their keep by improving future token predictions.
- **Neural mechanism:** Create a modified sequence: `[t1, t2, <start_thought>, thought_tokens, <end_thought>, t3, t4, ...]`. Run full forward pass. Mask out thought tokens from the loss computation.
- **PyTorch:** `loss_mask = torch.ones_like(labels)`; set `loss_mask[thought_positions] = 0`. `loss = CrossEntropy(logits.view(-1, vocab_size), labels.view(-1), reduction='none') * loss_mask.view(-1)`. Only the original-text cross-entropy contributes.
- **Loss:** Masked CrossEntropy on original tokens only. The thought tokens are trained only via the REINFORCE signal (reduction in future token perplexity).
- **Integration:** In the training loop, the sequence builder module inserts thoughts at positions selected by the gating network (5.1). The loss mask is constructed automatically.

**Idea 5.4 — Thinking-Progress-Encoding (TPE) Positional Bias**

- **What:** A positional encoding adjustment that tells the model how many "thinking steps" have been taken so far, separate from the token position.
- **Neural mechanism:** An additional learned positional embedding indexed by thought-step-count (not token index). Added to the standard positional encoding as a bias term.
- **PyTorch:** `think_pos_enc = nn.Embedding(max_thought_steps, d_model)`. During forward, each token at thought-step `s` receives `x = x + think_pos_enc(s)`. For non-thought tokens, `s = 0`.
- **Loss:** No separate loss — trained via gradients flowing through the main LM objective.
- **Integration:** Added to the transformer input embeddings before the first layer. The thought-step counter resets after each `<end_thought>` token.

---

## Cross-Paper Integration Map

| Module | Source Paper | Connects To |
|--------|------------|-------------|
| Value Head (thought-state scoring) | ToT 2.1 | Quiet-STaR 5.2 (parallel sampler selection), STaR 1.1 (rationale filtration) |
| Differentiable Program Execution | PoT 3.2 | Voyager 4.2 (execution trace critic) |
| Learnable Thought Tokens | Quiet-STaR 5.1 | STaR 1.2 (hint token extension) |
| Skill Retrieval (dual encoder) | Voyager 4.1 | PoT 3.3 (code-rationale head could retrieve relevant code snippets) |
| REINFORCE for Thought Placement | Quiet-STaR 5.1 | ToT 2.3 (backtracking trigger training) |
| Utility-Weighted Fine-Tuning | STaR 1.3 | Quiet-STaR 5.3 (extended teacher forcing weight scheme) |

### Suggested Build Order

1. Start with **STaR 1.1 + 1.3** (rationale filter + weighted finetuning) — simplest, requires only a classifier head and loss weighting.
2. Add **Quiet-STaR 5.1 + 5.3** (learned thought tokens with teacher forcing) — generalizes to any text.
3. Layer in **ToT 2.1** (value head) for search-based decision making.
4. Integrate **PoT 3.1** (code generation head) for numerical robustness.
5. Top off with **Voyager 4.1 + 4.3** (skill library + exploration curriculum) for lifelong agents.
