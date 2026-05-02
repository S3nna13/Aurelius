# Paper Ideas: Alignment & Safety Modules

---

## 1. Direct Preference Optimization (DPO) — arXiv:2305.18290

**Core idea:** Reparameterize the RLHF reward function so the optimal policy can be extracted in closed form, replacing PPO with a simple binary classification loss over preference pairs.

### Implementable Ideas

**1a. Closed-form DPO loss for verifier integration**
- Replace the Bradley-Terry reward model with the policy itself: `reward = β log(π_θ(y|x) / π_ref(y|x))`
- Loss: `L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]`
- Integrate with existing verifier by treating the verifier score as `reward` and training a lightweight π_θ head to match it via DPO rather than RL
- Saves: no reward model training, no RL sampling loop, no value network

**1b. Reference-policy cached log-prob head**
- During DPO, keep π_ref frozen and cache its log-probs. Attach a tiny MLP head to the base LM that predicts `log π_θ - log π_ref` offset directly
- This lets the base LM be any pretrained model; the DPO head is a small additive adapter (LoRA-style)
- Inference: drop the head, or fold the offset into the LM logits via a learned bias vector

**1c. DPO as a critic-finetuning stage**
- After a reasoning model produces candidate answers, use DPO on preference pairs (verifier-approved vs verifier-rejected) to nudge the generator
- The "reward" β(log π_θ - log π_ref) is computed per-token, enabling token-level credit assignment without RL
- Combine with MCTS rollouts: pairs are (winning_path, losing_path) from the tree

**1d. Iterative DPO for self-play**
- At each iteration, sample from current π_θ, have verifier score them, build new preference pairs from the top/bottom quantiles, then DPO-update
- No human labels needed after bootstrap — the verifier acts as the preference oracle

---

## 2. Constitutional AI (CAI) — arXiv:2212.08073

**Core idea:** Use a list of constitution principles to generate self-critiques and AI feedback, replacing human preference labels with model-generated supervision in both SL and RL phases.

### Implementable Ideas

**2a. Self-critique + revision finetuning (SL phase)**
- Given a prompt, sample response from model, then prompt the same model to critique its own response using a constitution principle, then revise
- Finetune on the revised response only
- Integration: plug into existing generator as a post-hoc refinement loop. The constitution is a set of prompts (e.g., "Did this response contain unsupported claims?")
- Implementation: use a separate "critic" LM call in the training loop, or train a dedicated critique head

**2b. RLAIF preference model from model judgments**
- Sample 2 responses from the finetuned model, use a judge LM with chain-of-thought to decide which is better per constitution principle
- Train a preference model (classifier) on these AI-labeled pairs, then use it as the reward signal for RL
- Architecture: preference model = shared LM backbone + learned scalar head `r(x, y) = w^T · h_[CLS]`
- Replaces human annotators entirely for preference data

**2c. Chain-of-thought constitution judge**
- Before outputting a preference judgment, the judge LM generates a CoT reasoning trace explaining which response better adheres to the constitution
- The CoT trace can be distilled into a smaller classifier or used as training data for an interpretable rule-based judge
- Integrate with the verifier: the verifier's factuality score is one constitution principle among many

**2d. Constitutional sampling filter**
- At inference, sample N responses, have a lightweight judge model score each against N constitution principles, rerank by aggregate score
- The judge can be a small BERT-based classifier finetuned on constitution-adherence data
- This is a drop-in module before the final output

---

## 3. TruthfulQA — arXiv:2109.07958

**Core idea:** Benchmark revealing that larger LMs are *less* truthful because they better mimic human falsehoods from training data; proposes that imitation objectives alone are insufficient for truthfulness.

### Implementable Ideas

**3a. Anti-falsehood training via contrastive pairs**
- Build a dataset of (false_question, true_answer) paired with (false_question, false_answer) using TruthfulQA and similar benchmarks
- Train a contrastive loss: `L = max(0, margin - score(true) + score(false))` where score is the LM's log-prob of the answer
- This directly penalizes the model for assigning high probability to known falsehoods
- Integrate with the verifier: the verifier identifies false claims in the model's output, which become the negative examples

**3b. Truthfulness probe + rejection head**
- Train a linear probe on the LM's hidden states to predict "is this claim truthful?" (binary)
- At inference, if the probe fires below threshold, trigger a revision pass or refuse to answer
- Architecture: probe = `p(truthful | h_last) = σ(w^T h_last + b)`, trained on hidden states from TruthfulQA questions
- The probe acts as a lightweight truthfulness verifier with near-zero latency overhead

**3c. MCQA calibration for generation**
- TruthfulQA's multiple-choice format can calibrate the LM's open-generation truthfulness
- Train the model to produce higher log-probs for the correct MC answer than the incorrect ones using margin ranking loss
- Then during generation, the model implicitly scores candidate answers lower if they align with known false MC choices
- Implementation: add a ranking head that scores full generations against the MC calibration set

**3d. "Know-what-you-don't-know" refusal head**
- Train a classifier on the model's own uncertainty embeddings to detect when it's about to hallucinate
- Features: entropy of next-token distribution, attention entropy on context tokens, probe score
- If `hallucination_risk > threshold`, route to "I don't know" response or retrieval augmentation
- Direct integration with the verifier: low verifier confidence triggers the same refusal path

---

## 4. RARR (Researching and Revising) — arXiv:2210.08726

**Core idea:** Post-hoc attribution and revision system that searches the web for evidence supporting/contradicting an LM's claims, then edits the output to remove unsupported content while preserving original meaning.

### Implementable Ideas

**4a. Claim decomposition + web search loop**
- Parse LM output into atomic claims (using an LM prompt: "Break this into individual factual claims")
- For each claim, issue a web search query, retrieve top-k passages
- Use an entailment model (e.g., DeBERTa finetuned on NLI) to classify each claim as SUPPORTED / CONTRADICTED / UNVERIFIED given retrieved passages
- Integration: wraps any generator as a post-processing step

**4b. Attributed claim classifier architecture**
- Classifier input: `[CLS] claim [SEP] passage [SEP]` → entailment label
- Train on a mix of NLI data + synthetic data (sample from LM, retrieve, label via human or LLM judge)
- The classifier can be distilled into a 350M-parameter model for fast online attribution
- This classifier IS the verifier — a drop-in replacement for the existing entailment verifier

**4c. Minimal-edit revision model**
- Given the attribution labels (supported/contradicted/unverified for each claim), a revision LM edits the original text to:
  - Remove contradicted claims
  - Qualify unverified claims ("According to some sources...")
  - Insert citations for supported claims
- Train a seq2seq model on (original_text, attribution_labels → revised_text) pairs
- Use a constrained decoding objective: minimize edit distance while maximizing attribution score

**4d. Attribution reward for RL/DPO**
- The attribution score (fraction of claims that are SUPPORTED) can be used as a reward signal
- During RL or DPO, the model is rewarded for generating claims that survive web verification
- This creates a direct optimization target for factuality without human feedback
- Combine with DPO from paper 1: preference pairs are (high-attribution output, low-attribution output)

---

## 5. GPT-4 System Card — OpenAI, 2023

**Core idea:** Comprehensive safety evaluation of GPT-4 including red-teaming, reward model hacking, sycophancy, and calibration; documents the safety mitigation stack.

### Implementable Ideas

**5a. Reward model ensemble for adversarial robustness**
- Train multiple reward models on different subsets of preference data or with different architectures (MLP head, linear head, Deep ENN head)
- At training time, use ensemble uncertainty (variance across RM scores) to detect out-of-distribution exploitation
- If RM variance > threshold, downweight the sample or query a human
- Integration: the verifier ensemble replaces the single RM in the DPO/RL loop

**5b. Sycophancy probe and debiasing**
- Train a classifier to detect when the model is agreeing with the user's mistaken premise (sycophancy)
- Feature: the model's hidden state sensitivity to the user's stated opinion vs factual ground truth
- During inference, if sycophancy score > threshold, generate a disclaimer or contrary response
- Architecture: a small adapter trained on synthetic sycophancy pairs

**5c. Calibration head for abstention**
- GPT-4 System Card shows models are poorly calibrated on hard questions
- Train a separate calibration head that predicts "probability my answer is correct" given the LM's hidden states
- Use temperature scaling or Platt scaling on the last hidden layer
- If calibrated probability < threshold, abstain or flag for human review
- Directly integrates with the verifier: the verifier's confidence becomes the calibration target

**5d. Red-teaming automated adversary**
- Use an attacker model to probe the target model for safety violations
- Attacker generates adversarial prompts using:
  - Gradient-based attacks (hotflip, GCG)
  - Evolutionary search over prompt templates
  - Role-playing / jailbreak patterns from the System Card taxonomies
- Track which attack categories the target fails on, generate targeted training data
- Integration: automated red-teaming loop feeds into DPO training data as new preference pairs

**5e. Hierarchical safety classifier**
- Inspired by the System Card's content-level filtering: train a fast classifier cascade
  - Tier 1: keyword/regex filter (O(1µs))
  - Tier 2: small BERT classifier for topic detection (O(1ms))
  - Tier 3: LM-based reasoning judge for nuanced violation detection (O(100ms))
- If any tier flags a violation, the output is blocked or revised before reaching the user
- The tier-3 judge can be shared with the CAI constitution judge (paper 2)

---

## Cross-Paper Integration Architecture

```
Generator LM
  ├── DPO finetuning loop (paper 1) — replaces RLHF
  ├── Constitutional SL/RL phase (paper 2) — self-critique + RLAIF
  ├── RARR attribution loop (paper 4) — post-hoc research + revise
  │
  ├── Verifier (shared across all papers):
  │   ├── Claim decomposer (paper 4)
  │   ├── Entailment / attribution classifier (paper 4b)
  │   ├── Truthfulness probe (paper 3b)
  │   ├── Sycophancy probe (paper 5b)
  │   └── Calibration / abstention head (paper 5c)
  │
  ├── RM Ensemble (paper 5a) — for adversarial robust DPO
  │
  ├── Automated red-teaming loop (paper 5d) → generates preference pairs
  │
  └── Safety classifier cascade (paper 5e) → inference-time guard
```

### Unified Loss Landscape

| Component | Loss Function | Paper |
|-----------|--------------|-------|
| DPO | `-log σ(β(log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))` | 1 |
| Truthfulness contrastive | `max(0, m - log p(true) + log p(false))` | 3 |
| Entailment classifier | Cross-entropy over {SUPPORTED, CONTRADICTED, UNVERIFIED} | 4 |
| Sycophancy probe | Binary CE: sycophantic vs honest | 5 |
| Attribution reward | `fraction_of_claims_supported / total_claims` | 4 |
| RM ensemble variance | `Var[{RM_1(x,y), ..., RM_k(x,y)}]` — used as loss weight | 5 |
