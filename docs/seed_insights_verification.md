# Verification / Anti-Hallucination: Seed Insights

## 1. PRM800K / Process Reward Model (Lightman et al., 2023)
**arXiv:2305.20050 — "Let's Verify Step by Step"**

### Core Mechanism
A token-level Process Reward Model (PRM) scores each intermediate reasoning step for correctness, rather than only judging the final answer (outcome supervision). Trained on 800K human step-level labels (PRM800K dataset). Process supervision enables fine-grained credit assignment, outperforming outcome-only models on MATH (78% solve rate).

### Neural Module Implementation
- **Step-level scorer**: A learned scalar head (linear projection on top of a frozen or fine-tuned base LM's hidden states) that emits a correctness logit per reasoning step (delineated by newlines or separator tokens).
- **Causal segmentation**: Pooling over token positions within each step segment — e.g., mean-pool the step's last N hidden states into a step embedding, then project to a scalar.
- **Active learning loop**: The scorer selects low-confidence steps for human labeling, iteratively expanding the supervision set.

### Verifier Features
1. **Step-segmented Transformer head**: Separate correctness scores for each reasoning step rather than a single answer score.
2. **Monotonic confidence decay detector**: Tracks whether per-step confidence drops sharply — a signal that the chain is degrading.
3. **Backward credit assignment**: Propagates the PRM gradient through the step boundary to weight individual tokens differently within a step.
4. **Self-consistency bridge**: Compares the PRM's step scores against sampled rollouts of the same problem — low-scoring steps that contradict majority-answer trajectories are flagged.

### Training Signal
- Step-level binary labels (correct/incorrect) from human annotators.
- Loss = binary cross-entropy per step, summed over the reasoning chain.
- Active learning prioritizes steps where PRM confidence is near 0.5 (highest epistemic uncertainty).

---

## 2. SelfCheckGPT (Manakul, Liusie & Gales, 2023)
**arXiv:2303.08896 — "Zero-Resource Black-Box Hallucination Detection"**

### Core Mechanism
Hallucination is detected via **stochastic sampling consistency**: factual claims reappear consistently across multiple LLM samples; hallucinated claims diverge. No external database or token-level probabilities required — works as a black-box detector.

### Neural Module Implementation
- **Consistency encoder**: A cross-attention module that takes the original passage as query and N sampled passages as key/value pairs, producing a sentence-level consistency score.
- **Sentence-level NLI-style classifier**: Fine-tune a BERT/RoBERTa model to classify a sentence from the original passage as "supported" or "contradicted" by each sampled passage, then aggregate via mean or min-pooling.
- **BERTScore variant**: Embed each sentence and each sampled-passage sentence, compute cosine similarity matrix, take the max-over-samples match score per sentence.

### Verifier Features
1. **Multi-sample contradiction network**: Processes K stochastic passes through a shared-weight Siamese encoder; pairwise contradiction scores aggregated into a single hallucination likelihood per sentence.
2. **NLI-based consistency classifier**: A small transformer (e.g., 6-layer DistilBERT) fine-tuned to classify (original_sentence, sampled_sentence) pairs as entailment/neutral/contradiction.
3. **Token-level inconsistency heatmap**: For each token, measure cross-sample Jensen-Shannon divergence of the predictive distribution — high divergence = hallucination-prone token.
4. **Adaptive sample budget**: Dynamically increases K (number of samples) for sentences where the consistency score variance is high, reducing compute for already-confident sentences.

### Training Signal
- **No ground-truth labels needed at inference** (zero-resource). For training the NLI classifier: use Wikipedia or KB-derived sentence pairs (entailed = factual, contradictory = hallucinated proxy).
- Supervised fine-tuning on SelfCheckGPT's released human-annotated WikiBio dataset (sentence-level factuality labels).
- AUC-PR optimized; loss = binary cross-entropy for sentence factuality.

---

## 3. FActScore (Min et al., 2023)
**arXiv:2305.14251 — "Fine-grained Atomic Evaluation of Factual Precision"**

### Core Mechanism
Decomposes generated text into **atomic facts** (minimal self-contained claims) and verifies each against a knowledge source (Wikipedia). Returns the percentage of supported atomic facts. ChatGPT achieves only 58% on biography generation, highlighting the gap.

### Neural Module Implementation
- **Atomic fact decomposer**: A fine-tuned sequence-to-sequence model (T5 or FLAN-T5) that takes a sentence and emits a list of atomic facts — each a short, verifiable claim.
- **Retrieval-augmented verifier**: An encoder that jointly represents (atomic_fact, retrieved_passage) and emits a binary supported/unsupported logit.
- **Dense fact retriever**: A bi-encoder (e.g., Contriever or ColBERT) that retrieves the top-k Wikipedia passages for each atomic fact.

### Verifier Features
1. **Atomic decomposition head**: A generative LM decoder with a controlled decoding constraint emitting one atomic fact per line; trained on human-decomposed fact lists.
2. **Cross-encoder fact-scoring module**: Takes (claim, evidence) and produces a fine-grained score (0.0-1.0) — not binary, allowing partial support detection.
3. **Evidence sufficiency predictor**: A learned threshold or small MLP that determines whether the retrieved passage contains enough evidence to make a judgment (avoiding false negatives from poor retrieval).
4. **Multi-hop fact chaining**: For atomic facts requiring compositional knowledge (e.g., "X was the first woman to win Y"), chains multiple retrievals and scores the logical conjunction.
5. **Self-consistency filter**: Re-ranks atomic facts by their overlap with facts from other model samples — low-overlap facts are sent to the verifier with higher scrutiny weight.

### Training Signal
- Human-annotated atomic fact lists + support labels (from FActScore's released data).
- Loss: binary cross-entropy for each atomic fact's support label, aggregated via macro-F1.
- Retrieve-then-verify pipeline trained end-to-end with hard-negative mining over Wikipedia (facts not supported by the retrieved passage but supported by a different passage).
