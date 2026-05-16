# Aurelian Memory Core: A Three-Tier Differentiable Memory Architecture for Stable Long-Horizon Language Agents

Draft: v0.1
Repository: Aurelius AMC-first branch
Status: technical paper draft; experimental results pending

## Abstract

Large language models exhibit strong in-context pattern matching but remain brittle when tasks require durable state, repeated corrections, provenance-aware recall, or explicit separation between transient context and trusted long-term knowledge. Existing approaches extend context length, add recurrence, retrieve external documents, or attach nearest-neighbor memories, but these mechanisms often treat memory as either extra tokens or an external lookup step. This paper proposes the Aurelian Memory Core (AMC), a three-tier memory architecture designed for Aurelius. AMC integrates per-forward working memory, session-scoped episodic memory, and consolidated long-term memory into the model/runtime boundary while preserving inspectability and safety controls.

AMC is organized around three memory tiers. Tier 1 is a transient working memory used inside generation and never persisted. Tier 2 is an episodic session memory written by surprise, correction, and usefulness gates. Tier 3 is a long-term consolidated store with provenance, confidence, trust level, decay policy, and quarantine semantics. Unlike retrieval-only systems, AMC defines both read and write paths, explicit promotion rules, contradiction handling, and telemetry for retrieval/write decisions. The goal is not merely longer context, but stable memory behavior: recall useful prior facts, prefer repeated high-confidence evidence, avoid storing noise, and quarantine unsafe or conflicting memories before reuse.

We specify the AMC architecture, training objectives, runtime contract, safety invariants, and an evaluation protocol. The initial AMC-Memory benchmark covers cross-session recall, surprise-gate selectivity, consolidation preference, and contradiction quarantine. We also define ablations comparing no-AMC, Tier-2 episodic AMC, and full Tier-2 plus Tier-3 AMC. The central hypothesis is that memory-native models and agents can improve long-horizon reliability without relying on unbounded context windows or opaque prompt templates.

## 1. Introduction

Transformer language models process context through attention over a finite sequence. This design is powerful, but it conflates several types of information:

1. transient reasoning state inside the current generation,
2. session facts and user corrections that should persist briefly,
3. durable knowledge or policy priors that should be retained across sessions,
4. untrusted observations that should not be reused without verification.

A single context window cannot cleanly represent all four. Increasing the window helps, but it does not solve provenance, forgetting, consolidation, contradiction, or safety. Similarly, retrieval-augmented generation can inject useful documents, but most RAG systems are read-heavy: they retrieve from an external store without making model-internal memory writes, promotion, decay, and quarantine first-class operations.

Aurelius adopts a different framing. Memory is not just additional context. Memory is a control surface spanning the model, runtime, alignment layer, and agent loop. The Aurelian Memory Core (AMC) is the proposed mechanism for this control surface.

The AMC-first direction for Aurelius is intentionally narrow. Instead of building many loosely connected subsystems, Aurelius should prove one core contribution:

AMC is a per-layer, three-tier memory hierarchy that can be trained, inspected, ablated, and evaluated directly.

The practical objective is to answer four questions:

- Can the model retrieve session facts that are not in the immediate user message?
- Can it decide which observations are worth storing?
- Can it promote repeated, high-confidence memories while ignoring one-off noise?
- Can it quarantine contradictions or unsafe memories instead of silently using them?

## 2. Background and Related Work

AMC builds on several lines of prior work while making a different system-level tradeoff.

### 2.1 Transformer attention

The Transformer introduced self-attention as a scalable sequence modeling primitive [Vaswani et al., 2017]. Standard self-attention treats all usable context as tokens inside the active sequence. This is expressive, but memory is bounded by context length and the model has no explicit write/consolidate/quarantine interface.

### 2.2 Recurrent and compressed context

Transformer-XL introduced segment recurrence so hidden states from earlier segments can condition later segments [Dai et al., 2019]. Compressive Transformers added compressed memories for longer-range sequence modeling [Rae et al., 2019]. These systems distinguish recent and older state, but their memory is mostly sequence-continuation state. AMC borrows the idea that memory should have tiers, but adds explicit session, long-term, provenance, write-gate, and safety semantics.

### 2.3 Retrieval-augmented generation

RAG retrieves documents from an external store and conditions generation on them [Lewis et al., 2020]. RETRO scales retrieval to trillions of tokens and improves language modeling with retrieved chunks [Borgeaud et al., 2021]. These systems show that external memory can improve factuality and data efficiency. AMC differs by making memory writes and consolidation part of the architecture, not only retrieval at inference time.

### 2.4 Nearest-neighbor and memorizing transformers

Memorizing Transformers augment attention with approximate kNN retrieval over prior activations [Wu et al., 2022]. This is close to AMC's premise that neural activations can be memory keys/values. AMC extends that premise into a three-tier lifecycle: transient working memory, session episodic memory, and trusted long-term memory with quarantine.

### 2.5 Test-time memory

Titans explores learning to memorize at test time using neural memory modules [Behrouz et al., 2024]. AMC shares the goal of adaptive memory during inference, but emphasizes inspectable runtime contracts, explicit tiering, and safety/provenance constraints suitable for agentic systems.

## 3. Design Goals

AMC is designed around six goals.

### Goal 1: Separate memory by lifetime

Aurelius should not use one mechanism for all memory. A scratchpad fact inside a forward pass is different from a session correction, and both are different from a durable policy entry. AMC therefore separates memory into three lifetimes:

- Tier 1: working memory, current forward pass only.
- Tier 2: episodic memory, session-scoped.
- Tier 3: long-term memory, consolidated and provenance-tracked.

### Goal 2: Make reads and writes explicit

A model that can retrieve memory but cannot explain why it stored something is incomplete. AMC must expose:

- retrieval scores,
- write scores,
- rejection reasons,
- source/provenance,
- trust level,
- age/decay,
- conflict/quarantine status.

### Goal 3: Keep the first implementation small

The first stable AMC implementation should not require a full platform rewrite. The minimal useful path is:

1. define the tier contracts,
2. expose deterministic AMC-Memory tasks,
3. wire a model/runtime runner later,
4. compare no-AMC vs Tier-2 vs full-AMC.

### Goal 4: Support model ablations

AMC should be evaluated by removing or enabling tiers:

- baseline_no_amc: standard transformer path,
- tier2_episodic: Tier 1 + Tier 2,
- full_amc: Tier 1 + Tier 2 + Tier 3.

The system is not considered validated until AMC variants outperform the baseline on memory-specific tasks without unacceptable regression on general benchmarks.

### Goal 5: Make alignment memory-native

Alignment should not live only in prompts or output filters. In AMC, constitutional policies are high-priority Tier 3 memories. Unsafe or conflicting observations are quarantined. User corrections can affect Tier 2 immediately but require verification before Tier 3 promotion.

### Goal 6: Avoid hidden state that cannot be audited

Agent systems fail when they accumulate hidden assumptions. AMC requires debug surfaces for retrieval and write decisions, especially for policy-relevant memories.

## 4. Architecture

AMC sits between transformer blocks, runtime memory services, and the agent/alignment loop.

```text
User/API/Agent request
        |
        v
Serving + policy gate
        |
        v
Model runtime
        |
        v
Transformer blocks with AMC hooks
        |
        +--> Tier 1: working memory
        +--> Tier 2: episodic memory
        +--> Tier 3: long-term memory
```

### 4.1 Tier 1: working memory

Tier 1 is transient model-local state. It exists only during the current forward pass or generation episode.

Purpose:

- summarize local attention state,
- store transient constraints,
- support latent planning within the current generation,
- provide cheap memory behavior enabled in all variants.

Rules:

- never persists across requests,
- has no user-visible durable identity,
- is not used for personalization,
- can be ablated independently.

### 4.2 Tier 2: episodic memory

Tier 2 is session-scoped. It stores observations that are useful within a session or task trajectory.

Examples:

- the user corrected a prior answer,
- a tool call failed and revealed a dependency constraint,
- the agent discovered a local repo convention,
- an intermediate plan succeeded or failed.

Rules:

- resettable by session,
- written through surprise/usefulness gates,
- retrieved by query similarity plus recency/importance,
- not automatically promoted to long-term memory,
- debug telemetry required.

### 4.3 Tier 3: long-term memory

Tier 3 is durable consolidated memory. It contains high-confidence facts, stable user/project preferences, and constitutional policy entries.

Required fields:

- id,
- text/value,
- source,
- provenance,
- confidence,
- trust_level,
- last_verified_at,
- created_at,
- decay_policy,
- quarantine_status.

Rules:

- only consolidation can write Tier 3,
- direct arbitrary generation cannot write Tier 3,
- conflicting entries must be quarantined,
- policy memories are high-priority retrievals,
- all Tier 3 writes must be auditable.

## 5. Formal Sketch

Let a transformer layer receive hidden state h_l,t for layer l and token t. AMC introduces tier-specific memory banks M_l^1, M_s^2, and M^3:

- M_l^1: transient per-layer working memory,
- M_s^2: session memory for session s,
- M^3: long-term consolidated memory.

Each memory bank stores key-value records:

m_i = (k_i, v_i, meta_i)

where meta_i may include source, confidence, trust_level, age, and quarantine_status.

### 5.1 Query construction

For each layer and token, construct a memory query:

q_l,t = W_q h_l,t

The runtime may augment q_l,t with task/session context c_s:

q'_l,t = f_q(h_l,t, c_s)

### 5.2 Tier retrieval

For each tier z in {1, 2, 3}, retrieve top-k values by score:

score_z(q, m_i) = sim(q, k_i) + beta_i - decay_i - quarantine_penalty_i

where:

- sim is cosine or dot-product similarity,
- beta_i reflects importance/confidence,
- decay_i reduces stale memory influence,
- quarantine_penalty_i is effectively infinite for quarantined records unless explicitly debugging.

The tier retrieval vector is:

r_z = sum_i softmax(score_z(q, m_i)) v_i

### 5.3 Gated fusion

AMC fuses memory into the layer through tier gates:

g_z = sigmoid(W_z [h_l,t ; r_z ; e_z])

where e_z contains tier telemetry such as score margin, record age, and trust summary.

The memory-conditioned hidden state is:

h'_l,t = h_l,t + sum_z g_z P_z r_z

where P_z projects each tier's value space back into the model dimension.

### 5.4 Write gate

A write candidate is computed after observations, tool results, user corrections, or generation outcomes:

w = sigmoid(W_w [h; surprise; usefulness; error; confidence; trust])

A Tier 2 write occurs when:

w > tau_write and quarantine(record) = false

Tier 3 writes require consolidation rather than immediate write:

promote(record) = repeated(record) AND confidence(record) > tau_conf AND verified(record)

### 5.5 Contradiction quarantine

For a new candidate record x and existing memory y, define a conflict predicate:

conflict(x, y) = semantic_contradiction(x, y) AND overlap(scope_x, scope_y)

If conflict is true and neither source dominates by a configured trust rule, AMC stores x in quarantine instead of retrieving it by default.

This is essential for safety: memory should not become more dangerous merely because the model saw a claim once.

## 6. Training Objectives

AMC can be trained incrementally. The first phase does not require all objectives at once.

### 6.1 Language modeling loss

The base loss remains next-token prediction:

L_lm = - sum_t log p(x_t | x_<t, AMC)

### 6.2 Retrieval supervision

When a target memory is known, train retrieval to rank it above distractors:

L_retrieval = -log exp(score(q, m+)) / sum_j exp(score(q, m_j))

### 6.3 Write sparsity

To prevent memory spam, penalize unnecessary writes:

L_write = lambda_w mean(w_t)

This should be balanced against recall utility so the model stores high-value corrections and durable facts.

### 6.4 Consolidation consistency

For repeated high-confidence session records, train the model/runtime to promote the stable version:

L_consolidate = CE(promote_label, promote_score)

### 6.5 Quarantine loss

For conflicting or untrusted memory pairs, train quarantine classification:

L_quarantine = CE(quarantine_label, quarantine_score)

### 6.6 Total objective

A practical objective is:

L = L_lm + alpha L_retrieval + beta L_write + gamma L_consolidate + delta L_quarantine

The initial implementation should tune these coefficients conservatively. Memory utility should be proven by ablation, not assumed.

## 7. Runtime Contract

The public serving surface should allow callers to opt into AMC behavior without exposing internal implementation details:

```json
{
  "model": "aurelius-forge-1b",
  "messages": [],
  "amc": {
    "episodic": true,
    "long_term": false,
    "session_id": "optional-session-id",
    "debug_retrievals": false,
    "consolidation_threshold": 0.65
  }
}
```

Minimum semantics:

- unknown AMC fields are ignored safely,
- session_id scopes Tier 2,
- long_term controls Tier 3 retrieval eligibility,
- debug_retrievals exposes sanitized telemetry,
- consolidation_threshold controls promotion strictness but cannot bypass trust/quarantine rules.

## 8. Memory-Native Agent Loop

AMC changes the agent loop from prompt assembly to memory control:

```text
observe
  -> retrieve Tier 2/Tier 3 memories
  -> plan conditioned on retrieved memories
  -> act with tools or response
  -> reflect on outcome
  -> write useful observations to Tier 2
  -> consolidate repeated verified facts into Tier 3
```

The design rule is simple:

If an agent feature cannot say how it reads from or writes to AMC, it is not core Aurelius architecture yet.

This prevents feature sprawl. Tool catalogs, persona systems, planning variants, and workflow engines should remain peripheral until they connect to memory telemetry and benchmark gates.

## 9. Alignment and Safety

AMC treats alignment as memory behavior.

### 9.1 Constitutional memory

Constitutional principles are stored as Tier 3 policy entries with high priority. They are retrieved for aligned generation and can be inspected in debug mode.

Policy record fields:

- id,
- text,
- version,
- provenance,
- priority,
- enabled,
- last_verified_at.

This makes policy state explicit rather than hiding it entirely in prompt templates.

### 9.2 Quarantine

Untrusted, contradictory, or unsafe records are quarantined. Quarantined records are not used by default during generation.

Quarantine reasons include:

- contradiction with a higher-confidence memory,
- untrusted source,
- unsafe instruction,
- low-confidence external observation,
- stale policy version.

### 9.3 User corrections

User corrections can update Tier 2 immediately. They should not automatically become Tier 3 durable truth. Promotion requires repeated evidence, source confidence, or explicit verification.

### 9.4 Debugging requirement

Every safety-relevant output should be able to answer:

- which policy memories were retrieved,
- which user/project memories were retrieved,
- which memories were rejected,
- whether any contradictions were quarantined.

## 10. Evaluation

AMC requires memory-specific evaluation, not only general language benchmarks.

### 10.1 General benchmarks

General capability must not regress beyond an accepted threshold:

- MMLU,
- HellaSwag,
- ARC-Challenge,
- TruthfulQA,
- GSM8K,
- HumanEval.

### 10.2 Long-context benchmarks

Long context should be measured separately:

- RULER,
- Needle-in-a-Haystack.

These test whether longer sequences are usable, but they are not sufficient for AMC because they do not fully test write/consolidation/quarantine behavior.

### 10.3 AMC-Memory benchmark

The initial AMC-Memory benchmark has four deterministic tasks:

1. cross_session_recall: retrieve a stored prior-session value,
2. surprise_gate_selectivity: choose the observation worth writing,
3. consolidation_preference: promote repeated high-confidence memory over transient details,
4. contradiction_quarantine: quarantine conflicting memories before use.

A model runner only needs:

```python
generate_fn(prompt: str) -> str
```

The benchmark returns per-task accuracy and an overall score. This keeps early experimentation simple and CI-friendly.

### 10.4 Ablation table

The minimum paper-quality ablation is:

| Variant | Tier 1 | Tier 2 | Tier 3 | Expected result |
| --- | --- | --- | --- | --- |
| no-AMC baseline | no | no | no | weakest on memory tasks |
| Tier-2 episodic | yes | yes | no | better session recall and write selection |
| full AMC | yes | yes | yes | best consolidation and quarantine behavior |

### 10.5 Required reporting

Each run should report:

- benchmark JSON output,
- per-task pass-rate table,
- retrieval latency,
- write latency,
- memory footprint,
- retrieval/write telemetry examples,
- general benchmark deltas.

## 11. Implementation Plan

The implementation should stay small until evidence supports expansion.

### Phase A: Scaffold

Already in progress in Aurelius:

- architecture doc,
- alignment scope doc,
- benchmark config,
- deterministic AMC-Memory benchmark,
- unit tests for builders/scoring.

### Phase B: Runner

Add a small CLI or script that runs AMC-Memory against:

- a stub function,
- a local checkpoint,
- an OpenAI-compatible endpoint.

Do not build a large platform for this. One simple runner with JSON output is enough.

### Phase C: Telemetry

Expose a compact telemetry object:

```json
{
  "retrievals": [
    {"tier": 2, "id": "...", "score": 0.81, "reason": "session correction"}
  ],
  "writes": [
    {"tier": 2, "score": 0.74, "accepted": true, "reason": "high surprise"}
  ],
  "quarantine": []
}
```

### Phase D: Model integration

Only after the runner is stable:

- add layer hooks,
- add Tier 2 memory bank,
- add write gate,
- add ablation flags,
- run no-AMC vs Tier-2.

Tier 3 should wait until Tier 2 wins on AMC-Memory.

## 12. Expected Contributions

If validated, AMC contributes:

1. a three-tier memory architecture for language agents,
2. explicit read/write/consolidate/quarantine semantics,
3. memory-native alignment through constitutional Tier 3 policy entries,
4. an ablation protocol separating memory ability from general language ability,
5. an inspectable runtime contract for long-horizon agents.

## 13. Limitations

AMC has several risks.

First, memory can amplify mistakes. This is why quarantine and provenance are core requirements rather than optional safety features.

Second, write gates can collapse into either spam or silence. The system must report write rates and tune sparsity carefully.

Third, retrieval can create privacy risks. Debug telemetry must avoid exposing private long-term memories unnecessarily.

Fourth, synthetic memory benchmarks can be gamed. AMC-Memory should be treated as a unit-test-like gate, not the final evidence of real-world performance.

Fifth, full per-layer memory can increase latency and memory footprint. The first implementation should measure overhead before adding larger tiers.

## 14. Conclusion

The Aurelian Memory Core reframes Aurelius around one central claim: stable agents need explicit memory lifecycles, not merely longer prompts or larger tool catalogs. AMC separates transient working state, session episodic memory, and trusted long-term memory. It defines retrieval, writing, consolidation, quarantine, and telemetry as first-class operations.

The immediate path is deliberately conservative. Aurelius should prove AMC with deterministic benchmarks, small ablations, and inspectable runtime behavior before scaling model variants or reintroducing broad v2-style subsystems. If Tier-2 and full-AMC variants beat the no-AMC baseline on memory-specific benchmarks while preserving general capability, AMC becomes a credible technical foundation for Aurelius models and agents.

## References

- Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. "Attention Is All You Need." arXiv:1706.03762, 2017.
- Zihang Dai, Zhilin Yang, Yiming Yang, et al. "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." arXiv:1901.02860, 2019.
- Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, et al. "Compressive Transformers for Long-Range Sequence Modelling." arXiv:1911.05507, 2019.
- Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv:2005.11401, 2020.
- Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al. "Improving language models by retrieving from trillions of tokens." arXiv:2112.04426, 2021.
- Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, et al. "Memorizing Transformers." arXiv:2203.08913, 2022.
- Ali Behrouz, Peilin Zhong, Vahab Mirrokni. "Titans: Learning to Memorize at Test Time." arXiv:2501.00663, 2024.

## Appendix A: Minimal AMC-Memory runner shape

```python
from src.eval.amc_memory_benchmark import AMCMemoryBenchmark

bench = AMCMemoryBenchmark()
results = bench.evaluate(generate_fn, context_tokens=1024, samples_per=5)
print(bench.overall_score(results))
```

## Appendix B: Claims not yet made

This draft does not claim that AMC currently improves benchmark scores. The implementation has a benchmark scaffold and architecture contract. Empirical claims require future ablation runs.
