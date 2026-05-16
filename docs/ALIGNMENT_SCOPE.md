# Aurelius Alignment Scope

Status: canonical scope for AMC-first stabilization

## Decision

Aurelius should narrow alignment work to five tracks until the AMC-first benchmark gates are reproducible. Existing alignment modules outside these tracks remain research experiments, not public product promises.

## Canonical tracks

### 1. Supervised fine-tuning (SFT)

Purpose: baseline instruction following and format control.

Gate:
- Improves instruction-following evals without degrading base perplexity beyond an agreed threshold.
- Produces model cards with data sources and known limitations.

### 2. Direct Preference Optimization (DPO)

Purpose: preference tuning without a separate reward model during optimization.

Gate:
- Evaluated on helpfulness, harmlessness, and reasoning preference sets.
- Compared against SFT-only checkpoint.

### 3. GRPO / RLVR

Purpose: verifiable-reward training for math, code, tool-use, and structured tasks.

Gate:
- Rewards are programmatic or independently checkable.
- No reward hacking regressions on held-out tasks.

### 4. Constitutional memory

Purpose: make policy retrieval explicit and inspectable by storing principles as trusted Tier 3 memory entries.

Policy entry fields:
- id
- text
- version
- priority
- provenance
- enabled
- last_verified_at

Gate:
- High-priority policies are retrieved on every aligned generation.
- Retrieval can be debugged without exposing private user memories.
- Policy updates are versioned.

### 5. Red-team memory quarantine

Purpose: prevent unsafe, contradictory, or untrusted observations from silently becoming durable memories.

Gate:
- Contradictions are detected and routed to quarantine/verification.
- Quarantined items are not retrieved for generation by default.
- The audit trail records source, confidence, and reason.

## How AMC changes alignment

Traditional alignment is often a final-stage model update plus runtime filtering. Aurelius should instead make alignment a memory-layer behavior:

```text
request
  -> retrieve constitutional Tier 3 policy memories
  -> retrieve task/user memories allowed for this session
  -> generate candidate
  -> verify candidate against policies and task reward
  -> emit / revise / refuse
  -> write outcome reflection to Tier 2
  -> consolidate only if trusted and useful
```

This makes the safety state inspectable and debuggable. It also prevents hidden prompt templates from becoming the only source of policy behavior.

## Experimental algorithms

Algorithms outside the five canonical tracks should be labeled experimental until they have:

- A named benchmark.
- A comparison baseline.
- Dependency isolation.
- A maintainer/owner.
- Tests that do not require optional heavyweight infrastructure by default.

Examples that should remain experimental unless promoted through evidence:

- Large collections of preference-optimization variants.
- Unverified self-rewarding loops.
- Persona tuning layers.
- Agent-planning algorithms not wired into AMC telemetry.
- Speculative safety filters that cannot explain which memory/policy caused a decision.

## Minimum release gate

An aligned Aurelius release must publish:

1. SFT baseline scorecard.
2. DPO or GRPO/RLVR delta from baseline.
3. AMC memory benchmark results.
4. Red-team quarantine test results.
5. A model card describing memory behavior and policy provenance.
