# Aurelius AMC-First Architecture

Status: accepted direction for the pre-v2 Aurelius repository
Scope: original `/Users/christienantonio/aurelius` tree only; do not merge the v2 backup wholesale

## Executive decision

Aurelius should be rebuilt around the Aurelian Memory Core (AMC) as the primary novel contribution, not around breadth of loosely connected subsystems. The stable direction is:

1. Keep one canonical model/runtime path working before adding more variants.
2. Prove AMC with ablations and memory-specific benchmarks.
3. Make alignment and agents memory-native rather than bolted-on filters or tool loops.
4. Move speculative or duplicated subsystems behind registries, quarantine them, or delete them.

This is Option C: AMC-First Focused Build.

## Non-goals

- Do not reintroduce the v2 rewrite wholesale.
- Do not expand the native skill catalog until AMC and evaluation gates are stable.
- Do not add another model-family abstraction before one benchmarked AMC variant exists.
- Do not treat module count as evidence of capability.

## Core architecture

Aurelius should have four explicit layers:

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
        +--> Tier 1: Working memory, per-forward-pass hidden state scratchpad
        +--> Tier 2: Episodic memory, session-scoped differentiable retrieval/write
        +--> Tier 3: Long-term store, consolidated durable memories and constitutional priors
```

### Tier 1: Working memory

Purpose: short-lived information inside the current forward pass.

Rules:
- Lives inside the model block or generation step.
- Never persists across requests.
- Used for transient chain-of-thought-like latent state, attention summaries, and local task constraints.
- Must be cheap enough to keep enabled in all model variants.

### Tier 2: Episodic memory

Purpose: session-local memory that helps with multi-turn tasks and repeated attempts.

Rules:
- Written by a surprise/importance gate after observations, tool results, failed attempts, or user corrections.
- Retrieved during generation when query similarity and recency exceed threshold.
- Resettable by session.
- Must expose telemetry: writes, retrievals, top-k scores, age, importance, and rejection reason.

### Tier 3: Long-term memory

Purpose: durable consolidated knowledge and policy priors.

Rules:
- Only written through consolidation, never directly by arbitrary generation.
- Entries must carry provenance, confidence, last_verified_at, trust_level, and decay policy.
- Constitutional/safety priors are high-priority LTS entries retrieved for every aligned generation.
- Unknown, untrusted, or externally-sourced memories must be quarantined until verified.

## Memory-native agent loop

The agent loop should be a memory-control process:

1. Observe: parse user input, environment state, and tool outputs.
2. Retrieve: query Tier 2 and Tier 3 with task embedding plus policy context.
3. Plan: generate candidate action trajectories conditioned on retrieved memories.
4. Act: call tools or produce response.
5. Reflect: score outcome, surprise, usefulness, and safety.
6. Write: store high-value observations in Tier 2.
7. Consolidate: promote repeated or high-confidence Tier 2 patterns into Tier 3.

The key design rule: if an agent feature cannot explain how it reads from or writes to AMC, it is not core architecture yet.

## Alignment architecture

Aurelius should converge on five alignment tracks only until benchmark evidence says otherwise:

1. SFT: instruction-following baseline.
2. DPO: simple pairwise preference tuning.
3. GRPO/RLVR: verifiable-reward reasoning and tool-use tasks.
4. Constitutional memory: durable safety principles stored as high-priority Tier 3 policy entries.
5. Red-team memory quarantine: unsafe, contradictory, or untrusted memories are isolated and require verification before reuse.

### Alignment flow

```text
Prompt
  -> retrieve constitutional Tier 3 policy entries
  -> retrieve task memories from Tier 2/Tier 3
  -> generate candidate
  -> score with preference/reward/verification heads
  -> block, revise, or emit
  -> write outcome reflection to Tier 2
```

### Safety invariants

- Policy memories are versioned and provenance-tracked.
- Tool outputs are not trusted merely because they were observed.
- User corrections can update session memory immediately but need consolidation before becoming durable defaults.
- Every durable memory write must record source, confidence, and reason.
- Retrieval must be inspectable in debug mode.

## Model variants after stabilization

The model family should be capability-tiered by AMC depth, not by marketing names alone:

| Variant | Purpose | AMC depth | Gate to build |
| --- | --- | --- | --- |
| Aurelius-Swift (~150M) | Fast local baseline | Tier 1 only | after base model eval works |
| Aurelius-Forge (~1B/1.3B) | Main research/workhorse | Tier 1 + Tier 2 | first priority |
| Aurelius-Atlas (3B+) | Agentic long-memory model | Tier 1 + Tier 2 + Tier 3 | only after Forge ablations win |

Do not start Atlas until Forge beats the no-AMC baseline on memory-specific evaluation.

## Kill, quarantine, keep

### Keep as core

- `src/model/` transformer core and AMC-capable model definitions.
- `src/memory/` as the canonical memory facade, after consolidation.
- `src/eval/` benchmark harnesses and AMC-specific evaluations.
- `src/alignment/` only for SFT/DPO/GRPO/RLVR/constitutional-memory path.
- `src/serving/` minimal OpenAI-compatible serving surface.

### Quarantine behind registries

- Speculative decoding variants.
- Numerous alignment algorithms beyond the five-track path.
- Agent planning variants not wired into AMC.
- Profiling/throughput experiments.
- Long-context experiments that do not report against RULER or AMC benchmarks.

### Defer or delete unless a real product requirement exists

- Trading-specific code.
- Federation/multi-party subsystems.
- Persona layers not needed for memory experiments.
- Simulation subsystems not used by benchmark/training gates.

## Required benchmark gates

The project is not stable until these are reproducible:

1. General capability: MMLU, HellaSwag, ARC-Challenge, TruthfulQA, GSM8K, HumanEval.
2. Long context: RULER and Needle-in-a-Haystack.
3. AMC memory: cross-session recall, surprise-gate selectivity, consolidation preference, contradiction quarantine.
4. Ablation: no-AMC baseline vs Tier-2-only vs Tier-2+Tier-3.
5. Runtime: tokens/sec, memory footprint, retrieval latency, write/consolidation latency.

## Public API direction

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

The initial implementation can ignore unknown `amc` options, but the contract should be documented now so serving, evaluation, and memory code evolve toward the same shape.

## Stabilization definition of done

Phase 1 is complete when:

- Namespace/package tests pass.
- `ruff check src/ --ignore=I001` passes.
- AMC benchmark builders and scorers are unit-tested.
- Docs identify one canonical architecture and one benchmark gate list.
- The repo has a written plan for deleting/quarantining non-core subsystems.
- No v2 code is merged back into main wholesale.
