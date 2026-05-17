# Aurelius AMC-First Core Surface

Status: canonical stabilization boundary for the AMC-first line

## Purpose

This document defines the code that is release-critical while Aurelius is being stabilized around the Aurelian Memory Core (AMC). It exists to prevent the repository from expanding every experimental subsystem at once.

The current rule is simple: stabilize Tier-2 evidence first, then decide whether Tier-3 / Atlas work earns promotion.

## Release-critical core

These areas are in-scope for stabilization, validation, and release gating:

- `src/model/` — model architecture and AMC-compatible model surfaces.
- `src/memory/` — memory contracts, AMC seams, Tier-2 retrieval/write policy, and prompt-context integration.
- `plugins/memory/` — existing memory primitives that back the current `src.memory` shims.
- `src/eval/` — AMC-Memory, long-context gates, and benchmark runners.
- `src/training/` — training loops and alignment plumbing required to produce usable checkpoints.
- `src/runtime/` — feature flags, capability contracts, runtime policy, and safety-facing glue.
- `src/serving/` and `gateway/` — minimal OpenAI-compatible serving/API path.
- `aurelius_cli/` — user-facing CLI entrypoints that expose the stable surface.
- `agent/` — minimal agent loop and sandbox components needed to exercise memory safely.
- `src/security/` and selected `src/safety/` — release hardening, policy checks, and memory quarantine safety.
- `tests/` covering the areas above — do not delete tests just because the suite is large.

## Quarantined / non-blocking areas

These can remain in the repository, but they are not allowed to block AMC-first stabilization unless a focused test or benchmark proves they are part of the release path:

- `src/trading/`
- `src/simulation/`
- `src/federation/`
- `src/profiling/`
- broad `src/inference/` variants not tied to benchmark gates
- broad `src/alignment/` variants outside SFT, DPO, GRPO/RLVR, constitutional memory, and quarantine
- `src/ui/` unless a concrete dashboard requirement is active
- `cron/` and `acp_adapter/` unless a release task explicitly needs platform integration

## Tier-2 evidence boundary

Tier-2 is allowed to grow only through measurable evidence:

1. A deterministic or model-backed benchmark runner must produce a JSON/JSONL artifact.
2. The change must expose operational metrics, not just new code paths.
3. A no-memory vs Tier-2 comparison must be possible for the affected path.
4. Tests must show the metric/ablation surface works before model-quality claims are made.

Current Tier-2 primitives:

- `src/memory/amc_tier2.py`
  - surprise-gated episodic writes
  - query retrieval with recent-memory fallback
  - prompt-context construction
  - write/retrieval evidence metrics
  - no-memory vs Tier-2 context ablation result
- `src/eval/amc_memory_runner.py`
  - oracle/null generators
  - serving-engine/checkpoint adapter
  - JSON output and ignored JSONL append output

## Tier-3 / Atlas hold boundary

Tier-3, Atlas, and full hierarchical AMC expansion are intentionally on hold.

Do not start any of the following until Tier-2 has benchmark evidence:

- Tier-3 long-term semantic consolidation as a production path.
- Atlas-specific memory orchestration.
- learned memory router training.
- Rust-backed AMC promotion as the default path.
- model-family expansion based primarily on Tier-3 claims.
- large directory moves to make space for unproven AMC layers.

Tier-3 / Atlas work may move forward only after all of these are true:

1. AMC-Memory has a repeatable baseline for `null`, `oracle`, and at least one real checkpoint/backend adapter.
2. Tier-2 ablation artifacts show retrieval improves the context available to generation without unsafe leakage.
3. Tier-2 metrics are recorded in benchmark JSONL so regressions can be compared over time.
4. Existing release-critical tests remain green after Tier-2 integration.
5. The proposed Tier-3 change has a focused plan that lists files, tests, benchmark deltas, rollback path, and a non-expansion guarantee.

Until those conditions are met, Tier-3 and Atlas should be discussed in docs/plans only, not expanded in active runtime code.

## Cleanup rule

Repository cleanup should proceed one candidate at a time:

1. Run an import-reference scan.
2. Verify dedicated tests for the affected package.
3. Update stale docs that mention removed artifacts.
4. Stage only intentional files.
5. Validate with ruff, compileall, focused pytest, and `git diff --check`.

The first cleanup completed under this rule removed tracked `.bak` backup artifacts and moved rollback guidance to git history.
