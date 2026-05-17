# Aurelius Directory Clutter Inventory

Status: AMC-first stabilization inventory
Scope: main branch after AMC-first realignment
Generated from repository file counts and import-reference scan

## Executive summary

The repository is still too broad for the current AMC-first objective. The codebase is not broken merely because it is large, but there are several feature families that should not be treated as core until AMC benchmark evidence exists.

Recommended rule for the next phase:

1. Keep code needed for model, training, eval, runtime, memory, serving, safety, and the minimal agent loop.
2. Quarantine broad research families behind registries/docs instead of deleting them immediately.
3. Do not remove directories that still have dedicated tests and external imports until a separate deletion PR proves no consumers.
4. Prioritize small, testable reductions over mass refactors.

## Size snapshot

Top-level Python footprint:

| Path | Python files | Approx. lines | Classification |
| --- | ---: | ---: | --- |
| `src/` | 2139 | 482584 | Core source tree, but internally cluttered |
| `tests/` | 2381 | 491971 | Large but valuable; do not mass-delete |
| `agent/` | 98 | 25710 | Keep minimal loop; avoid expanding |
| `gateway/` | 76 | 15400 | Keep serving/API surface |
| `aurelius_cli/` | 33 | 7785 | Keep CLI surface |
| `tools/` | 27 | 2364 | Keep only if used by agent/runtime |
| `plugins/` | 21 | 2513 | Keep memory plugins relevant to AMC |
| `cron/` | 18 | 1710 | Quarantine unless wired to core agent runtime |
| `acp_adapter/` | 25 | 3413 | Quarantine unless needed for active agent protocol |
| `training_data/` | 14 | 9693 | Keep if used by training scripts; otherwise document ownership |
| `archive/` | 55 | 13547 | Keep excluded; never lint as active code |

Largest `src/` directories:

| Path | Python files | Approx. lines | Classification |
| --- | ---: | ---: | --- |
| `src/training/` | 329 | 92194 | Keep, but needs submodule ownership |
| `src/model/` | 271 | 71283 | Keep, core model surface |
| `src/inference/` | 245 | 63842 | Quarantine variants not tied to benchmark gates |
| `src/eval/` | 174 | 48479 | Keep; central to AMC-first proof |
| `src/alignment/` | 179 | 46829 | Reduce to SFT/DPO/GRPO/RLVR/constitutional-memory path |
| `src/data/` | 106 | 26476 | Keep if used by training/data pipeline |
| `src/serving/` | 66 | 14958 | Keep minimal OpenAI-compatible path |
| `src/security/` | 69 | 12231 | Keep; required for stable release |
| `src/safety/` | 42 | 12967 | Keep selectively; avoid duplicate policy engines |
| `src/ui/` | 27 | 7521 | Quarantine unless tied to active product surface |

## Keep as AMC-first core

These should remain first-class while stabilizing Aurelius:

- `src/model/` — model architecture and AMC hooks.
- `src/memory/` and `plugins/memory/` — memory primitives; consolidate toward AMC contracts.
- `src/eval/` — benchmark harnesses, AMC-Memory, RULER/long-context gates.
- `src/training/` — training loops and optimization primitives.
- `src/runtime/` — feature flags, capability contracts, runtime glue.
- `src/serving/` and `gateway/` — minimal serving/API path.
- `src/security/` and selected `src/safety/` — release hardening and memory quarantine safety.
- `aurelius_cli/` — user-facing commands.
- `agent/` — keep the minimal loop that can read/write AMC.

## Quarantine behind registries or docs

These are not deletion targets yet, but should stop expanding until AMC benchmark evidence exists:

| Area | Why quarantine |
| --- | --- |
| `src/inference/` variants | 245 files; likely multiple speculative/attention/KV experiments. Keep benchmarked paths only as core. |
| `src/alignment/` variants | 179 files; current canonical scope is only SFT, DPO, GRPO/RLVR, constitutional memory, quarantine. |
| `src/profiling/` | Useful for experiments but not product-critical; 14 external references found, mostly tests. |
| `src/federation/` | Broad feature family with dedicated tests, but not part of AMC-first proof. |
| `src/simulation/` | Broad testing/research surface; keep separate from core model/runtime. |
| `src/persona/` | Has real consumers (`agent/`, `aurelius_cli/`, `src/chat/`), so do not delete; narrow to explicit product need. |
| `cron/` and `acp_adapter/` | Useful platform features, but should not block AMC stabilization. |
| `src/ui/` | Quarantine unless tied to active dashboard requirements. |

## Delete-candidate areas, not delete-now areas

These are candidates for later removal only after import rewrites and test decisions:

| Area | External refs found | Recommendation |
| --- | ---: | --- |
| `src/trading/` | 9 | Delete-candidate. Appears isolated to trading tests; not AMC core. |
| `src/simulation/` | 17 | Delete-candidate or move to `archive/experiments` after test triage. |
| `src/federation/` | 17 | Delete-candidate for product core; preserve only if distributed training is an explicit roadmap item. |
| `src/profiling/` | 14 | Defer/delete-candidate; keep only minimal runtime metrics needed by benchmark reports. |
| `src/persona/` | 11 | Not safe to delete now because agent/CLI/chat import it. Narrow instead. |

## Immediate refactor guidance

Do not start with a large directory move. The safe sequence is:

1. Add labels/docs first, not deletions.
2. Keep tests passing for the current core path.
3. For each delete-candidate package:
   - run its tests once,
   - identify non-test importers,
   - either rewrite importers or mark the package experimental,
   - delete only in a focused PR/commit.
4. For `src/inference/` and `src/alignment/`, introduce a small registry/allow-list if needed instead of moving hundreds of files.
5. Tie every retained subsystem to one of these gates:
   - general capability,
   - long-context capability,
   - AMC-Memory,
   - runtime safety,
   - serving/CLI usability.

## Next smallest useful cleanup

Completed in this pass: tracked `.bak` backup artifacts were removed after an import-reference scan found no runtime consumers. The old rollback documentation now points to git history instead of in-tree backup files.

The next lowest-risk cleanup is not another deletion. It is to keep `docs/CORE_SURFACE.md` current as the release-critical AMC-first boundary and then remove future clutter only with the same import-reference/test-validation sequence.
