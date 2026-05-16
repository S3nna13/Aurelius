# AMC-First Focused Build Implementation Plan

> Scope: keep implementation inside the original Aurelius repo. Do not merge the `aurelius-v2-backup` branch wholesale. Cherry-pick ideas only after tests and architecture gates are green.

Goal: make Aurelius stable by centering the repository on the Aurelian Memory Core (AMC), proving it with benchmark gates, and reducing architecture sprawl.

Architecture: one canonical `src/` runtime, one memory facade, one evaluation path, and five alignment tracks. Model variants are unlocked by measured AMC capability rather than added up front.

Tech Stack: Python 3.12, pytest, ruff, existing `src/model`, `src/memory`, `src/eval`, and `src/alignment` packages.

---

## Phase 0: Repo guardrails

### Task 0.1: Protect v2 work

Objective: preserve the v2 state while keeping main pre-v2.

Files/commands:
- Verify branch: `git branch --list aurelius-v2-backup`
- Verify main HEAD is pre-v2: `git log --oneline -3`

Acceptance:
- `aurelius-v2-backup` exists.
- No v2-only docs or `src/skills/` tree are present on main.

Status: done in current working tree.

### Task 0.2: Restore green package/lint baseline

Objective: remove known pre-v2 instability before larger architecture work.

Files:
- `src/model/__init__.py`
- `src/alignment/__init__.py`
- `src/serving/__init__.py`
- `tests/test_package_consolidation.py`
- `src/serving/web_ui.py`
- `src/inference/code_execution.py`
- `src/security/request_throttler.py`
- `src/training/tst_trainer.py`

Commands:
- `.venv/bin/python -m pytest tests/test_package_namespace.py tests/test_package_consolidation.py tests/security/test_ssrf_guard.py -q --tb=short`
- `.venv/bin/python -m ruff check src/ --ignore=I001`

Acceptance:
- Targeted package/SSRF tests pass.
- Ruff passes on `src/` ignoring only import-sort debt.

Status: done in current working tree.

---

## Phase 1: Canonical architecture and benchmark gates

### Task 1.1: Add AMC-first architecture decision document

Objective: make Option C explicit and prevent drift back into module-count expansion.

File:
- `docs/AMC_FIRST_ARCHITECTURE.md`

Acceptance:
- Defines Tier 1/Tier 2/Tier 3 memory.
- Defines five-track alignment path.
- Defines keep/quarantine/delete categories.
- Defines benchmark gates.

Status: done in current working tree.

### Task 1.2: Add AMC benchmark scaffold

Objective: create deterministic, unit-testable tasks that evaluate memory-specific behavior without requiring a live model.

Files:
- Create: `src/eval/amc_memory_benchmark.py`
- Modify: `src/eval/benchmark_config.py`
- Test: `tests/eval/test_amc_memory_benchmark.py`
- Test: `tests/eval/test_benchmark_config.py`

Benchmark families:
- `cross_session_recall`: retrieve a value from an earlier session transcript.
- `surprise_gate_selectivity`: identify which observation should be written to episodic memory.
- `consolidation_preference`: choose the repeated/high-confidence memory over distractors.
- `contradiction_quarantine`: detect that contradictory memories require quarantine or verification.

Acceptance:
- Builders are deterministic by seed.
- Oracle answers pass at 1.0.
- Null answers score 0.0.
- Config registry contains an `AMC-Memory` spec with metric `exact_match`.

Status: done in current working tree.

### Task 1.3: Add benchmark config file

Objective: give future training/eval runs a single config for the Option C gate list.

File:
- Create: `configs/amc_first_benchmark.yaml`

Acceptance:
- Lists general benchmarks, long-context benchmarks, AMC-specific benchmarks, ablation variants, and pass/fail gates.

Status: done in current working tree.

---

## Phase 2: Memory consolidation

### Task 2.1: Inventory memory implementations

Objective: identify the smallest canonical memory surface.

Commands:
- `find src/memory -name '*.py' | sort`
- `find plugins/memory src/memory -name '*.py' | sort`

Acceptance:
- Produce a table: keep, merge, shim, delete.
- Identify one canonical facade (`src/memory/manager.py`) and one durable store path.

Status: pending.

### Task 2.2: Add AMC telemetry contract

Objective: ensure every memory write/retrieval can be inspected.

Files:
- Create or modify: `src/memory/telemetry.py`
- Tests: `tests/memory/test_amc_telemetry.py`

Events:
- `memory_write_rejected`
- `memory_write_accepted`
- `memory_retrieved`
- `memory_consolidated`
- `memory_quarantined`

Acceptance:
- Events serialize to plain dict/JSON.
- No torch dependency.
- Every event includes `session_id`, `tier`, `reason`, and timestamp.

Status: pending.

---

## Phase 3: Alignment narrowing

### Task 3.1: Freeze alignment scope

Objective: stop treating 80+ algorithms as the product surface.

Files:
- Create: `docs/ALIGNMENT_SCOPE.md`
- Optional tests: registry tests if alignment registry exposes deprecated algorithms.

Canonical tracks:
- SFT
- DPO
- GRPO/RLVR
- Constitutional memory
- Red-team quarantine

Acceptance:
- Docs identify non-canonical algorithms as experimental.
- Future user-facing docs point to these five tracks only.

Status: pending.

### Task 3.2: Constitutional memory policy format

Objective: store alignment policy as inspectable Tier 3 memory, not opaque prompt text.

Files:
- Create: `src/alignment/constitutional_memory.py`
- Test: `tests/alignment/test_constitutional_memory.py`

Acceptance:
- Policy entry has id, text, version, priority, provenance, and enabled flag.
- Retrieval sorts by priority and version.
- Serialization is stable.

Status: pending.

---

## Phase 4: Delete/quarantine sprawl

### Task 4.1: Produce deletion/quarantine ledger

Objective: make cleanup explicit before deleting code.

File:
- Create: `docs/AMC_CLEANUP_LEDGER.md`

Categories:
- Core now
- Port behind registry
- Experiment/archive
- Delete candidate

Acceptance:
- Every large non-core package has a classification and rationale.
- No code deleted until the ledger is reviewed.

Status: pending.

### Task 4.2: Move experiments behind registry flags

Objective: preserve research while reducing import-time fragility.

Files:
- Depends on ledger.

Acceptance:
- Importing `src` or core packages does not eagerly import heavy optional experiments.
- Optional features fail gracefully when dependencies are absent.

Status: pending.

---

## Phase 5: Full validation

Commands:
- `.venv/bin/python -m pytest tests/eval/test_amc_memory_benchmark.py tests/eval/test_benchmark_config.py -q`
- `.venv/bin/python -m pytest tests/test_package_namespace.py tests/test_package_consolidation.py tests/security/test_ssrf_guard.py -q --tb=short`
- `.venv/bin/python -m ruff check src/ --ignore=I001`

Full-suite strategy:
- Run by directory chunks to avoid timeout.
- Record pre-existing failures separately from new regressions.
- Do not claim full green unless every chunk completes.

Definition of complete for this pass:
- Option C architecture doc exists.
- AMC benchmark scaffold exists and passes unit tests.
- Targeted stability tests pass.
- Ruff passes for touched code and current `src/` baseline.
- Remaining work is documented as explicit follow-up, not hidden.
