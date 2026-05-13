# Gap Ledger — Aurelius Integrations Loop v10

## Completed Slices

### Slice 1: skills_registry.py
- **Created**: `skills_registry.py` — 7 registry entries, contract wrapper, verify_contract
- **Tests**: 4 new (test_skills_registry_*)
- **Bugfix**: `skills.py:SkillRetriever` — missing `self.skill_dim` attribute
- **Docs**: Updated AURELIUS_SKILL.md

### Slice 2: agent_registry.py
- **Created**: `agent_registry.py` — 12 registry entries, 6 contract wrappers
- **Tests**: 10 new (test_agent_registry_*)
- **Bugfix**: `agent_core.py:ValueHead` — handle 2D input in forward()
- **Bugfix**: `agent_loop.py:write_to_memory` — handle batched (B, D) written tensor
- **Bugfix**: `agent_loop.py:reflect` — handle multi-batch critic score with .mean()
- **Docs**: Updated AURELIUS_SKILL.md

### Slice 3: api_registry.py
- **Created**: `api_registry.py` — 14 registry entries, 5 contract wrappers
- **Tests**: 4 new (test_api_registry_*)
- **Docs**: Updated AURELIUS_SKILL.md

### Slice 4: tool_schema_registry.py
- **Created**: `tool_schema_registry.py` — 13 registry entries, verify_imports()
- **Tests**: 4 new (test_tool_schema_registry_*)
- **Docs**: Updated AURELIUS_SKILL.md

## Bugs Fixed
| Bug | File | Line | Fix |
|-----|------|------|-----|
| Missing self.skill_dim | skills.py | 43 | Added `self.skill_dim = skill_dim` |
| ValueHead 2D crash | agent_core.py | 101 | Added `if h.dim() == 3: h = h[:, -1]` |
| write_to_memory batch crash | agent_loop.py | 110 | Added `if written.dim() > 1: written = written[0]` |
| reflect multi-batch crash | agent_loop.py | 63 | Changed `.item()` to `.mean().item()` |

## Test Suite
- **Before**: 110 tests
- **After**: 132 tests (+22)
- **Status**: 132/132 pass, 40 Python files syntax-OK

## Remaining Gaps (Next Slices)

### High Priority
1. `aurelius/serving/` or `src/serving/` — directories don't exist, referenced in Workstream A as `api_server.py` and `aurelius_api.py`. Need to define what the serving layer should be.
2. `middle/src/routes/` — referenced but doesn't exist. No `chat.ts`, `registry.ts`, `brain.ts`, or `provider_router.ts`.
3. No endpoint routing / HTTP API surface exists yet.

### Medium Priority
4. `agent_registry.py` — could add `AgentMemoryBridgeContract._verify_write_shape` for multi-batch edge cases.
5. `api_registry.py` — `TrainContract.verify_contract` depends on specific model configs; could add config validation for all 4 size variants.

### Low Priority
6. No Docker/compose file for serving.
7. No CI workflow in `.github/`.
8. Rust bridge (`rust_bridge.py`) — no tests exist for it.
9. `brain_layer.py` — no tests.

## How to Continue
```
1. Pick the next slice from "Remaining Gaps"
2. Run `make test` for baseline
3. Implement contract surface or fix
4. Run `make test && make check`
5. Update this ledger
```
