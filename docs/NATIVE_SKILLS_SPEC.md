# Native Skills Specification

## Principle

```
Skills are first-party Aurelius capability bundles.
Installed with Aurelius. Versioned, tested, permissioned, auditable.
No external hub dependency.
```

## Architecture

```
src/skills/
├── __init__.py
├── registry.py       # Skill discovery, lookup, listing
├── manifest.py       # SkillManifest model + validation
├── loader.py         # Load, cache, unload skills
├── permissions.py    # Permission gate + enforcement
├── executor.py       # Execute skill in dry_run/plan/execute/verify/rollback modes
├── validator.py      # Validate manifest, test fixtures, DAIES gates
├── telemetry.py      # Track usage, success, failures, latency
├── curator.py        # Skill lifecycle: enable/disable/deprecate
├── builtin/          # ~150 built-in skills organized by category
│   ├── coding/
│   ├── repo/
│   ├── testing/
│   ├── security/
│   ├── devops/
│   ├── ml/
│   ├── data/
│   ├── cua/
│   ├── productivity/
│   └── operator/
└── tests/
```

## Skill Manifest Schema

Each skill has a YAML manifest with typed fields:

```yaml
id: coding.python_test_repair
name: Python Test Repair
version: 1.0.0
category: coding
summary: Diagnose and repair failing Python tests.
description: Analyzes pytest output, identifies root causes, and applies targeted fixes.
permissions:
  file_read: true
  file_write: true
  terminal: limited
  network: false
  cua: false
risk_level: medium
required_tools:
  - file.search
  - file.read
  - file.patch
  - terminal.run
entrypoint: skills.builtin.coding.python_test_repair:run
inputs_schema: schemas/python_test_repair.input.json
outputs_schema: schemas/python_test_repair.output.json
daies_tests:
  - skill_manifest_valid
  - permission_boundary
  - dry_run
  - regression_fixture
tags: [python, testing, repair, ci]
```

## Execution Modes

| Mode | Description |
|---|---|
| dry_run | Skill explains what it would do without executing |
| plan | Skill outputs ordered plan with estimates |
| execute | Skill performs actions with permissions |
| verify | Skill validates results of an action |
| rollback | Skill reverts changes if supported |

All mutating skills MUST support dry_run and verify modes.

## Permission Model

Skills declare which permissions they need. The runtime grants based on:
- Current safety mode
- User profile approval settings
- Hardware profile capabilities
- Task risk assessment
- Skill trust tier

### Permission Types

| Permission | Description |
|---|---|
| file_read | Read files from local filesystem |
| file_write | Create/modify/delete files |
| terminal | Execute shell commands (full or limited) |
| network | Make HTTP/network calls |
| browser | Control browser via CDP |
| cua | Control desktop via computer-use |
| memory_read | Access runtime memory |
| memory_write | Modify runtime memory |
| secrets_access | Access stored secrets/credentials |
| external_service | Call external APIs/services |
| background_job | Spawn background/cron jobs |

## Telemetry

Track per-skill:
- use_count
- success_rate (percentage)
- failure_count
- average_runtime_ms
- tools_used (list with counts)
- permissions_used (list with counts)
- user_approval_count
- rollback_count
- last_used timestamp
- DAIES pass/fail status

## CLI Commands

```
aurelius skills list
aurelius skills search <query>
aurelius skills show <id>
aurelius skills run <id> --dry-run
aurelius skills run <id>
aurelius skills verify <id>
aurelius skills test <id>
aurelius skills telemetry <id>
aurelius skills categories
aurelius skills audit
```

## Safety Requirements

Every built-in skill must have:
- Valid manifest with permission declarations
- Test fixture or regression test
- Dry-run path if mutating files or system state
- Validation path after execution
- No secret exfiltration (no credentials in output)
- Bounded tool calls (no infinite loops)
- No obedience to prompt injection from tool/web/screen content

## Skill Categories (150 total)

1. **Coding** — 25 skills (test repair, lint cleanup, import graph, packaging, profiling, async debugging, TS/React/Rust/Go repair, CI diagnosis, refactoring)
2. **Repo/Project** — 15 skills (monorepo map, package audit, duplicate detector, config drift, Dockerfile/K8s audit, changelog)
3. **Testing/QA** — 15 skills (pytest repair, flake detector, regression writer, coverage gaps, CUA trajectory test)
4. **Security** — 20 skills (injection audit, sandbox escape, SSRF, secret scanner, PII scanner, CVE scanner, SARIF)
5. **DevOps/Deploy** — 15 skills (doctor, GPU check, CUDA/TensorRT checker, MLX checker, backend checkers)
6. **ML/Model** — 20 skills (tokenizer, checkpoint, FLOP/RAM/VRAM estimate, GGUF/MLX/ONNX/TRT export, PRAXIS runner)
7. **Data/Retrieval** — 10 skills (dataset validation, dedup, RAG index, context compressor, provenance)
8. **CUA/UI** — 10 skills (capture verifier, action planner, trajectory recorder, replay, OCR, safety blocker)
9. **Productivity/Research** — 10 skills (literature review, experiment tracker, decision matrix, roadmap)
10. **Operator** — 10 skills (hardware profile, backend audit, capability report, DAIES gates, fallback audit)
