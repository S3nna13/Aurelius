# Aurelius v2

> Three models. One protocol. Native skills. Adaptive execution everywhere. DAIES decides what ships.

Aurelius v2 is a **frontier AI system** — a three-model LLM family and runtime platform with a fully functioning CLI, production UI (Mission Control), RAM/VRAM-aware execution, computer-use/CUA with safety verifiers, a built-in library of **150 native skills**, and DAIES governance.

Unlike v1, Aurelius v2 is not just a checkpoint. It is a complete AI system:

```
Aurelius v2
├── Model Cognition Plane
│   ├── Aurelius Swift (~0.6B) — edge, router, verifier, fallback
│   ├── Aurelius Forge (~3B) — default local agent, coding, CUA
│   └── Aurelius Atlas (~32B MoE) — frontier reasoning, orchestration
│
├── Runtime Capability Plane
│   ├── tools, native skills, memory, CUA
│   ├── cron, sessions, checkpoints, delegation
│   ├── profiles, telemetry, DAIES validation
│   └── MCP, gateway, approvals
│
├── Native Skill Plane — 150 built-in skills
│   ├── coding (25), repo (15), testing (15)
│   ├── security (20), devops (15), ml (20)
│   ├── data (10), cua (10), productivity (10), operator (10)
│   └── registry, permission gates, telemetry, curator
│
├── Product Surfaces
│   ├── CLI (first-class daily driver)
│   ├── API/Runtime server (truth layer)
│   └── Mission Control UI (observability)
│
└── Hardware Adaptation Layer
    ├── Jetson Nano/Orin/AGX profiles
    ├── Mac Silicon 8GB → Ultra profiles
    ├── RTX 8GB → PRO 6000 Blackwell 96GB
    └── remote/split execution profiles
```

## Quick Start

```bash
# System health check
aurelius doctor

# Detect hardware, recommend profile
aurelius hardware detect

# Interactive chat
aurelius chat

# List 150 native skills
aurelius skills list

# Quick DAIES validation
aurelius daies quick

# Show current status
aurelius status

# Start API server
aurelius serve --port 8000

# Open Mission Control UI
aurelius ui
```

## The Three Models

| Model | Size | Purpose | Target Hardware |
|---|---|---|---|
| **Aurelius Swift** | ~0.6B dense | Edge, router, verifier, fallback | Jetson Nano+, Mac 8GB+, any GPU |
| **Aurelius Forge** | ~3B dense/hybrid | Default local agent, coding, CUA | Mac 16GB+, RTX 8-24GB+, Jetson Orin+ |
| **Aurelius Atlas** | ~32B total / ~8B active MoE | Frontier reasoning, orchestration | RTX 6000+, Blackwell, Mac Ultra, remote |

No additional model names. Hardware variation is handled through artifacts, profiles, backends, quantization, offload, and split execution — **not** additional public model names.

### Universal Execution Model

Every machine can request every model. The runtime chooses the execution mode:

| Mode | Meaning |
|---|---|
| `Native Local` | Model fits and runs locally |
| `Quantized Local` | Local reduced-precision model |
| `Offloaded Local` | Weights/KV split between accelerator and RAM |
| `Split Local/Remote` | Local tools/CUA/memory, remote inference |
| `Remote Hosted` | Model entirely remote through Aurelius API |
| `Verifier/Controller Only` | Local device verifies and routes but cannot run model |

**No silent fallback**: the runtime always reports requested vs actual model, backend, artifact, and execution mode.

### Hardware Ladder

| Hardware | Swift | Forge | Atlas |
|---|---|---|---|
| Jetson Nano 2/4GB | native Q3/Q4 | split/remote | split/remote |
| Jetson Orin Nano 8GB | native | quantized/split | split/remote |
| Mac Silicon 8GB | native | quantized | split/remote |
| Mac Silicon 16GB | native | Q4 local | offload/split |
| Mac Silicon 32GB | native | native Q4 | Q4 offload |
| Mac Ultra 128GB+ | native | native | native Q4 |
| RTX 8-12GB | native | Q4/offload | split/remote |
| RTX 24GB | native | native Q4 | Q4 offload |
| RTX 6000 Ada 48GB | native | native | Q4/native |
| RTX PRO 6000 Blackwell 96GB | native | native | native FP4 |

## RAM/VRAM-Aware Runtime

Aurelius never lets OOM be the first fallback. The system monitors memory before, during, and after model loading, and degrades gracefully through an **11-step degradation ladder**:

1. Unload inactive skills
2. Reduce batch size to 1
3. Reduce context budget (128K → 64K → 32K → 16K)
4. Quantize KV cache (KIVI-style 2/4-bit)
5. Shrink active prompt
6. Evict prefix cache
7. Offload cold KV to CPU/unified memory
8. Downshift or disable vision encoder
9. Switch to smaller model artifact
10. Switch to split/remote execution
11. Ask user before cancelling

Memory policies: `conservative`, `balanced`, `performance`, `frontier`.

## 150 Native Skills

Aurelius ships with **150 built-in skills** — no external skill hub dependency. Skills are first-party capability bundles: installed with Aurelius, versioned, tested, permissioned, and auditable.

### Skill Categories

| Category | Count | Examples |
|---|---|---|
| **Coding** | 25 | Python test repair, lint cleanup, async debugging, Rust compile repair |
| **Repo/Project** | 15 | Monorepo map, Dockerfile audit, config drift, changelog generator |
| **Testing/QA** | 15 | Pytest repair, regression writer, Playwright builder, safety red-team |
| **Security** | 20 | Prompt injection audit, sandbox escape, CVE scanner, threat model |
| **DevOps** | 15 | Doctor, GPU check, CUDA/TensorRT checker, K8s manifest audit |
| **ML/Model** | 20 | Tokenizer validator, checkpoint converter, FLOP estimator, PRAXIS runner |
| **Data/Retrieval** | 10 | Dataset dedup, RAG index, context compressor, provenance checker |
| **CUA/UI** | 10 | Capture verifier, action planner, trajectory recorder, screen OCR |
| **Productivity** | 10 | Literature review, experiment tracker, roadmap generator, PDF extractor |
| **Operator** | 10 | Hardware profile, DAIES gates, fallback auditor, quarantine reviewer |

### Skill Execution Modes

Every skill supports: `dry_run`, `plan`, `execute`, `verify`, `rollback`. All mutating skills **must** support dry_run.

### Skill Permission Model

Skills declare permissions: `file_read`, `file_write`, `terminal`, `network`, `browser`, `cua`, `memory_read/write`, `secrets_access`, `external_service`, `background_job`. The runtime grants/denies based on safety mode, user profile, hardware profile, task risk, and skill trust tier.

## CLI Commands

```
aurelius chat                 # Interactive chat
aurelius run <prompt>         # One-shot task
aurelius agent               # Autonomous task mode
aurelius computer             # CUA / computer-use
aurelius models               # List/load/export model artifacts
aurelius family               # Swift/Forge/Atlas management
aurelius backend              # Runtime backend status
aurelius hardware             # Detect hardware, recommend profile
aurelius profile              # Manage runtime profiles
aurelius memory               # Inspect/search runtime memory
aurelius skills list          # List native skills
aurelius skills run <id>      # Execute a skill
aurelius skills audit         # Audit all skills
aurelius tools                # List/manage tools
aurelius mcp                  # MCP server management
aurelius cron                 # Scheduled jobs
aurelius sessions             # Session management
aurelius checkpoint           # Workspace snapshots
aurelius serve                # Start API server
aurelius ui                   # Open Mission Control
aurelius train                # Training workflows
aurelius eval                 # Evaluation suites
aurelius daies quick          # Quick validation gate
aurelius export               # Model artifact export
aurelius doctor               # Full system health check
aurelius logs                 # Application logs
aurelius traces               # Execution traces
aurelius status               # Current system status
aurelius config               # Configuration management
```

### Interactive Status Bar

```
Aurelius Forge | local mlx q4 | ctx 32K | RAM 14.2/32GB | CUA local_full | skills 150 | tools 18 | profile mac_silicon_32gb
```

## Computer Use (CUA)

Aurelius CUA provides verified, auditable desktop automation with **hardware-adaptive safety**:

| Mode | Description | Hardware |
|---|---|---|
| `verifier_only` | Validates actions, does not execute | All |
| `remote_driver` | Local sends plans to remote desktop | Split |
| `browser_only` | Browser automation via CDP | All |
| `local_basic` | Click/type/scroll with strict verifier | Mac/RTX |
| `local_full` | Screenshot + AX tree + background driver | Mac/RTX |
| `multimodal_full` | Full CUA with vision/GUI reasoning | RTX 6000+ |

### CUA Safety Gates

- **Permission dialog detection** — blocked
- **Password/secret entry** — blocked
- **Payment/checkout** — blocked
- **Destructive UI** — requires confirmation
- **Before/after screenshot logging** — every action
- **Action verifier** — every step validated
- **Audit trail** — immutable log of all actions

## Architecture

```
aurelius/
├── aurelius_cli/              # CLI v2
│   ├── main.py                # Main entry (1329 lines)
│   └── v2_cli.py              # V2 command layer
├── docs/                      # Contract documents (56+ specs)
│   ├── MODEL_CARD_V2.md       # Three-model specs
│   ├── CLI_V2_SPEC.md         # CLI specification
│   ├── NATIVE_SKILLS_SPEC.md  # Skill system spec
│   ├── DAIES_V2_GATES.md      # Validation gates
│   ├── RAM_VRAM_POLICY.md     # Memory budget policy
│   ├── CUA_PROTOCOL_V2.md     # CUA protocol
│   ├── EXPORT_CONTRACTS.md    # Export specs
│   └── ...
├── src/
│   ├── runtime/               # Phase 1: Runtime foundation
│   │   ├── profile_schema.py  # Profile data models
│   │   ├── hardware_detector.py # Hardware detection
│   │   ├── memory_budget.py   # RAM/VRAM budget + degradation
│   │   ├── capability_report.py # Capability truth reports
│   │   └── backend_selector.py  # Backend selection logic
│   ├── skills/                # Phase 2: Native skill system
│   │   ├── manifest.py        # SkillManifest schema
│   │   ├── registry.py        # Discovery, lookup, loading
│   │   ├── permissions.py     # Permission gate + enforcement
│   │   ├── executor.py        # Execute in dry/plan/execute/verify
│   │   ├── validator.py       # DAIES validation
│   │   ├── telemetry.py       # Usage tracking + stats
│   │   ├── curator.py         # Enable/disable/deprecate
│   │   └── builtin/           # 150 skill manifests (JSON)
│   │       ├── coding/        # 25 coding skills
│   │       ├── repo/          # 15 repo skills
│   │       ├── testing/       # 15 testing skills
│   │       ├── security/      # 20 security skills
│   │       ├── devops/        # 15 devops skills
│   │       ├── ml/            # 20 ML skills
│   │       ├── data/          # 10 data skills
│   │       ├── cua/           # 10 CUA skills
│   │       ├── productivity/  # 10 productivity skills
│   │       └── operator/      # 10 operator skills
│   ├── api/                   # Phase 4: API truth layer
│   │   └── server.py          # FastAPI runtime server
│   ├── decision/              # Phase 6: Decision system
│   │   ├── decision_head.py   # Decisive action routing
│   │   └── action_heads.py    # ToolCall, Skill, CUA, etc.
│   ├── computer_use/          # Phase 7: CUA system
│   │   ├── driver_base.py     # ABC for all CUA drivers
│   │   ├── verifier.py        # Action safety verification
│   │   ├── audit_log.py       # Immutable audit trail
│   │   └── trajectory.py      # Recording and replay
│   ├── efficiency/            # Phase 8: Efficiency wiring
│   │   ├── kv_cache.py        # Paged KV, quantization, prefix
│   │   ├── attention.py       # Cross-layer, sinks, sparse
│   │   ├── prefill.py         # Chunked prefill scheduler
│   │   └── compression.py     # KV and context compression
│   └── export/                # Phase 9: Export contracts
│       └── converter.py       # GGUF, MLX, ONNX, TensorRT-LLM
└── ui/                        # Phase 5: Mission Control UI
    └── src/components/
        ├── HardwareDashboard.tsx
        └── SkillCatalog.tsx
```

## DAIES v2 Gates

DAIES (Decide, Adapt, Integrate, Evaluate, Scale) governs every Aurelius v2 feature. No feature ships without passing gates.

### Gate Categories

- **CLI Gates** — help accuracy, hardware detect, model load, no silent fallback
- **UI Gates** — model hub truth, RAM/VRAM health, no secret exposure
- **Skill Gates** — manifest valid, permission boundary, dry run, no exfiltration
- **RAM/VRAM Gates** — budget estimation, KV cache limits, no OOM first response
- **Frontier Gates** — code repair, long context, CUA completion, agent no loop

## Efficiency Stack

- **PagedKVCache** — non-contiguous block allocation (vLLM-style)
- **KVCacheQuantizer** — KIVI 2/4/8-bit quantization
- **PrefixCache** — shared system prompts, document prefixes
- **CrossLayerKVSharing** — reduces KV across adjacent layers
- **DynamicSparseAttention** — O(n*k) instead of O(n^2)
- **ChunkedPrefillScheduler** — manages prefill memory spikes
- **KVCacheCompressor** — PCA projection, token merging
- **ContextCompressor** — prompt shrinking under pressure
- **AttentionSinkManager** — stability for long contexts

## Training Pipeline

1. Tokenizer/control/skill protocol freeze
2. Base pretraining
3. Memory curriculum
4. Tool/action/skill SFT
5. CUA imitation
6. Model spec midtraining
7. PRAXIS alignment
8. Agent RL
9. Skill-use and skill-composition training
10. Distillation (Atlas → Forge → Swift)
11. Hardware export and DAIES validation

Context ramp: 4K → 8K → 16K → 32K → 64K → 128K → 256K → Atlas-only extended

## PRAXIS Alignment

PRAXIS remains the core alignment system:
- SteeringRewardCorrespondence
- ExpertSafetyAffinity
- MultiTokenAlignmentHorizon
- PrecisionFusion
- DAPO, KL penalty, constitutional gate
- WARP / model anchoring

**PRAXIS v2 additions:**
- DecisionHead reward
- Native skill selection reward
- CUA safety reward
- Memory contamination penalty
- Escalation calibration reward
- No-loop penalty
- Cost/risk/utility calibration
- Tool provenance reward
- Runtime honesty reward

## Project Structure Stats

| Component | Files | Description |
|---|---|---|
| Python source | 79+ | Runtime, skills, CUA, decision, efficiency, export, API |
| Skill manifests | 150 | JSON manifests across 10 categories |
| Documentation | 56+ | Specs, protocols, policies, models cards |
| Total LOC | ~15,000+ | Core v2 implementation |

## Configuration

Aurelius uses configuration files at `~/.aurelius/config.yaml`:

```yaml
preferred_model: forge
preferred_backend: mlx
max_tokens: 4096
approval_mode: inline
memory_policy: balanced
cua_mode: local_basic
safety_mode: balanced
remote_endpoint: ""
```

## License

[Add license]

## Contributors

Aurelius v2 — Built for the future of local AI.
