# Aurelius Canonical Interface Contract
**Date:** 2026-04-21  
**Status:** Draft  
**Scope:** CLI, IDE, web, plugin, and background-agent surfaces

Machine-readable companions:
- `docs/plans/2026-04-21-aurelius-canonical-interface-contract.json`
- `docs/plans/2026-04-21-aurelius-canonical-interface-contract.schema.json`
- `docs/plans/2026-04-21-aurelius-canonical-interface-contract.prompt.yaml`

## 1. Purpose

Aurelius should expose one canonical agent interface that can be rendered
through multiple hosts:

- a terminal/CLI surface like Codex CLI,
- an IDE surface like Cline,
- a plugin/delegation surface like `codex-plugin-cc`,
- a capability-pack surface like OpenAI Skills,
- a multi-channel gateway surface like OpenClaw,
- and a background-job surface for autonomous runs.

The goal is not pixel-level cloning of any single product. The goal is a
shared contract for:

- instruction layering,
- mode selection,
- approval and tool gating,
- checkpoint/resume,
- subagent delegation,
- skill loading,
- and auditable task completion.

This contract is the behavioral boundary above the existing Aurelius model,
backend, serving, and agent primitives.

## 2. Contract Principles

1. One task = one thread.
2. One thread = one stable id, one timeline, one checkpoint history.
3. No silent fallbacks. Unsupported operations must fail loudly.
4. Human approval is explicit and stateful.
5. Skills are installable capability bundles, not ad hoc prompt blobs.
6. Modes are policy presets, not model-config booleans.
7. Tool use is auditable and replayable.
8. Background work must be resumable or cancelable.
9. Host adapters may differ, but the core thread semantics must not.
10. Model family identity, tokenizer contract, checkpoint format, backend
    contract, and release-track policy remain separate from user-facing modes.

## 3. Canonical Nouns

### 3.1 Task Thread

A `TaskThread` is the top-level unit of work.

Fields:

- `thread_id`: stable UUID-like identifier
- `title`: short human label
- `mode`: selected mode name
- `status`: `draft`, `pending_approval`, `running`, `blocked`, `completed`,
  `failed`, `canceled`
- `created_at`, `updated_at`
- `host`: `cli`, `ide`, `web`, `plugin`, or `api`
- `workspace`: repo path, worktree path, or remote workspace descriptor
- `instruction_stack`: ordered list of loaded instruction layers
- `skills`: ordered list of active skill bundles
- `approvals`: outstanding approval requests and their decisions
- `checkpoints`: ordered durable snapshots
- `steps`: ordered event log

### 3.2 Mode

A `Mode` is a policy preset that controls prompt framing, tool access, and
approval behavior.

Canonical draft modes:

- `ask`
- `code`
- `debug`
- `architect`
- `review`
- `background`
- `chat`

Modes are not tied to a single host. Each host may present modes differently,
but all hosts must map to the same mode names and semantics.

### 3.3 Instruction Layer

An `InstructionLayer` is a text or structured rule source that contributes to
the task thread prompt.

Canonical layers, in order of precedence:

1. system policy
2. user task prompt
3. repo instructions
4. workspace instructions
5. mode instructions
6. skill instructions
7. thread memory / checkpoints

The contract is that higher-precedence layers may constrain or override lower
layers, but lower layers must still be preserved for auditing.

### 3.4 Skill

A `Skill` is a reusable capability bundle.

Canonical fields:

- `skill_id`
- `name`
- `description`
- `scope` (`global`, `org`, `repo`, `thread`)
- `instructions`
- `scripts`
- `resources`
- `entrypoints`
- `version`
- `provenance`

Skills are intended to cover the same design space as OpenAI Skills and the
workflow packs used by Cline.

OpenClaw adds two important extensions to this idea:

- workspace-scoped skills that live with the assistant runtime.
- registry-backed skill publishing and search via ClawHub.
- archived skill snapshots in `openclaw/skills` for provenance and audit.
- injected instruction files such as `AGENTS.md`, `SOUL.md`, and `TOOLS.md`.

Those instruction files should be treated as first-class layers in Aurelius,
not as an implementation detail of one host.

### 3.5 Approval

An `Approval` is a stateful decision requested from a human before a risky
operation proceeds.

Approval categories:

- file write
- file delete
- shell command
- network request
- browser navigation
- MCP tool call
- external process launch
- model or checkpoint mutation

An approval must be explicit, scoped, and recorded in the thread timeline.

### 3.6 Checkpoint

A `Checkpoint` is a durable snapshot that can be resumed later.

Minimum contents:

- thread metadata
- instruction stack snapshot
- active skills
- workspace path or worktree id
- tool/approval state
- step log
- last model response
- last tool result
- compact memory summary

### 3.7 Subagent

A `Subagent` is a nested task thread spawned to handle a bounded subtask.

It inherits:

- task context,
- approval policy,
- workspace scope,
- and skill visibility.

It does not inherit mutable execution state by default.

### 3.8 Background Job

A `BackgroundJob` is a task thread detached from the foreground interaction.

It must support:

- status polling,
- result retrieval,
- cancellation,
- and optional resume.

### 3.9 Tool Call

A `ToolCall` is a structured action emitted by the model and validated by the
runtime before execution.

Tool calls are transport-specific on the wire, but normalized internally to a
single schema with:

- `tool_name`
- `arguments`
- `call_id`
- `host_step_id`
- `status`

## 4. Canonical Lifecycle

### 4.1 Create

The host creates a thread with:

- task prompt,
- chosen mode,
- workspace pointer,
- optional attached skills,
- optional repository instructions,
- and optional parent checkpoint.

### 4.2 Plan

The runtime may produce an explicit plan before taking actions.

Plans are advisory but must be recorded. A thread may be in:

- plan-first mode,
- react mode,
- or hybrid mode.

### 4.3 Approve

If the next action crosses an approval boundary, the thread enters
`pending_approval`.

The host must be able to render:

- what will happen,
- why it is risky,
- what data it touches,
- and what will happen if denied.

### 4.4 Act

The model emits a tool call or a final answer.

Tool calls must be validated against the registry and policy before execution.

### 4.5 Observe

Tool output is recorded as an immutable observation event.

The runtime may feed the observation back into the next prompt turn.

### 4.6 Checkpoint

Threads may checkpoint:

- on user request,
- on step boundaries,
- on significant state change,
- before long-running jobs,
- and before handoff across hosts.

### 4.7 Resume

A resumed thread must preserve:

- identity,
- history ordering,
- mode,
- attached skills,
- and checkpoint lineage.

### 4.8 Complete

Completion is either:

- successful final answer,
- explicit failure,
- cancellation,
- or policy denial.

Every completion must have a final status and a final artifact.

## 5. Mode Semantics

### 5.1 Ask

Low-risk clarification mode.

- minimal tool access
- no write operations
- no background job creation by default

### 5.2 Code

Implementation mode.

- file editing enabled
- test execution allowed
- approval required for risky writes and external side effects

### 5.3 Debug

Failure-analysis mode.

- prioritizes diagnosis and reproduction
- may run tests and inspect logs
- write access gated by approval

### 5.4 Architect

Planning / decomposition mode.

- no writes by default
- produces design, sequencing, and task breakdowns

### 5.5 Review

Audit mode.

- read-only by default
- focuses on risks, regressions, missing tests, and policy issues

### 5.6 Background

Long-running autonomous mode.

- must emit progress updates
- must expose cancel / result / status
- must checkpoint periodically

### 5.7 Chat

General conversation mode.

- low-friction interaction
- may escalate into other modes when the user requests work

## 6. Skill Contract

Skills should look like installable folders or packages:

```text
skill-name/
  skill.md
  scripts/
  resources/
  tests/
  metadata.json
```

Minimum expectations:

- portable instructions
- scriptable actions
- explicit provenance
- versioned compatibility
- clear scope

Skill loading order:

1. system policy
2. repo instructions
3. active mode rules
4. explicit skill packs
5. local thread memory

## 7. Approval Contract

Approval requests must include:

- action summary
- affected paths or resources
- why the action is risky
- whether it is reversible
- and the minimum required scope

Approval decisions must be one of:

- `allow`
- `deny`
- `allow_once`
- `allow_for_thread`
- `allow_for_scope`

The runtime must never infer approval from silence.

## 8. Checkpoint Contract

A checkpoint must support:

- serialization
- restoration
- diffing
- lineage tracking
- and resume validation

Recommended checkpoint state:

- thread metadata
- current mode
- mode-specific parameters
- active skill set
- instruction stack hash
- workspace hash or path
- last completed step
- pending approval requests
- active background job ids
- compact memory summary

## 9. Host Adapter Contract

### 9.1 CLI Adapter

Must support:

- mode selection
- thread creation
- task submission
- checkpoint save / load
- status and result commands
- cancel
- skills listing and activation

### 9.2 IDE Adapter

Must support:

- inline approvals
- file diff review
- checkpoint restore
- multi-root workspace support
- subagent spawning
- background job control

### 9.3 Web Adapter

Must support:

- thread list
- thread detail
- status timeline
- checkpoints
- approvals
- result view

### 9.4 Plugin Adapter

Must support:

- delegated review
- status
- result
- cancel
- and background job visibility

### 9.5 Multi-Channel Gateway Adapter

Must support:

- inbound message routing across multiple channels or accounts
- per-channel and per-peer isolation
- pairing / allowlist policies for untrusted senders
- workspace-scoped assistant sessions
- voice or rich-media hooks where the host supports them
- live canvas or companion-surface rendering where applicable
- channel-aware reply delivery

## 10. Relationship to Existing Aurelius Modules

Aurelius already has most of the substrate for this contract:

- agent loops: [src/agent/react_loop.py](/Users/christienantonio/Desktop/Aurelius/src/agent/react_loop.py), [src/agent/plan_and_execute.py](/Users/christienantonio/Desktop/Aurelius/src/agent/plan_and_execute.py), [src/agent/budget_bounded_loop.py](/Users/christienantonio/Desktop/Aurelius/src/agent/budget_bounded_loop.py)
- tool and MCP plumbing: [src/agent/tool_call_parser.py](/Users/christienantonio/Desktop/Aurelius/src/agent/tool_call_parser.py), [src/agent/tool_registry_dispatcher.py](/Users/christienantonio/Desktop/Aurelius/src/agent/tool_registry_dispatcher.py), [src/agent/mcp_client.py](/Users/christienantonio/Desktop/Aurelius/src/agent/mcp_client.py)
- repo context packing: [src/agent/repo_context_packer.py](/Users/christienantonio/Desktop/Aurelius/src/agent/repo_context_packer.py)
- serving: [src/serving/api_server.py](/Users/christienantonio/Desktop/Aurelius/src/serving/api_server.py), [src/serving/function_calling_api.py](/Users/christienantonio/Desktop/Aurelius/src/serving/function_calling_api.py), [src/serving/continuous_batching.py](/Users/christienantonio/Desktop/Aurelius/src/serving/continuous_batching.py)
- UI primitives: [src/ui/ui_surface.py](/Users/christienantonio/Desktop/Aurelius/src/ui/ui_surface.py), [src/ui/motion.py](/Users/christienantonio/Desktop/Aurelius/src/ui/motion.py), [src/ui/welcome.py](/Users/christienantonio/Desktop/Aurelius/src/ui/welcome.py)
- model identity and release governance: [src/model/family.py](/Users/christienantonio/Desktop/Aurelius/src/model/family.py), [src/model/manifest_v2.py](/Users/christienantonio/Desktop/Aurelius/src/model/manifest_v2.py), [src/model/release_track_router.py](/Users/christienantonio/Desktop/Aurelius/src/model/release_track_router.py), [src/model/compatibility.py](/Users/christienantonio/Desktop/Aurelius/src/model/compatibility.py), [src/model/manifest.py](/Users/christienantonio/Desktop/Aurelius/src/model/manifest.py)

OpenClaw is the main reference for the gateway-shaped assistant product:

- multi-channel inbox and account routing,
- workspace-scoped skills,
- injected instruction files (`AGENTS.md`, `SOUL.md`, `TOOLS.md`),
- onboarding-driven setup,
- and optional companion surfaces such as canvas and voice.

Related OpenClaw ecosystem repos add adjacent interface pieces:

- `clawhub`: public skill registry with moderation hooks, vector search, and a native package catalog.
- `skills`: archived skill history sourced from ClawHub for provenance and analysis.
- `acpx`: headless ACP client with persistent sessions, named workstreams, and cooperative cancel.
- `lobster`: typed local-first workflow shell with JSON-first pipelines, approval gates, and reusable macros.
- `openclaw-ansible`: hardened installer and provisioning path with Docker, Tailscale, UFW, and fail2ban.
- `ironclaw`: Rust reimplementation focused on privacy and security, with WASM sandboxes, persistent memory, routines, and plugin architecture.

What is still missing is the product-facing layer that unifies those pieces into
one thread model and exposes it through CLI / IDE / web / plugin adapters.

## 11. Non-Goals

This contract does not define:

- the model architecture,
- training loops,
- tokenizer internals,
- or backend implementation details.

Those are governed by the family and backend contracts already in Aurelius.

## 12. Implementation Sequence

If this contract is adopted, the next build order should be:

1. thread and checkpoint dataclasses
2. mode registry
3. skill pack loader
4. approval state machine
5. background job controller
6. CLI adapter
7. IDE adapter
8. web/plugin adapters

## 13. Open Questions

- Should mode names be fixed or allow aliases?
- Should checkpoints be file-based, database-backed, or both?
- Should skills be loaded from repo-local folders, a global registry, or both?
- Should background jobs always map to threads, or can they exist independently?
- Should approvals expire automatically or persist until explicit action?
