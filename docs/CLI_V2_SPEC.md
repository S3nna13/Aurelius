# CLI v2 Specification

## Principle

```
CLI first. UI is Mission Control, not the daily driver.
No silent fallback. Every command reports truthfully.
```

## Command Tree

```
aurelius
├── chat                 # Interactive chat/assistant mode
├── run                  # One-shot prompt/task
├── agent                # Autonomous task mode (ReAct + tools + memory)
├── computer             # CUA / computer-use mode
├── models               # List/pull/load/export model artifacts
├── family               # Swift/Forge/Atlas family management
├── backend              # Runtime backend status/selection
├── hardware             # Detect hardware, recommend profile
├── profile              # Manage hardware/runtime/user profiles
├── memory               # Inspect/search/import/export runtime memory
├── skills               # List/run/test/audit native skills
├── tools                # List/enable/disable/test tools
├── mcp                  # List/add/test MCP servers
├── cron                 # Manage scheduled/cron jobs
├── sessions             # List/resume/export/delete sessions
├── checkpoint           # Snapshot/rollback workspace state
├── serve                # Start API/runtime server
├── ui                   # Launch/open/build Mission Control UI
├── train                # Train/fine-tune/distill
├── eval                 # Run evaluation suites
├── daies                # Run DAIES validation gates
├── export               # Export GGUF/MLX/ONNX/TensorRT artifacts
├── doctor               # Full health/dependency/environment check
├── logs                 # Tail/search application logs
├── traces               # Inspect tool/agent/CUA execution traces
├── config               # View/set/edit configuration
└── quit                 # Exit
```

## Modes

| Mode | Description |
|---|---|
| `chat` | Normal assistant conversation |
| `code` | File + terminal + tests + checkpoints for software development |
| `agent` | Autonomous ReAct/workflow with tools, memory, skills |
| `computer` | CUA observe-act-verify loop |
| `research` | Web search + memory + provenance tracking |
| `operator` | Runtime/admin/system health management |
| `training` | Training/eval/export workflows |
| `safe` | No terminal execution, no file writes, no CUA |

## Slash Commands (within interactive sessions)

```
/help              Show available commands
/status            Show current model/backend/profile/RAM-VRAM status
/model [name]      Switch model (swift/forge/atlas)
/backend [name]    Switch backend
/profile [name]    Switch profile
/hardware          Show detected hardware
/capabilities      Show current capability report
/memory search <q> Search runtime memory
/skills            List native skills
/skill <id>        Show skill details
/tools             List available tools
/cua capture       Capture screenshot for CUA
/cua mode          Show/set CUA mode
/checkpoint        Create workspace checkpoint
/rollback          Rollback to last checkpoint
/daies quick       Run quick DAIES gate check
/daies full        Run full DAIES validation
/export [format]   Start model export
/serve             Runtime status
/ui                Open Mission Control
/logs              Show recent logs
/traces            Show recent traces
/config            View current configuration
/quit              Exit interactive session
```

## Interactive Status Bar

The CLI renders a persistent status bar:

```
Aurelius Forge | local mlx q4 | ctx 32K | RAM 14.2/32GB | CUA local_full | skills 150 | tools 18 | profile mac_silicon_32gb
```

```
Aurelius Atlas | remote | local controller | ctx 128K | RAM 3.1/4GB | CUA verifier_only | skills 150 | profile jetson_nano_4gb
```

## Safety UX

### Permission Cards

All risky actions require explicit approval:

```
Permission required
Action: write file
Target: /path/to/file.py
Risk: medium
Checkpoint: will create before action
Approve? yes / no / always-this-session
```

### CUA Blocking

CUA risky actions are blocked at the driver level:

```
Computer-use action blocked
Reason: password/payment/permission/destructive UI detected
Next: user takeover or explicit safe alternative required
```

## Definition of Done

CLI is complete when:
- `aurelius --help` is accurate for all commands
- `aurelius doctor` detects hardware/backends/artifacts
- `aurelius hardware detect` produces profile recommendation
- `aurelius profile use` switches runtime profile
- `aurelius models list/load` works with local artifacts
- `aurelius backend list/select` works
- `aurelius skills list/run/test/audit` works
- `aurelius chat` works with live model or clearly labeled mock
- `aurelius run` supports one-shot tasks with full metadata
- `aurelius agent` supports autonomous task mode
- `aurelius computer capture` works in available CUA mode
- `aurelius daies quick` runs validation gates
- `aurelius export` validates export contracts
- `aurelius serve` starts API/runtime server
- `aurelius ui open` launches Mission Control
- RAM/VRAM status is visible in status bar
- No silent fallback — all substitutions are labeled
