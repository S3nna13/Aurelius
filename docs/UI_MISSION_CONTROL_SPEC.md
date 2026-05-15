# Mission Control UI Specification

## Purpose

Mission Control is the observability and management interface for Aurelius v2. It is the secondary product surface after CLI, providing visual monitoring, configuration, and audit capabilities.

## UI Routes

```
/dashboard           Overview: model status, RAM/VRAM, active sessions
/models              Swift/Forge/Atlas + artifacts + backends + quantization
/hardware            Hardware detection and profile recommendation
/backends            Backend status, selection, configuration
/capabilities        Capability report (requested vs actual)
/daies               DAIES dashboard: gate results, validation status
/cua                 Computer-use dashboard
/cua/traces/:id      CUA trace replay with screenshots
/exports             Artifact export status and history
/profiles            Runtime, hardware, and user profile manager
/checkpoints         Workspace checkpoint/rollback UI
/approvals           Pending action approvals
/memory              Runtime memory with quarantine view
/skills              Native 150-skill catalog
/skills/:id          Skill details: permissions, tests, telemetry
/chat                Interactive chat with execution metadata
/agents              Agent workspace and trace viewer
/logs                Application log viewer with search
/traces              Tool/agent/CUA execution trace viewer
/settings            Configuration management
/playground          Model playground for testing
```

## UI Must Show

- Requested model vs actual model
- Backend and artifact being used
- Quantization level
- Execution mode (local/remote/split/verifier)
- Max context in use
- RAM/VRAM usage with bar chart
- Enabled vs disabled capabilities
- Native skill status (loaded count, failures)
- DAIES gate status
- Safety status (blocked actions, warnings)
- Memory quarantine items
- CUA blocking reasons
- Trace provenance

## UI Must NOT

- Expose secrets or credentials in any response
- Show mocked/stubbed features as live
- Hide model fallback from user
- Imply full local CUA when only verifier mode exists
- Imply external skill hub dependency
