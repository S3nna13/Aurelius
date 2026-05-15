# Security Threat Model v2

## Required Threat Coverage

| Threat Category | Specific Threats |
|---|---|
| Prompt Injection | Direct, indirect, skill injection, multi-step |
| Tool Exploitation | Call chain injection, transport risks, SSRF, unbounded execution |
| Token Attacks | Exhaustion, overflow, context stuffing |
| Privilege Escalation | Cross-agent contamination, permission bypass |
| Desktop Security | Unsafe actions, credential capture, destructive UI |
| Memory Security | Poisoning, contamination, secret leakage |
| File/Network | Path traversal, malicious local files, SSRF |
| Plugin/Extension | Malicious MCP server, skill hijack, plugin abuse |
| Supply Chain | Package vulnerabilities, compromised dependencies |

## Required Defenses

### Input Layer
- Capability-gated tools (each tool declares its access level)
- Native skill permission manifests (explicit permission declarations)
- Deterministic tool-call boundary enforcement
- Prompt injection detection in tool/web/screen content
- Token count limits per request

### Execution Layer
- Skill IR validation before execution
- Permission gate stress tests
- Runtime risk accumulator (blocks if risk exceeds threshold)
- Process isolation for untrusted code execution
- Path traversal validation on all file operations
- Private/reserved IP blocking for network calls
- Rate limiting on tool/skill invocations

### Memory Layer
- Memory quarantine for untrusted content
- Source classification (trusted/untrusted/unknown)
- Trust level assignment before memory writes
- Cross-agent message signing
- No obedience to embedded instructions from observations

### Output Layer
- Tool/skill result provenance tracking
- Explicit approval for high-risk actions
- Rollback/checkpoint before destructive operations
- Sandboxed failure = blocked action (fail-closed)

### UI Layer
- No secret exposure in UI responses
- Approval queue with risk details
- CUA blocking with explicit reason display
- Audit log viewer for all agent actions

## Threat Response Protocol

1. **Detect** — Identify threat type and severity
2. **Contain** — Block action, quarantine data
3. **Log** — Record full audit trail
4. **Report** — Surface to user with details
5. **Adapt** — Update rules to prevent recurrence
6. **Verify** — Test that threat is blocked
