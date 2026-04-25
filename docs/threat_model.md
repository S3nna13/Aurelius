# Aurelius — Threat Model

## Scope

This document provides a STRIDE analysis for each product surface of the Aurelius platform, identifying assets, adversaries, threat vectors, and mitigations per the Aurelius Loop v8 threat_model specification.

## Assets Protected

| Asset | Threat Category |
|-------|----------------|
| Training data integrity | Tampering, Repudiation |
| Model weights integrity | Tampering, Elevation of Privilege |
| Tokenizer contract integrity | Tampering |
| Host system (training/inference) | Elevation of Privilege, DoS |
| User secrets (HF token, API keys) | Information Disclosure, Spoofing |
| User prompts and outputs | Information Disclosure |
| MCP server trust boundary | Spoofing, Elevation of Privilege |
| Computer-use sandbox | Elevation of Privilege |
| Agent loop | DoS, Tampering |
| Serving endpoint | DoS, Spoofing |
| Supply chain (PyPI, HF Hub) | Tampering, Elevation of Privilege |

## Adversaries

| Adversary | Capability |
|-----------|-----------|
| Malicious PyPI package author | Typosquat, dependency confusion |
| Malicious model/dataset author on HF Hub | Data poisoning, weight tampering |
| Prompt-injection attacker | Jailbreak, tool-call hijack, data exfil |
| Adversarial input crafter | NaN/Inf weights, oversized inputs |
| Insider with commit access | Backdoored weights, supply chain |
| Network attacker on serving endpoint | DoS, MITM |
| Sandbox escaper via computer-use / MCP | RCE, privilege escalation |

## Per-Surface STRIDE Analysis

### Surface 0: model-family-infrastructure
- **S**: Manifest spoofing → mitigated by SHA-256 checkpoint/tokenizer verification
- **T**: Weight tampering → mitigated by detached signatures, `weights_only=True`
- **R**: Undocumented config changes → mitigated by compatibility versioning
- **I**: Model card disclosure of internals → acceptable, no secrets in model cards
- **D**: Config validation DoS → mitigated by assertion-based validation
- **E**: Config privilege escalation → mitigated by default-OFF feature flags

### Surface 1: base-lm
- **T**: Arithmetic exceptions (NaN/Inf) → mitigated by gradient monitors, loss spike recovery
- **D**: Oversized input OOM → mitigated by max_seq_len, token budget enforcement
- **E**: Weight exfiltration via canary tokens → mitigated by `canary_pipeline.py`

### Surface 6: safety-alignment
- **S**: Jailbreak via prompt injection → mitigated by `jailbreak_detector.py`, `prompt_injection_scanner.py`, `rule_engine.py`
- **T**: Adversarial text augmentation → mitigated by `adversarial_text_augment.py`
- **I**: Canary token leakage → mitigated by `canary_auditor.py`
- **D**: Red-team regression → mitigated by `red_team_dataset_gen.py`, synthetic probes
- **E**: Guardrail bypass → mitigated by `guardrails.py` fail-closed design

### Surface 8: serving
- **S**: API key spoofing → mitigated by HMAC `compare_digest` in `auth_middleware.py`
- **T**: Response tampering → mitigated by response streaming integrity
- **D**: DoS via unbounded requests → mitigated by rate_limiter.py (100 rps, burst 200), request size caps
- **I**: Exception info disclosure → mitigated by generic error responses
- **E**: SSRF → mitigated by URL scheme validation in `http_backend.py`

### Surface 9: terminal-ide-ui
- **S**: Terminal escape injection → mitigated by input sanitization
- **I**: Secret leakage in logs → mitigated by canary-token guard, PII detector

### Surface 12: computer-use-browser-use
- **E**: Sandbox escape via MCP/tool call → mitigated by `tool_sandbox_denylist.py`, `sandbox_executor.py`
- **S**: Tool-call origin spoofing → mitigated by attestation design
- **D**: Infinite tool-call loops → mitigated by `budget_bounded_loop.py`

### Surface 13: tooling-mcp-ecosystem
- **S**: Malicious MCP server manifest → mitigated by schema validation
- **T**: MCP manifest tampering → mitigated by `mcp_server.py` integrity checks
- **I**: Data exfiltration via MCP → mitigated by canary tokens
- **E**: Privilege escalation via MCP tools → mitigated by scope manifests

### Surface 15: security-hardening
- Full coverage in `.aurelius-cves.log` (AUR-SEC-2026-0001 through 0027)
- Deserialization: `weights_only=True`, `SafeLoader`, safe-extract wrappers
- Path traversal: `_SAFE_ID` regex, resolved-path jail
- DoS: Rate limiting, size caps, ReDoS-bounded regexes
- Supply chain: Hash-pinned lockfile, `pip-audit`, `syft` SBOM

## Findings Summary

| ID | Severity | CWE | Status |
|----|----------|-----|--------|
| AUR-SEC-2026-0001 | Medium | CWE-78 | Fixed |
| AUR-SEC-2026-0002 | Low | CWE-327 | Fixed |
| AUR-SEC-2026-0003 | Low | CWE-327 | Fixed |
| AUR-SEC-2026-0004 | High | CWE-502 | Fixed |
| AUR-SEC-2026-0005 | High | CWE-502 | Fixed |
| AUR-SEC-2026-0006 | High | CWE-502 | Fixed |
| AUR-SEC-2026-0007 | High | CWE-22 | Fixed |
| AUR-SEC-2026-0008 | Medium | CWE-400 | Fixed |
| AUR-SEC-2026-0009 | Medium | CWE-400 | Fixed |
| AUR-SEC-2026-0010 | Medium | CWE-209 | Fixed |
| AUR-SEC-2026-0011 | Medium | CWE-942 | Fixed |
| AUR-SEC-2026-0012 | High | CWE-400 | Fixed |
| AUR-SEC-2026-0013 | High | CWE-670 | Fixed |
| AUR-SEC-2026-0014 | Low | CWE-327 | Fixed |
| AUR-SEC-2026-0015 | Medium | CWE-209 | Fixed |
| AUR-SEC-2026-0020 | Medium | CWE-22 | Fixed |
| AUR-SEC-2026-0021 | Medium | CWE-78 | Fixed |
| AUR-SEC-2026-0022 | Medium | CWE-78 | Fixed |
| AUR-SEC-2026-0023 | Medium | CWE-605 | Fixed |
| AUR-SEC-2026-0024 | High | CWE-502 | Fixed |
| AUR-SEC-2026-0025 | High | CWE-611 | Fixed |
| AUR-SEC-2026-0026 | Low | CWE-1104 | Waived |
| AUR-SEC-2026-0027 | Medium | CWE-693 | Fixed |

**Open Critical: 0 | Open High: 0 | Open Medium: 0**