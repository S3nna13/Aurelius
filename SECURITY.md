# Security Policy

## Supported Versions

Aurelius is under active development. Security fixes are applied to the `main` branch.

## Reporting a Vulnerability

Please report security vulnerabilities to **cantonio@bluegracegroup.com**.

Do not open public GitHub issues for security vulnerabilities. We will acknowledge reports within 48 hours and aim to release a fix within 14 days for confirmed high/critical findings.

## CVE Ledger

All security findings are tracked in `.aurelius-cves.log` at the repository root.

| ID | Title | Severity | CWE | Status |
|----|-------|----------|-----|--------|
| AUR-SEC-2026-0001 | Shell injection in cli/main.py + shell_tool.py | Medium | CWE-78 | Fixed (cycle-139) |
| AUR-SEC-2026-0002 | Weak MD5 hash in synthetic_code.py (non-security use) | Low | CWE-327 | Fixed (cycle-139) |
| AUR-SEC-2026-0003 | Weak SHA1 hash in corpus_indexer.py (non-security use) | Low | CWE-327 | Fixed (cycle-139) |
| AUR-SEC-2026-0004 | weights_only=False in checkpoint.py optimizer load | High | CWE-502 | Fixed (cycle-139) |
| AUR-SEC-2026-0005 | weights_only=False in warm_start.py interpolate branch | High | CWE-502 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0006 | weights_only=False in warm_start.py stack_layers branch | High | CWE-502 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0007 | Path traversal in FileConversationStore | High | CWE-22 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0008 | ReDoS in JS import regex | Medium | CWE-400 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0009 | ReDoS in Go import regex | Medium | CWE-400 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0010 | Exception info disclosure in SSEMCPServer | Medium | CWE-209 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0011 | Wildcard CORS on SSE MCP server | Medium | CWE-942 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0012 | Unbounded Content-Length DoS on SSE server | High | CWE-400 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0013 | Broken harm detection logic in guardrails | High | CWE-670 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0014 | MD5 in session_router._hash | Low | CWE-327 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0015 | Exception info disclosure in StdioMCPServer | Medium | CWE-209 | Fixed (cycle-139-sec) |
| AUR-SEC-2026-0016 | Canary pipeline (new defense) | N/A | N/A | Deployed (cycle-139-sec) |
| AUR-SEC-2026-0017 | Safe archive extractor (new defense) | N/A | N/A | Deployed (cycle-139-sec) |
| AUR-SEC-2026-0018 | Auth middleware / HMAC timing-safe (new defense) | N/A | N/A | Deployed (cycle-139-sec) |
| AUR-SEC-2026-0019 | Token-bucket rate limiter (new defense) | N/A | N/A | Deployed (cycle-139-sec) |
| AUR-SEC-2026-0020 | urllib SSRF / unchecked scheme in http_backend + ollama_adapter | High | CWE-918 | Fixed (cycle-150-sec) |
| AUR-SEC-2026-0021 | Unsafe subprocess in code-exec surfaces (sandbox, test_runner, red_team, exec_tool) | High | CWE-78, CWE-426 | Fixed (cycle-150-sec) |
| AUR-SEC-2026-0022 | CycloneDX 1.5 SBOM generator (new defense) | N/A | N/A | Deployed (cycle-150-sec) |
| AUR-SEC-2026-0023 | Typosquat guard for uv.lock (new defense) | N/A | CWE-494 | Deployed (cycle-150-sec) |
| AUR-SEC-2026-0024 | Bind-address gate — require opt-in for 0.0.0.0 (new defense) | N/A | CWE-1327 | Deployed (cycle-150-sec) |

## Security Architecture

Aurelius applies defense-in-depth:

- **Deserialization**: All `torch.load()` calls use `weights_only=True` (CWE-502 mitigation)
- **Path traversal**: `FileConversationStore` enforces `_SAFE_ID` allowlist + resolved-path jail (CWE-22 mitigation)
- **DoS**: SSE server enforces 1 MiB Content-Length cap; token-bucket rate limiter (100 rps, burst 200) via `src/serving/rate_limiter.py`
- **ReDoS**: All import regexes use bounded quantifiers (`[^\n]{0,200}?`, `[^)]{0,4096}`) with no DOTALL
- **Info disclosure**: Exception details are logged internally only; all API surfaces return generic error strings
- **CORS**: Origin allowlist (no wildcard); `SSEMCPServerConfig.cors_origins` defaults to `[]`
- **Auth**: HMAC `compare_digest` timing-safe API key comparison via `src/serving/auth_middleware.py`
- **Canary tokens**: One-shot exfiltration detection via `src/security/canary_pipeline.py`
- **Archive safety**: zip-slip/tar-slip/bomb/symlink guards via `src/security/safe_archive.py`
- **Harm detection**: `_harm_score` returns 1.0 on any single suspicious substring match (fail-closed)
