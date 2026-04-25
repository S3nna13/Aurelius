# Changelog

All notable changes to Aurelius are documented in this file.

## [Unreleased]

### Security
- **CRITICAL** Fix sandbox escape via `object.__subclasses__()` in three
  in-process execution modules (`sandbox_executor.py`, `code_execution.py`,
  `code_eval.py`).
- Add SSRF IP blocklist to `http_backend.py` (blocks private/reserved IPs).
- Change `DEFAULT_AUTH_MIDDLEWARE` to `require_auth=True` (fail-closed).
- Harden `shell_tool.py`: replace `shell=True` + denylist with `shell=False`
  + `shlex.split()` + explicit allow-list.

### Fixed
- CLI tokenizer bug: `_load_generate_fn` now attempts to load a real
  `AureliusTokenizer` before falling back to byte-tokenizer.
- Add `max_seq_len` guard to `generate()` and `generate_stream()` in
  `transformer.py` to prevent RoPE buffer OOB crashes.
- Fix `get_workstream(..., missing_ok=False)` to actually raise when missing.
- Add file-locking to `session_manager.py` JSON persistence.
- Save/load Muon optimizer state in training checkpoints.
- Map `use_zclip`, `zclip_z_threshold`, `zclip_ema_alpha` in
  `TrainConfig.from_yaml()`.

### CI/CD
- Rewrite GitHub Actions workflow: Python 3.12/3.13 matrix, full pytest run,
  ruff lint+format, Bandit security scan, pip-audit.
- Add `cycle/*`, `sec/*`, `feat/*`, `deploy/*` to CI trigger branches.

### Added
- `CONTRIBUTING.md`, `CHANGELOG.md`, `.dockerignore`.

## Cycle 200 — 2026-04-25

- `abductive_reasoner.py` + `analogy_engine.py` (reasoning)
- `memory_retrieval_reranker.py` + `memory_fusion.py` (memory)
- Integration test fixes for `FeatureFlagRegistry` API

## Cycle 199 — 2026-04-24

- `tgi_backend_adapter.py` + `context_quality_scorer.py`
- `video_frame_sampler.py` + `query_intent_classifier.py`
- `accessibility_announcer.py`
- 157 new tests

## Cycle 139-sec — 2026-04-18

- Security gate: closed AUR-SEC-2026-0001 through 0027
- `weights_only=True` on all `torch.load()` calls
- Path-traversal hardening in `FileConversationStore`
- ReDoS-bounded regexes
- Canary pipeline, safe archive extractor, HMAC auth middleware
