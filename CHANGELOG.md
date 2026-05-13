# Changelog

All notable changes to Aurelius are documented in this file.


### Added
- **Unified sampling processors**: centralized temperature, top-p, and repetition handling in `src/inference/logit_processors.py` and `src/model/logits_processor.py`. Eliminates duplicate logic across inference modules.
- **Dynamic MoE router** (`src/model/dynamic_moe.py`): learnable per-expert temperature + EMA load-balancing bias for improved expert utilisation during training.
- **Memory-mapped dataset** (`src/data/memory_mapped_dataset.py`): streaming token shards without full RAM load; integrates with DataLoader.
- **Production FastAPI inference server** (`gateway/aurelius_api.py`): OpenAI-compatible `/v1/chat/completions` endpoint, WebSocket `/ws`, Prometheus `/metrics`, and automatic engine loading via `gateway.engine_loader`. Suitable for cloud/container deployment.
- **Docker and release automation**: multi-stage `Dockerfile`, `docker-compose.yml` (GPU-enabled), and GitHub Actions workflow for PyPI trusted publishing.

### Changed
- Tokeniser vocabulary size aligned to 8 192 (was 128 000) to match training shards.
- Checkpoint loading prefers `model.safetensors` directories with legacy `.pt` fallback and deprecation warning.
- All CLI scripts (`scripts/*.py`) now use `argparse` with no hardcoded absolute paths.

### Fixed
- CPU‑device mismatch: `last_moe_aux_loss` buffer now registered correctly via `register_buffer` and device‑safe assignment.
- Missing imports/aliases in mosaic harness (ModelAdapter, JudgeLLM, etc.).
- Circular‑import risk in `tests/eval/test_semantic_entropy.py` — resolved.
- Multiple Ruff/lint issues (import ordering, line length) across touched modules.

## [Unreleased]
## [2026-05-13] — Hardening & Production Readiness

### Added
- **Gateway security hardening**: security headers (CSP, HSTS, X-Frame, X-Content-Type), host allow-listing (`AURELIUS_ALLOWED_HOSTS`), per-IP rate limiting (`AURELIUS_RATE_LIMIT`, `AURELIUS_RATE_WINDOW`), request size limits (1 MiB normal, 10 MiB streaming).
- **Request tracing**: `X-Request-ID` header injected by middleware for end-to-end correlation.
- **Robust error handling**: clean HTTP error responses without stack traces; explicit 413/429/400 handlers.

### Changed
- **Dynamic MoE routing** fully integrated into `MoERouter` (removed separate `DynamicMoERouter` wrapper). Supports learnable per-expert temperature and EMA load‑bias.
- **Centralized logit processing**: `UnifiedLogitsProcessor` consolidates temperature, top‑p, repetition penalty; migrated `speculative_rejection.py` to use unified API.
- Checkpoint loading prefers `safetensors` with automatic `.pt` fallback + deprecation warning.
- Tokenizer vocab default corrected to 8 192 across configuration and tests.

### Fixed
- `SparseMoELayer` now consumes `RouterOutput` correctly; backward‑compatible tuple unpacking restored.
- `TopKRouter` accepts legacy `capacity_factor` (ignored) and `load_balance_alpha` for compatibility with ReMoDE.
- Removed hardcoded absolute paths from all CLI data-generation scripts; full `argparse` CLI now available.
- CPU‑device bug in `last_moe_aux_loss` buffer handling (now safely `.copy_()` with `.to(device)`).



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
- **Task Scheduler** (`agent/task_scheduler.py`): cron/interval/delayed job scheduling with thread-safe management.  
- **Pipeline Processor** (`aurelius_cli/pipeline_processor.py`): fluent ETL-style data transformation pipeline.  
- **Pipeline CLI** (`aurelius_cli/pipeline_commands.py`): `aurelius pipeline` command for streaming JSONL filter/map/sort/head/tail/dedup with expression language.  
- **SRE Metrics** (`src/monitoring/sre_metrics.py`): golden-signals metrics collector (latency, errors, traffic, saturation).  
- **CLI integration** (`aurelius_cli/scheduler_commands.py`): `aurelius schedule cron|interval|once -- <shell cmd>` commands.  
- **Metrics CLI** (`aurelius_cli/metrics_commands.py`): `aurelius metrics demo` synthetic workload reporter with `--requests`, `--error-rate`, `--latency-mean`, `--latency-std`, `--json` output options.  
- **Examples** (`examples/`): runnable scripts showcasing the new utilities (scheduler, pipeline, sre metrics).  
+ Examples: `sre_metrics_demo.py` added.  
- Tests added: 59 task_scheduler, 9 pipeline_processor, 10 sre_metrics, 5 scheduler_commands, 6 metrics_commands.  
- README updated with Developer Utilities section and CLI examples.

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
## Cycle 140-sec — 2026-05-03

- Fixed malformed f‑strings and missing quotes in `distributed.py`.
- Re‑indented `_require_auth` in `src/serving/aurelius_server.py`.
- Hardened checkpoint loading (`GradScaler` ctor) in `train_7b.py`.
- Fixed KV‑cache eviction shape mismatch.
- Updated all production `torch.load(..., weights_only=True)` calls, including `generate_jsonl.py`.
- Added pre‑commit hook to forbid `weights_only=False` outside tests.
- Added CI step to run pre‑commit checks.
- Added benchmark scripts for KV‑cache eviction and GradScaler performance.
