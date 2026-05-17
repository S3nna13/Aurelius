# Aurelius current structural/clutter audit

Generated: 2026-05-17T06:41:48Z

Scope: tracked repository files on `main`, after excluding local-only `.git`, `.venv`, cache, node_modules, and ignored runtime artifacts. This audit is aligned with the AMC-first core boundary in `docs/CORE_SURFACE.md`.

## Summary

- Tracked files after this cleanup: 5471
- Top-level tracked entries: 57
- Root files after this cleanup: 23
- Tracked generated/cache artifacts after cleanup: 0
- Root temporary-looking files after cleanup: 0
- Root backup-snapshot files after cleanup: 0

## Top-level file distribution

- `tests`: 2384
- `src`: 2147
- `frontend`: 176
- `agent`: 100
- `gateway`: 80
- `crates`: 70
- `docs`: 62
- `middle`: 58
- `archive`: 55
- `aurelius_cli`: 33
- `scripts`: 33
- `server`: 32
- `tools`: 32
- `aurelius`: 28
- `acp_adapter`: 25
- `plugins`: 21
- `configs`: 19
- `cron`: 18
- `deployment`: 18
- `training_data`: 15
- `.github`: 11
- `examples`: 8
- `alembic`: 5
- `rust_memory`: 4
- `.hermes`: 2
- `benchmarks`: 2
- `data`: 2
- `k8s`: 2
- `.cargo`: 1
- `.dockerignore`: 1

## Root files that remain

These are active configuration, package, and project-facing entrypoints. No temporary root scripts or pyproject backup snapshots remain tracked.

- `.dockerignore`
- `.env.example`
- `.gitattributes`
- `.gitignore`
- `.pre-commit-config.yaml`
- `AUTONOMOUS_IMPROVEMENT_SUMMARY.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `Cargo.lock`
- `Cargo.toml`
- `Dockerfile`
- `LICENSE`
- `Makefile`
- `README.md`
- `SECURITY.md`
- `alembic.ini`
- `bandit-baseline.json`
- `docker-compose.yml`
- `package-lock.json`
- `package.json`
- `pyproject.toml`
- `rust-toolchain.toml`
- `uv.lock`

## Cleanup performed in this pass

Removed tracked files that were generated output, unreferenced temporary root clutter, or obsolete root backup snapshots:

- `frontend/dist/favicon.svg`
- `frontend/dist/icons.svg`
- `frontend/dist/index.html`
- `tmp_job_test.sh`
- `pyproject.toml.consolidated`
- `pyproject.toml.pre_consolidation`

Rationale:

- `frontend/dist/` is produced by `npm run build -w frontend` and already ignored by `.gitignore`.
- Docker deploy builds copy the freshly generated Vite output from the builder stage, not from committed `frontend/dist` files.
- `tmp_job_test.sh` only wrote a tick into `/tmp/aurelius_job_log.txt`, had no tracked references, and was not part of the AMC-first runtime or release gates.
- The pyproject backup snapshots had no tracked references and are superseded by the active root `pyproject.toml` plus git history.
- `.gitignore` now explicitly blocks the removed root temp/backup filenames from being re-added accidentally.

## Remaining structural observations

1. `tests/` is the largest tracked tree by file count. That is expected after stabilizing release gates; do not prune tests without a separate coverage-aware pass.
2. `archive/` is intentionally retained as historical context. It is not imported by active CI gates and should remain read-only unless a future archive policy is adopted.
3. `bandit-baseline.json` is now a large root-level file. It is intentionally root-level because CI reads it directly and treats it as the explicit low-severity Bandit baseline.
4. `docs/executive/Aurelius_Executive_Overview.pdf` is the largest tracked artifact. It is product/documentation collateral, not build output; removal would require a docs/product decision.
5. `training_data/` is still tracked despite `.gitignore` containing `training_data/`; these are existing tracked generator sources and should not be removed in a cleanup-only pass because they may be part of dataset generation workflows.

## Largest tracked files

- `docs/executive/Aurelius_Executive_Overview.pdf`: 2,008,989 bytes
- `bandit-baseline.json`: 1,274,519 bytes
- `uv.lock`: 692,267 bytes
- `package-lock.json`: 314,206 bytes
- `training_data/sft_generator.py`: 152,783 bytes
- `middle/package-lock.json`: 139,393 bytes
- `server/package-lock.json`: 127,861 bytes
- `training_data/pretrain_generator.py`: 109,182 bytes
- `Cargo.lock`: 78,068 bytes
- `src/ui/aurelius_shell.py`: 77,900 bytes
- `scripts/generate_dissertation.py`: 74,122 bytes
- `docs/plans/2026-04-20-harvest-implementation.md`: 63,061 bytes
- `src/agent/session_manager.py`: 63,036 bytes
- `agent/session_manager.py`: 63,036 bytes
- `src/model/interface_framework.py`: 60,946 bytes
- `docs/plans/2026-05-09-mosaic-v2-implementation.md`: 59,442 bytes
- `docs/executive/overview.html`: 58,796 bytes
- `data/pretrain/glm5/train_shard_000.npy`: 58,372 bytes
- `training_data/_build_math_gen.py`: 52,950 bytes
- `docs/BRAIN_ARCHITECTURE.md`: 50,941 bytes

## Current recommendation

The repository is no longer blocked by obvious tracked build-output, temp-script, or root pyproject-backup clutter. The next structural cleanup should be planned, not opportunistic:

- Decide whether `archive/` should stay in-repo, move to a branch/tag, or remain as a read-only historical tree.
- Decide whether generated documentation artifacts such as PDFs belong in git or should be release artifacts.
- Decide whether tracked training-data generator sources should stay at root-level `training_data/` or move under `src/data/`/`scripts/` in a dedicated migration.
- Keep future cleanups constrained to one category per PR/commit so CI regressions are easy to attribute.
