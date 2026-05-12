# Contributing to Aurelius

Welcome! Here's everything you need to know to work on Aurelius effectively.

---

## Quick Start

```bash
# Clone and bootstrap
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
bash scripts/bootstrap.sh --fast   # Python deps only (skip Rust build)

# Or use Make
make setup-dev
```

---

## Development Environment

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ (3.14 works with caveats*) | Backend, CLI, training |
| Node.js | 20+ | BFF middleware, frontend |
| Rust | 2024+ | Data engine (optional for most dev) |
| pre-commit | latest | Lint/format hooks |

*Python 3.14: Some tests using `torch.multiprocessing` (code execution, plugin sandbox) are skipped on 3.14 due to a torch incompatibility with the `spawn` start method. All other tests run normally.

---

## Running Tests

### Python Backend Tests

```bash
# Full suite (excludes security/ which has network-hanging integration tests)
.venv/bin/python -m pytest tests/ -q --ignore=tests/security/ --tb=no

# Run a specific directory
.venv/bin/python -m pytest tests/backends/ -q --tb=no
.venv/bin/python -m pytest tests/serving/ -q --tb=no
.venv/bin/python -m pytest tests/data/ -q --tb=no
.venv/bin/python -m pytest tests/inference/ -q --tb=no

# Run a specific test file
.venv/bin/python -m pytest tests/backends/test_http_backend.py -v --tb=short

# Show output on failures (don't use -q)
.venv/bin/python -m pytest tests/backends/ -v --tb=short
```

### Expected test results on Python < 3.14
- `tests/backends/` — all pass
- `tests/serving/` — all pass
- `tests/data/` — all pass
- `tests/inference/` — all pass (2 skipped on Python 3.14)
- `tests/agent/test_plugin_sandbox.py` — all skipped on Python 3.14
- `tests/security/` — individual files pass; running as a suite hangs

### Running Linters

```bash
# Ruff (Python linter + formatter)
.venv/bin/python -m ruff check src/      # must show "All checks passed!"
.venv/bin/python -m ruff check src/ --fix   # auto-fix what can be auto-fixed

# Bandit (security scanner)
.venv/bin/python -m bandit -r src/ -f txt
# High/Medium severity = blocking. Low severity = acceptable scanner noise.

# Type checking (if pyright is installed)
.venv/bin/python -m pyright src/
```

---

## Project Structure

```
aurelius/           # Top-level Python packages (agent, alignment, etc.)
src/                # Main source (backends, serving, inference, model)
agent/              # Agent system (registry, tools, plugins, skills)
gateway/            # API server (OpenAI-compatible endpoints)
aurelius_cli/       # CLI tools (chat, serve, shell)
configs/            # YAML configs (model, training, serving)
crates/             # Rust data engine
frontend/           # React/TypeScript UI
middle/             # Node.js BFF
scripts/            # Dev scripts (bootstrap, lint-all, etc.)
tests/              # Test suite (mirrors src/ layout)
```

### Key entry points

| Command | Module | Description |
|---------|--------|-------------|
| `aurelius` | `src/cli/main.py` | Interactive chat CLI |
| `aurelius serve` | `gateway/aurelius_server.py` | API server + web UI |
| `aurelius-shell` | `src/cli/aurelius_shell.py` | REPL shell |

---

## Code Style

- **Python**: Ruff with line-length 120. Import order: stdlib → third-party → local.
- **TypeScript**: Prettier with default settings.
- **Rust**: `cargo fmt` + `cargo clippy`.

Use pre-commit to run all checks before pushing:

```bash
pre-commit run --all-files
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AURELIUS_ALLOW_PRIVATE_URLS` | `0` | Set to `1` to allow localhost/private URLs in backend tests |
| `AURELIUS_API_KEY` | — | Single API key for server auth |
| `AURELIUS_API_KEYS` | — | Multi-key store: `id:key:scope1,scope2;...` |
| `AURELIUS_MEMORY_DIR` | platform default | Path for memory store SQLite DB |
| `AURELIUS_DATA_DIR` | `/var/lib/aurelius` | Path for usage pipeline data (tests use temp dir) |
| `AURELIUS_VERSION` | `0.1.0` | Server version string |

---

## Common Tasks

### Add a new test

1. Tests live next to the code they test: `tests/backends/test_http_backend.py` for `src/backends/http_backend.py`
2. Use `pytest` fixtures from `tests/conftest.py` (autouse fixtures snapshot/restore state)
3. Mark Python 3.14-incompatible tests with:
   ```python
   import sys
   pytestmark = pytest.mark.skipif(
       sys.version_info >= (3, 14),
       reason="torch multiprocessing spawn incompatible with Python 3.14",
   )
   ```

### Fix a failing test

1. Run with `--tb=short -v` to see the actual error
2. Check the test's imports and environment variable setup (look for `monkeypatch.setenv`)
3. Ensure the code doesn't make real network calls to `localhost` without setting `AURELIUS_ALLOW_PRIVATE_URLS=1`

### Add a new tool to the agent

1. Define the tool function in `src/agent/tools/` (or appropriate subsystem)
2. Register it in the tool registry (see `agent/tool_registry.py`)
3. Add tests in `tests/agent/`

---

## Branch Strategy

- `main` — stable, always deployable
- `feat/*` — feature branches
- Fixes go through PR → review → merge to `main`

---

## Getting Help

- Open an issue on GitHub for bugs or feature requests
- Check `docs/` for architecture diagrams and design docs
- Read `README.md` for project overview and security audit notes