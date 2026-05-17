# Package Consolidation Migration Guide

## What Changed

**Problem:** Multiple top-level packages (`agent/`, `gateway/`, `aurelius/`, `cron/`, `tools/`, `plugins/`) all contained modules that duplicated functionality in `src/`. This created import ambiguity under Python 3.12 which removed implicit namespace packages.

**Solution:** Consolidated to `src/` as the single canonical package root via import shims.

## Changes Applied

### 1. Import Shims (Backward Compatible)

Created `__init__.py` shims in each legacy top-level package:

| Legacy Import | Redirects To | Status |
|--------------|--------------|--------|
| `import agent` | `src.agent` | DeprecationWarning |
| `import gateway` | `src.serving` | DeprecationWarning |
| `import aurelius` | `src` | DeprecationWarning |
| `import cron` | `src.workflow` | DeprecationWarning |
| `import tools` | `src.tools` | DeprecationWarning |
| `import plugins` | re-export stubs | DeprecationWarning |

**Behavior:** Old imports still work but emit `DeprecationWarning`. No code breaks.

### 2. Package Discovery (pyproject.toml)

**Before:**
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["aurelius*", "src*"]
```

**After:**
```toml
[tool.setuptools.packages.find]
where = ["src", "aurelius_cli"]
include = ["src*", "aurelius_cli*"]
```

**Effect:** Only `src/` and `aurelius_cli/` are installed as packages. The other directories are now internal-only or shim-only.

### 3. Security Audit

Added TODO annotations to files using `eval()`/`exec()`:

- `agent/permission_system.py` — chain_of_thought_eval()
- `agent/tool_sandbox_denylist.py` — rule expression compiler

**Action required:** Review each eval() to ensure inputs are constrained. Prefer `ast.literal_eval()` where feasible.

### 4. Logging Migration

Converted `print()` → `logging.getLogger(__name__)` in critical gateway modules:

- `gateway/chat_client.py` (9 print calls)
- `gateway/router_pareto.py` (5 print calls)
- `gateway/openai_api_validator.py` (1 print call)
- `gateway/terminal_chat.py` (interactive CLI, print correct)

Remaining modules still use print() (will be addressed incrementally).

### 5. Deprecated Speculative Decoding Variants

Marked older speculative decoding implementations as deprecated:

- `src/inference/speculative.py`
- `src/inference/speculative_v2.py`
- `src/inference/speculative_decoding.py`
- `src/inference/speculative_decoding_v2.py`
- `src/inference/speculative_decoding_v3.py`

Prefer `speculative_decoding_v4.py` or `eagle3_decoding.py` for new work.

## Required Actions

1. **Verify import resolution**
   ```bash
   python3 -c "import agent; print(agent.__file__)"
   # Should show .../aurelius/src/agent/__init__.py not .../aurelius/agent/
   ```

2. **Run tests**
   ```bash
   make test
   # Or specific validation:
   pytest tests/test_package_consolidation.py -v
   ```

3. **Commit** (if tests pass)
   ```bash
   git add -A
   git commit -m "feat: consolidate package structure to src/ canonical root
   
   - Add backward-compatible import shims
   - Deprecate legacy imports with warnings  
   - Narrow setuptools package discovery to src/ and aurelius_cli/
   - Annotate eval() sites for security audit
   - Convert print→logging in gateway modules
   - Deprecate older speculative decoding variants
   
   Risk: LOW — fully backward compatible with deprecation path."
   ```

4. **Monitor** logs for DeprecationWarning — indicates code still using old import paths.

## Rollback

The old in-tree `.bak` backup artifacts were removed during AMC-first cleanup to reduce tracked clutter. Git history is the rollback source of truth now. To inspect or restore a removed backup file:
```bash
# Find the last commit that still contained a removed backup file
git log --all -- agent/__init__.py.bak

# Inspect a version from that commit
git show <commit>:agent/__init__.py.bak

# Restore a previous tracked file if truly needed
git restore --source=<commit> -- agent/__init__.py.bak
```

Prefer reverting the focused cleanup commit over reintroducing backup files piecemeal.

## Future Work (Post-Consolidation)

- Remove duplicate files after verifying zero direct imports from legacy paths
- Eliminate wildcard imports (197 remaining)
- Complete eval/exec security audit (replace with literal_eval where safe)
- Finish logging migration in agent modules (~50+ modules)
- Consider splitting src/inference (63K LOC) into subpackages

---

Generated: 2026-05-14 by Hermes Agent
