# Contributing to Aurelius

Thank you for your interest in contributing to Aurelius! This document provides guidelines and workflows for contributors.

## Development Setup

```bash
git clone https://github.com/S3nna13/Aurelius
cd Aurelius
pip install -e ".[dev]"
```

## Project Structure

Aurelius uses a **surface-based architecture**. Each domain (model, training, inference, security, etc.) lives in its own directory under `src/`. When adding features:

- Place new modules in the appropriate `src/<surface>/` directory
- Add corresponding tests in `tests/<surface>/`
- Follow the existing naming convention: `snake_case.py` for modules, `test_<module>.py` for tests

## Adding a New Surface

1. Create `src/<surface>/` with an `__init__.py`
2. Add `tests/<surface>/` with an `__init__.py`
3. Add tests following the existing pattern (shape/dtype, gradient flow, determinism, edge cases)
4. Update `README.md` repo structure table

## Coding Standards

- **Pure PyTorch** — no HuggingFace Transformers, einops, flash-attn, or framework wrappers at runtime
- **Stdlib first** — prefer standard library solutions over external packages
- **Type hints** — use Python type hints for public APIs
- **Docstrings** — document classes and public methods

## Running Tests

```bash
# Full suite (~30 min on CPU)
pytest -q

# Single surface
pytest -q tests/model/

# Single module
pytest -q tests/security/test_gcg_attack.py
```

## Submitting Changes

1. Create a branch: `git checkout -b feat/<description>` or `git checkout -b fix/<description>`
2. Make your changes with tests
3. Ensure all tests pass: `pytest -q`
4. Update documentation (README, docstrings) if needed
5. Open a pull request using the provided template

## Release Cycle

Aurelius follows an implementation cycle model. Each cycle adds new modules without modifying existing files (additive development). Cycles are documented in the README "Current status" section.

## Questions?

Open a [discussion](https://github.com/S3nna13/Aurelius/discussions) or [issue](https://github.com/S3nna13/Aurelius/issues).
