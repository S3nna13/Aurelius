# Contributing to Aurelius

Thank you for your interest in contributing to Aurelius!

## Development Setup

```bash
git clone https://github.com/S3nna13/Aurelius.git
cd Aurelius
pip install -e ".[dev]"
```

## Code Style

We use **Ruff** for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Testing

Run the full test suite:

```bash
pytest -q
```

Run a specific module:

```bash
pytest -q tests/agent/test_react_loop.py
```

## Branch Strategy

- **main** — stable releases only.
- **cycle/<n>-<name>** — implementation cycles (additive development).
- **sec/<id>-<name>** — security hardening branches.
- **feat/<name>** — feature branches.
- **deploy/<name>** — deployment-related changes.

Open a Pull Request to `main` when your cycle is complete and all tests pass.

## Security

Please do **not** open public issues for security vulnerabilities. Email
**S3nna13** instead. See [SECURITY.md](SECURITY.md) for
our CVE ledger and threat model.

## Additive Development

Aurelius follows an additive development model: each cycle adds new files;
existing files are not modified. This preserves history and makes reviews
surgical.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
