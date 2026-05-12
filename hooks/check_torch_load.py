#!/usr/bin/env python3
"""Pre‑commit hook that forbids ``torch.load(..., weights_only=False)``
outside the ``tests/`` directory.

The hook scans the files passed by pre‑commit (via ``sys.argv[1:]``) and
exits with status 1 if it finds a disallowed pattern.  Test files are allowed
to keep the existing unit‑test ``weights_only=False`` usage.
"""

import pathlib
import re
import sys

PATTERN = re.compile(r"torch\.load\(.*weights_only\s*=\s*False", re.IGNORECASE)

for path_str in sys.argv[1:]:
    path = pathlib.Path(path_str)
    # Skip non‑Python files
    if path.suffix != ".py":
        continue
    # Allow any occurrence inside a tests/ directory
    if "tests" in path.parts:
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: S112
        continue
    if PATTERN.search(text):
        print(f"Forbidden torch.load(..., weights_only=False) in {path}")
        sys.exit(1)

sys.exit(0)
