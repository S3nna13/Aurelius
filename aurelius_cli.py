"""Thin CLI entry point for Aurelius agent."""
from __future__ import annotations

from aurelius_cli.main import main as _main


def main() -> None:
    """Entry point for aurelius-cli script."""
    _main()


if __name__ == "__main__":
    main()
