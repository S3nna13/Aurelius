"""Aurelius CLI — entry point. Makes the CLI runnable as `aurelius-cli` or `dracarys`."""

from __future__ import annotations

import sys

from .terminal_cli import main


def dracarys_main():
    """Entry point for 'dracarys' command."""
    sys.argv[0] = "dracarys"
    main()


def aurelius_main():
    """Entry point for 'aurelius-cli' command."""
    sys.argv[0] = "aurelius-cli"
    main()


if __name__ == "__main__":
    main()
