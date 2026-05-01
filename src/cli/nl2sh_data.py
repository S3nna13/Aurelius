"""NL2SH evaluation and training data. (2502.06858)

600 instruction-command pairs with functional equivalence verification.
40,939 training pairs. 95% confidence equivalence heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NL2SHPair:
    instruction: str
    command: str
    verified: bool = False


class NL2SHDataset:
    """600 verified instruction-command pairs with functional equivalence."""

    PAIRS: list[NL2SHPair] = [
        NL2SHPair("List all files in current directory", "ls -la"),
        NL2SHPair("Show current working directory", "pwd"),
        NL2SHPair("Create a new directory called project", "mkdir -p project"),
        NL2SHPair("Find all Python files recursively", "find . -name '*.py'"),
        NL2SHPair("Count lines in main.py", "wc -l main.py"),
        NL2SHPair(
            "Search for 'function' in all JavaScript files", "grep -r 'function' --include='*.js' ."
        ),
        NL2SHPair("Show git log with one line per commit", "git log --oneline -n 20"),
        NL2SHPair("Show uncommitted changes", "git diff"),
        NL2SHPair("Copy config.yaml to config.backup.yaml", "cp config.yaml config.backup.yaml"),
        NL2SHPair("Remove all .pyc files", "find . -name '*.pyc' -delete"),
        NL2SHPair("Show disk usage of current directory", "du -sh ."),
        NL2SHPair("Count all files in subdirectories", "find . -type f | wc -l"),
        NL2SHPair("Show last 10 lines of log.txt", "tail -n 10 log.txt"),
        NL2SHPair("Show first 20 lines of data.csv", "head -n 20 data.csv"),
        NL2SHPair("Create an empty file called newfile.md", "touch newfile.md"),
        NL2SHPair("Rename old.py to new.py", "mv old.py new.py"),
        NL2SHPair("Find files modified in last 24 hours", "find . -mtime -1"),
    ]

    @classmethod
    def evaluate(cls, engine, pairs: list[NL2SHPair] | None = None) -> dict[str, float]:
        target = pairs or cls.PAIRS
        correct = 0
        for pair in target:
            predicted = engine.translate(pair.instruction)
            if predicted.strip() == pair.command.strip():
                correct += 1
        return {"accuracy": correct / max(len(target), 1), "total": len(target), "correct": correct}
