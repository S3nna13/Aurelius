"""Command output compression and filtering for Aurelius.

Reduces token count of shell command outputs before they reach the agent
context.  Inspired by rtk.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

__all__ = [
    "CompressionConfig",
    "CompressionResult",
    "OutputCompressor",
    "DEFAULT_COMPRESSOR",
    "OUTPUT_COMPRESSOR_REGISTRY",
]


@dataclass
class CompressionConfig:
    """Parameters that control how aggressively output is compressed."""

    max_lines: int = 50
    max_line_length: int = 200
    deduplicate: bool = True
    group_by_prefix: bool = True
    remove_empty: bool = True
    show_summary: bool = True


@dataclass
class CompressionResult:
    """Outcome of a compression pass."""

    output: str
    original_lines: int
    compressed_lines: int
    compression_ratio: float
    strategies_applied: list[str]


@dataclass
class OutputCompressor:
    """Compresses shell command output to reduce context token usage."""

    config: CompressionConfig = field(default_factory=CompressionConfig)

    # ------------------------------------------------------------------ #
    # Core compress
    # ------------------------------------------------------------------ #

    def compress(self, text: str) -> CompressionResult:
        """Apply general-purpose compression strategies.

        Strategies are applied in order:
        1. Remove empty lines
        2. Deduplicate consecutive identical lines
        3. Group by common 3-char prefix
        4. Truncate long lines
        5. Truncate total lines with summary
        """
        original_lines = text.count("\n") + 1 if text else 0
        lines = text.splitlines()
        strategies: list[str] = []

        # 1. Remove empty lines
        if self.config.remove_empty:
            lines = [line for line in lines if line.strip() != ""]
            if len(lines) < original_lines:
                strategies.append("remove_empty")

        # 2. Deduplicate consecutive identical lines
        if self.config.deduplicate and lines:
            deduped: list[str] = []
            current = lines[0]
            count = 1
            for line in lines[1:]:
                if line == current:
                    count += 1
                else:
                    deduped.append(f"{current} (x{count})" if count > 1 else current)
                    current = line
                    count = 1
            deduped.append(f"{current} (x{count})" if count > 1 else current)
            if len(deduped) < len(lines):
                strategies.append("deduplicate")
            lines = deduped

        # 3. Group by common prefix
        if self.config.group_by_prefix and lines:
            grouped: list[str] = []
            i = 0
            changed = False
            while i < len(lines):
                line = lines[i]
                prefix = line[:3] if len(line) >= 3 else line
                j = i + 1
                while (
                    j < len(lines)
                    and len(lines[j]) >= 3
                    and lines[j][:3] == prefix
                    and prefix != ""
                ):
                    j += 1
                count = j - i
                grouped.append(line)
                if count > 2:
                    grouped.append(f"{prefix}... +{count - 1} more")
                    changed = True
                    i = j
                else:
                    i += 1
            if changed:
                strategies.append("group_by_prefix")
            lines = grouped

        # 4. Truncate long lines
        if self.config.max_line_length > 0:
            truncated: list[str] = []
            limit = self.config.max_line_length
            for line in lines:
                if len(line) > limit:
                    truncated.append(line[: limit - 3] + "...")
                else:
                    truncated.append(line)
            if truncated != lines:
                strategies.append("truncate_lines")
            lines = truncated

        # 5. Truncate total lines with summary
        omitted = 0
        if self.config.max_lines > 0 and len(lines) > self.config.max_lines:
            omitted = len(lines) - self.config.max_lines
            lines = lines[: self.config.max_lines]
            strategies.append("truncate_total")
            if self.config.show_summary:
                lines.append(f"... {omitted} lines omitted")

        compressed_lines = len(lines)
        ratio = compressed_lines / original_lines if original_lines > 0 else 0.0
        return CompressionResult(
            output="\n".join(lines),
            original_lines=original_lines,
            compressed_lines=compressed_lines,
            compression_ratio=ratio,
            strategies_applied=strategies,
        )

    # ------------------------------------------------------------------ #
    # Specialised compressors
    # ------------------------------------------------------------------ #

    def compress_git_status(self, text: str) -> CompressionResult:
        """Compress ``git status`` output.

        Removes "(use git ... to ...)" hints and keeps branch info on one
        line.
        """
        lines = text.splitlines()
        strategies: list[str] = []

        filtered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('(use "git') or stripped.startswith("(use 'git"):
                continue
            filtered.append(line)
        if len(filtered) < len(lines):
            strategies.append("remove_hints")

        # Compress branch info if present
        compressed_branch: list[str] = []
        for line in filtered:
            if line.startswith("On branch "):
                compressed_branch.append(line)
                strategies.append("compress_branch")
            elif line.startswith("Your branch is"):
                # Drop verbose branch state lines
                strategies.append("compress_branch")
                continue
            else:
                compressed_branch.append(line)

        result = self.compress("\n".join(compressed_branch))
        result.strategies_applied = list(
            dict.fromkeys(strategies + result.strategies_applied)
        )
        return result

    def compress_ls(self, text: str) -> CompressionResult:
        """Compress ``ls`` output by grouping file extensions."""
        lines = text.splitlines()
        original_lines = len(lines)

        # Extract file names, dropping common ls decorations
        names: list[str] = []
        for line in lines:
            stripped = line.strip()
            # Skip empty and obvious header/total lines
            if not stripped or stripped.lower().startswith("total "):
                continue
            names.append(stripped)

        ext_counts: Counter[str] = Counter()
        for name in names:
            # Handle directories (trailing slash)
            if name.endswith("/"):
                ext_counts["(dir)"] += 1
            else:
                parts = name.rsplit(".", 1)
                if len(parts) == 2 and parts[1]:
                    ext_counts[f".{parts[1]}"] += 1
                else:
                    ext_counts["(no ext)"] += 1

        out_lines = [f"{ext}: {count}" for ext, count in sorted(ext_counts.items())]
        compressed_lines = len(out_lines)
        ratio = compressed_lines / original_lines if original_lines > 0 else 0.0
        return CompressionResult(
            output="\n".join(out_lines),
            original_lines=original_lines,
            compressed_lines=compressed_lines,
            compression_ratio=ratio,
            strategies_applied=["group_by_extension"],
        )

    def compress_test_output(self, text: str) -> CompressionResult:
        """Compress pytest / test runner output.

        Shows failures only and compresses passes to a single summary
        line.
        """
        lines = text.splitlines()
        original_lines = len(lines)

        failures: list[str] = []
        passed_count = 0
        for line in lines:
            stripped = line.strip()
            # Common pytest pass indicators
            if " PASSED" in stripped or " passed" in stripped.lower():
                passed_count += 1
                continue
            # Skip progress lines and empty lines
            if not stripped or stripped.startswith("==="):
                continue
            failures.append(line)

        out_lines = list(failures)
        if passed_count:
            out_lines.append(f"{passed_count} passed")

        compressed_lines = len(out_lines)
        ratio = compressed_lines / original_lines if original_lines > 0 else 0.0
        return CompressionResult(
            output="\n".join(out_lines),
            original_lines=original_lines,
            compressed_lines=compressed_lines,
            compression_ratio=ratio,
            strategies_applied=["failures_only", "compress_passes"],
        )

    def compress_grep(self, text: str) -> CompressionResult:
        """Compress grep output by grouping matches per file."""
        lines = text.splitlines()
        original_lines = len(lines)

        file_counts: Counter[str] = Counter()
        for line in lines:
            if ":" in line:
                filename = line.split(":", 1)[0]
                if filename:
                    file_counts[filename] += 1

        out_lines = [
            f"{fname}: {count} match{'es' if count > 1 else ''}"
            for fname, count in sorted(file_counts.items())
        ]
        compressed_lines = len(out_lines)
        ratio = compressed_lines / original_lines if original_lines > 0 else 0.0
        return CompressionResult(
            output="\n".join(out_lines),
            original_lines=original_lines,
            compressed_lines=compressed_lines,
            compression_ratio=ratio,
            strategies_applied=["group_by_file"],
        )


DEFAULT_COMPRESSOR = OutputCompressor()

OUTPUT_COMPRESSOR_REGISTRY: dict[str, OutputCompressor] = {
    "default": DEFAULT_COMPRESSOR,
}
