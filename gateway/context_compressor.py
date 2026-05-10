"""Multi-turn context compression: truncation strategies, token budget enforcement."""

from dataclasses import dataclass
from enum import StrEnum


class CompressionStrategy(StrEnum):
    TRUNCATE_OLDEST = "truncate_oldest"
    SUMMARIZE_MIDDLE = "summarize_middle"
    DROP_TOOL_RESULTS = "drop_tool_results"


@dataclass
class CompressedTurn:
    role: str
    content: str
    was_compressed: bool = False


class ContextCompressor:
    """Compress multi-turn conversation histories to fit within a token budget."""

    def __init__(
        self,
        max_turns: int = 20,
        strategy: CompressionStrategy = CompressionStrategy.TRUNCATE_OLDEST,
    ) -> None:
        self.max_turns = max_turns
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, turns: list[dict]) -> list[CompressedTurn]:
        """Compress *turns* according to the configured strategy.

        Parameters
        ----------
        turns:
            List of dicts with at least "role" and "content" keys.

        Returns
        -------
        list[CompressedTurn]
            Kept turns only; dropped turns are *not* represented in the output.
        """
        if self.strategy == CompressionStrategy.TRUNCATE_OLDEST:
            return self._truncate_oldest(turns)
        elif self.strategy == CompressionStrategy.SUMMARIZE_MIDDLE:
            return self._summarize_middle(turns)
        elif self.strategy == CompressionStrategy.DROP_TOOL_RESULTS:
            return self._drop_tool_results(turns)
        else:
            # Fallback: return everything uncompressed.
            return [CompressedTurn(role=t["role"], content=t["content"]) for t in turns]

    def compression_ratio(self, original: list[dict], compressed: list[CompressedTurn]) -> float:
        """Return len(compressed) / len(original).

        Returns 1.0 when *original* is empty (no compression needed).
        """
        if not original:
            return 1.0
        return len(compressed) / len(original)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _truncate_oldest(self, turns: list[dict]) -> list[CompressedTurn]:
        """Keep the system turn + the most-recent *max_turns* non-system turns."""
        system_turns = [t for t in turns if t.get("role") == "system"]
        non_system = [t for t in turns if t.get("role") != "system"]

        kept_non_system = non_system[-self.max_turns :] if self.max_turns > 0 else []

        result: list[CompressedTurn] = []
        for t in system_turns:
            result.append(CompressedTurn(role=t["role"], content=t["content"]))
        for t in kept_non_system:
            result.append(CompressedTurn(role=t["role"], content=t["content"]))
        return result

    def _summarize_middle(self, turns: list[dict]) -> list[CompressedTurn]:
        """Keep first 2 turns + last (max_turns-2) turns; replace middle with summary."""
        n = len(turns)
        keep_first = 2
        keep_last = max(self.max_turns - keep_first, 0)

        # If fits entirely, return all.
        if n <= keep_first + keep_last:
            return [CompressedTurn(role=t["role"], content=t["content"]) for t in turns]

        first_turns = turns[:keep_first]
        last_turns = turns[n - keep_last :] if keep_last > 0 else []
        middle_count = n - keep_first - len(last_turns)

        result: list[CompressedTurn] = []
        for t in first_turns:
            result.append(CompressedTurn(role=t["role"], content=t["content"]))

        if middle_count > 0:
            result.append(
                CompressedTurn(
                    role="system",
                    content=f"[{middle_count} turns summarized]",
                    was_compressed=True,
                )
            )

        for t in last_turns:
            result.append(CompressedTurn(role=t["role"], content=t["content"]))

        return result

    def _drop_tool_results(self, turns: list[dict]) -> list[CompressedTurn]:
        """Remove turns where role=='tool', then keep last max_turns of remaining."""
        non_tool = [t for t in turns if t.get("role") != "tool"]
        kept = non_tool[-self.max_turns :] if self.max_turns > 0 else []
        return [CompressedTurn(role=t["role"], content=t["content"]) for t in kept]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONTEXT_COMPRESSOR_REGISTRY: dict[str, ContextCompressor] = {
    "default": ContextCompressor(
        max_turns=20,
        strategy=CompressionStrategy.TRUNCATE_OLDEST,
    ),
    "aggressive": ContextCompressor(
        max_turns=8,
        strategy=CompressionStrategy.TRUNCATE_OLDEST,
    ),
    "tool_drop": ContextCompressor(
        max_turns=20,
        strategy=CompressionStrategy.DROP_TOOL_RESULTS,
    ),
}
