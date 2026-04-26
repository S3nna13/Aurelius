"""Cross-provider parallel consensus with pluggable adjudicator."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ProviderResponse",
    "AdjudicationStrategy",
    "ConsensusResult",
    "ParallelConsensusEngine",
]

_DEFAULT_PROVIDERS = ["local", "openai", "anthropic"]


@dataclass
class ProviderResponse:
    provider: str
    response: str
    latency_ms: float
    confidence: float


class AdjudicationStrategy(StrEnum):
    MAJORITY = "majority"
    HIGHEST_CONFIDENCE = "highest"
    LLM_ADJUDICATOR = "llm"
    WEIGHTED = "weighted"


@dataclass
class ConsensusResult:
    winner: ProviderResponse
    all_responses: list[ProviderResponse]
    strategy: AdjudicationStrategy
    agreement_score: float


def _hash_confidence(provider: str, prompt: str) -> float:
    """Deterministic 0–1 confidence from provider+prompt digest."""
    digest = hashlib.sha256(f"{provider}:{prompt}".encode()).digest()
    return (int.from_bytes(digest[:4], "big") % 1000) / 1000.0


def _hash_response(provider: str, prompt: str) -> str:
    """Deterministic stub response from provider+prompt digest."""
    digest = hashlib.sha256(f"resp:{provider}:{prompt}".encode()).hexdigest()
    return f"[{provider}] response-{digest[:12]}"


def _hash_latency(provider: str, prompt: str) -> float:
    """Deterministic latency 10–100 ms from digest."""
    digest = hashlib.sha256(f"lat:{provider}:{prompt}".encode()).digest()
    return 10.0 + (int.from_bytes(digest[:2], "big") % 90)


class ParallelConsensusEngine:
    """Cross-provider parallel consensus with pluggable adjudicator."""

    def __init__(
        self,
        strategy: AdjudicationStrategy = AdjudicationStrategy.WEIGHTED,
        providers: list[str] | None = None,
    ) -> None:
        self._strategy = strategy
        self._providers: list[str] = list(providers) if providers else list(_DEFAULT_PROVIDERS)

    async def query_provider(self, provider: str, prompt: str) -> ProviderResponse:
        latency = _hash_latency(provider, prompt)
        await asyncio.sleep(latency / 1000.0)
        return ProviderResponse(
            provider=provider,
            response=_hash_response(provider, prompt),
            latency_ms=latency,
            confidence=_hash_confidence(provider, prompt),
        )

    async def gather_responses(self, prompt: str) -> list[ProviderResponse]:
        tasks = [self.query_provider(p, prompt) for p in self._providers]
        return list(await asyncio.gather(*tasks))

    def adjudicate(self, responses: list[ProviderResponse]) -> ConsensusResult:
        if not responses:
            raise ValueError("adjudicate requires at least one response")

        strategy = self._strategy

        if strategy == AdjudicationStrategy.HIGHEST_CONFIDENCE:
            winner = max(responses, key=lambda r: r.confidence)
        elif strategy == AdjudicationStrategy.LLM_ADJUDICATOR:
            winner = max(responses, key=lambda r: len(r.response))
        elif strategy == AdjudicationStrategy.MAJORITY:
            from collections import Counter

            counts: Counter[str] = Counter(r.response for r in responses)
            top_response = counts.most_common(1)[0][0]
            winner = next(r for r in responses if r.response == top_response)
        else:
            total_conf = sum(r.confidence for r in responses)
            if total_conf == 0.0:
                winner = responses[0]
            else:
                winner = max(responses, key=lambda r: r.confidence)

        agreeing = sum(1 for r in responses if r.response == winner.response)
        agreement_score = agreeing / len(responses)

        return ConsensusResult(
            winner=winner,
            all_responses=responses,
            strategy=strategy,
            agreement_score=agreement_score,
        )

    async def consensus(self, prompt: str) -> ConsensusResult:
        responses = await self.gather_responses(prompt)
        return self.adjudicate(responses)

    def add_provider(self, provider: str) -> None:
        if provider not in self._providers:
            self._providers.append(provider)

    def remove_provider(self, provider: str) -> None:
        self._providers = [p for p in self._providers if p != provider]
