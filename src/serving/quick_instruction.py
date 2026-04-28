"""Quick Instruction system — auxiliary task tokens appended to input.

Uses special tokens appended directly to the input sequence to trigger
auxiliary tasks (search, intent detection, domain classification, etc.)
without redundant prefilling. Reuses the existing KV cache.

Supported special tokens:
  <|action|>     — determines if web search is needed
  <|query|>      — generates search queries
  <|authority|>  — classifies source authority demand
  <|domain|>     — identifies domain of the prompt
  <|title|>      — generates conversation title
  <|extracted_url|> / <|read_url|> — URL fetch decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuickInstructionToken(Enum):
    ACTION = "<|action|>"
    QUERY = "<|query|>"
    AUTHORITY = "<|authority|>"
    DOMAIN = "<|domain|>"
    TITLE = "<|title|>"
    EXTRACTED_URL = "<|extracted_url|>"
    READ_URL = "<|read_url|>"

    @classmethod
    def token_ids(cls, tokenizer) -> dict[str, int]:
        return {t.value: tokenizer.encode(t.value)[-1] for t in cls}


@dataclass
class QuickInstructionConfig:
    enabled: bool = False
    search_action: bool = True
    search_query: bool = True
    authority_classification: bool = False
    domain_classification: bool = True
    title_generation: bool = True
    url_fetch_decisions: bool = False


class QuickInstructionManager:
    """Manages appending and parsing quick instruction tokens.

    Appends special tokens to the input and parses the corresponding
    logit outputs to determine auxiliary task results.
    """

    def __init__(self, config: QuickInstructionConfig, token_ids: dict[str, int]):
        self.config = config
        self.token_ids = token_ids

    def build_prompt_suffix(self, prompt: str) -> str:
        if not self.config.enabled:
            return ""

        suffix = ""
        if self.config.search_action:
            suffix += QuickInstructionToken.ACTION.value
        if self.config.search_query:
            suffix += QuickInstructionToken.QUERY.value
        if self.config.authority_classification:
            suffix += QuickInstructionToken.AUTHORITY.value
        if self.config.domain_classification:
            suffix += QuickInstructionToken.DOMAIN.value
        return suffix

    def should_search(self, action_logits: list[float]) -> bool:
        return max(action_logits) > 0.5

    def extract_query(self, query_logits: list[int], tokenizer) -> str:
        return tokenizer.decode(query_logits)

    def classify_domain(
        self, domain_logits: list[float], domain_labels: list[str]
    ) -> str:
        idx = max(range(len(domain_logits)), key=lambda i: domain_logits[i])
        return domain_labels[idx] if idx < len(domain_labels) else "general"

    def build_title_suffix(self) -> str:
        if not self.config.enabled:
            return ""
        if self.config.title_generation:
            return QuickInstructionToken.TITLE.value
        return ""
