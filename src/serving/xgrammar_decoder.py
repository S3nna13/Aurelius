"""XGrammar-backed structured-output decoder for Aurelius serving.

Provides :class:`XGrammarDecoder` — a drop-in replacement for
:class:`StructuredOutputDecoder` that uses the XGrammar C++ kernel
(<40 µs per-token overhead) instead of the Python stdlib state machine.

If ``xgrammar`` is not installed, :class:`XGrammarDecoder` transparently
falls back to :class:`StructuredOutputDecoder` so the serving layer never
breaks.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import torch

from .structured_output_decoder import STRUCTURED_OUTPUT_REGISTRY, StructuredOutputDecoder

__all__ = ["XGrammarDecoder"]

# ---------------------------------------------------------------------------
# Lazy xgrammar import
# ---------------------------------------------------------------------------

try:
    import xgrammar as xgr

    _XGRAMMAR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGRAMMAR_AVAILABLE = False


# ---------------------------------------------------------------------------
# XGrammarDecoder
# ---------------------------------------------------------------------------

if _XGRAMMAR_AVAILABLE:

    class XGrammarDecoder:
        """Fast JSON-schema–guided decoder powered by XGrammar.

        Mirrors the public interface of :class:`StructuredOutputDecoder` so it
        can be swapped in without changing call-sites.

        Parameters
        ----------
        vocab_size:
            Number of tokens in the model vocabulary.
        eos_token_id:
            Token ID for the end-of-sequence symbol.
        """

        def __init__(self, vocab_size: int, eos_token_id: int) -> None:
            if vocab_size <= 0:
                raise ValueError(f"vocab_size must be positive, got {vocab_size}")
            self.vocab_size = vocab_size
            self.eos_token_id = eos_token_id
            self._fallback = StructuredOutputDecoder(vocab_size, eos_token_id)
            self._tokenizer_info_cache: dict[str, Any] = {}
            self._compiler_cache: dict[str, Any] = {}
            self._grammar_cache: dict[tuple[str, str], Any] = {}
            self._arange_cache: dict[torch.device, torch.Tensor] = {}

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        @staticmethod
        def _normalize_schema(schema: dict | str) -> dict:
            if isinstance(schema, str):
                return json.loads(schema)
            return schema

        def _get_tokenizer_info(self, vocab: list[str]) -> Any:
            vocab_hash = hashlib.sha256("".join(vocab).encode()).hexdigest()
            if vocab_hash not in self._tokenizer_info_cache:
                self._tokenizer_info_cache[vocab_hash] = xgr.TokenizerInfo(
                    encoded_vocab=vocab,
                    vocab_type=xgr.VocabType.RAW,
                    stop_token_ids=[self.eos_token_id],
                )
            return self._tokenizer_info_cache[vocab_hash]

        def _get_compiler(self, vocab: list[str]) -> Any:
            vocab_hash = hashlib.sha256("".join(vocab).encode()).hexdigest()
            if vocab_hash not in self._compiler_cache:
                tokenizer_info = self._get_tokenizer_info(vocab)
                self._compiler_cache[vocab_hash] = xgr.GrammarCompiler(
                    tokenizer_info,
                )
            return self._compiler_cache[vocab_hash]

        def _compile_schema(self, schema: dict | str, vocab: list[str]) -> Any:
            schema_str = json.dumps(schema, sort_keys=True) if isinstance(schema, dict) else schema
            vocab_hash = hashlib.sha256("".join(vocab).encode()).hexdigest()
            cache_key = (vocab_hash, schema_str)
            if cache_key not in self._grammar_cache:
                compiler = self._get_compiler(vocab)
                self._grammar_cache[cache_key] = compiler.compile_json_schema(schema_str)
            return self._grammar_cache[cache_key]

        @staticmethod
        def _build_trie(vocab: list[str]) -> dict:
            trie: dict = {}
            for tid, tok in enumerate(vocab):
                node = trie
                for ch in tok:
                    if ch not in node:
                        node[ch] = {}
                    node = node[ch]
                node["__token_id__"] = tid
                node["__token_len__"] = len(tok)
            return trie

        @staticmethod
        def _tokenize(text: str, vocab: list[str]) -> list[int]:
            """Greedy longest-match tokenization of *text* using *vocab*."""
            trie = XGrammarDecoder._build_trie(vocab)
            token_ids: list[int] = []
            pos = 0
            while pos < len(text):
                node = trie
                best_id: int | None = None
                best_len = 0
                i = pos
                while i < len(text) and text[i] in node:
                    node = node[text[i]]
                    if "__token_id__" in node:
                        best_id = node["__token_id__"]
                        best_len = node["__token_len__"]
                    i += 1
                if best_id is None:
                    pos += 1
                    continue
                token_ids.append(best_id)
                pos += best_len
            return token_ids

        @staticmethod
        def _bitmask_to_bool_mask(bitmask: torch.Tensor, vocab_size: int) -> torch.Tensor:
            """Convert an xgrammar int32 bitmask to a 1-D bool tensor.

            Parameters
            ----------
            bitmask:
                Tensor of shape ``[1, num_words]`` or ``[num_words]`` with
                dtype ``torch.int32`` where ``num_words = ceil(vocab_size/32)``.
            vocab_size:
                Size of the vocabulary.

            Returns
            -------
            torch.Tensor
                Boolean tensor of shape ``[vocab_size]``.
            """
            bitmask = bitmask.view(-1)
            if bitmask.device not in self._arange_cache:
                self._arange_cache[bitmask.device] = torch.arange(
                    self.vocab_size, device=bitmask.device
                )
            token_ids = self._arange_cache[bitmask.device]
            word_indices = token_ids // 32
            bit_positions = token_ids % 32
            return ((bitmask[word_indices] >> bit_positions) & 1).bool()

        # ------------------------------------------------------------------
        # Public API (StructuredOutputDecoder-compatible)
        # ------------------------------------------------------------------

        def is_valid_prefix(self, schema: dict, partial_json: str) -> bool:
            """Return True if *partial_json* is a valid prefix of some JSON
            matching *schema*.  Delegates to the fallback validator for
            correctness.
            """
            return self._fallback.is_valid_prefix(schema, partial_json)

        def is_complete(self, schema: dict, json_str: str) -> bool:
            """Return True if *json_str* is a complete, schema-conforming JSON
            document.  Delegates to the fallback validator for correctness.
            """
            return self._fallback.is_complete(schema, json_str)

        def build_token_mask_from_schema(
            self,
            schema: dict,
            partial_output: str,
            vocab: list[str],
        ) -> torch.Tensor:
            """Return a boolean mask of shape ``[vocab_size]``.

            ``mask[i] == True`` iff token *i* is an allowed next token
            according to the XGrammar-compiled JSON schema.
            """
            if len(vocab) != self.vocab_size:
                raise ValueError(
                    f"vocab length {len(vocab)} != vocab_size {self.vocab_size}"
                )

            grammar = self._compile_schema(schema, vocab)
            matcher = xgr.GrammarMatcher(grammar)

            token_ids = self._tokenize(partial_output, vocab)
            for tid in token_ids:
                if not matcher.accept_token(tid):
                    # Partial output violates the grammar — only allow EOS so
                    # generation can terminate rather than entering an invalid
                    # state.
                    mask = torch.zeros(self.vocab_size, dtype=torch.bool)
                    mask[self.eos_token_id] = True
                    return mask

            token_bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)
            matcher.fill_next_token_bitmask(token_bitmask)
            mask = self._bitmask_to_bool_mask(token_bitmask, self.vocab_size)

            # EOS is always allowed once the matcher has terminated.
            if matcher.is_terminated():
                mask[self.eos_token_id] = True

            # Safety fallback: if the grammar is so restrictive that nothing
            # is allowed, permit EOS to avoid an all-inf logits row downstream.
            if not mask.any():
                mask[self.eos_token_id] = True

            return mask

        def get_mask(
            self,
            schema: dict,
            partial_output: str,
            vocab: list[str],
        ) -> torch.Tensor:
            """Alias for :meth:`build_token_mask_from_schema`.

            Returns a boolean tensor of shape ``[vocab_size]``.
            """
            return self.build_token_mask_from_schema(schema, partial_output, vocab)

        def constrained_logits(
            self,
            logits: torch.Tensor,
            schema: dict,
            partial_output: str,
            vocab: list[str],
        ) -> torch.Tensor:
            """Return a copy of *logits* with disallowed token positions set to
            ``-inf``.
            """
            mask = self.build_token_mask_from_schema(schema, partial_output, vocab)
            out = logits.clone()
            if out.dim() == 2:
                out[:, ~mask] = float("-inf")
            else:
                out[~mask] = float("-inf")
            return out

else:
    # xgrammar is not installed — fall back transparently.
    XGrammarDecoder = StructuredOutputDecoder  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Registry injection — give XGrammarDecoder priority when available.
# ---------------------------------------------------------------------------

STRUCTURED_OUTPUT_REGISTRY["json_schema"] = XGrammarDecoder
