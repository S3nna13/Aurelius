"""Aurelius 128K BPE tokenizer — training, saving, loading, encoding, decoding.

Uses HuggingFace ``tokenizers`` (Rust-backed) for fast byte-level BPE.
Designed for a 128 000-token vocabulary with 512 reserved special-token slots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from tokenizers.normalizers import NFC

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE: int = 128_000
NUM_RESERVED_SPECIAL: int = 512

SPECIAL_TOKENS: list[str] = [
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|fim_prefix|>",
    "<|fim_suffix|>",
    "<|fim_middle|>",
    "<|tool_call|>",
    "<|tool_result|>",
]

# Pad the reserved block with placeholder tokens so future additions don't
# shift existing IDs.  Named special tokens occupy the first len(SPECIAL_TOKENS)
# slots; the rest are "<|reserved_N|>".
RESERVED_TOKENS: list[str] = SPECIAL_TOKENS + [
    f"<|reserved_{i}|>"
    for i in range(len(SPECIAL_TOKENS), NUM_RESERVED_SPECIAL)
]

# Explicit multi-space tokens for code-heavy corpora.
MULTI_SPACE_TOKENS: list[str] = [
    "    ",   # 4 spaces (standard indent)
    "  ",     # 2 spaces
    "        ",  # 8 spaces (double indent)
]

# Convenience name -> id mapping (populated after training / loading).
_SPECIAL_NAME_TO_ID: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class AureliusTokenizer:
    """High-level wrapper around a byte-level BPE ``Tokenizer``.

    Parameters
    ----------
    tokenizer:
        An already-built ``tokenizers.Tokenizer`` instance.  Prefer the
        class-methods :meth:`train` and :meth:`load` instead of constructing
        directly.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok: Tokenizer = tokenizer
        self._build_special_id_map()

    # -- convenience properties ---------------------------------------------

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return self._special_id("<|bos|>")

    @property
    def eos_id(self) -> int:
        return self._special_id("<|eos|>")

    @property
    def pad_id(self) -> int:
        return self._special_id("<|pad|>")

    @property
    def unk_id(self) -> int:
        return self._special_id("<|unk|>")

    # -- encode / decode ----------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode *text* to token ids.

        Parameters
        ----------
        text:
            Raw string to tokenize.
        add_bos:
            If ``True``, prepend the ``<|bos|>`` id.
        add_eos:
            If ``True``, append the ``<|eos|>`` id.
        """
        ids: list[int] = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_id, *ids]
        if add_eos:
            ids = [*ids, self.eos_id]
        return ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[list[int]]:
        """Encode a batch of strings in parallel (Rust-level parallelism)."""
        encodings = self._tok.encode_batch(texts)
        results: list[list[int]] = []
        for enc in encodings:
            ids = enc.ids
            if add_bos:
                ids = [self.bos_id, *ids]
            if add_eos:
                ids = [*ids, self.eos_id]
            results.append(ids)
        return results

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token ids back to a string."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Decode a batch of id-sequences in parallel."""
        return self._tok.decode_batch(
            batch_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def token_to_id(self, token: str) -> int | None:
        """Return the id for *token*, or ``None`` if not in the vocabulary."""
        return self._tok.token_to_id(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Return the token string for *token_id*, or ``None``."""
        return self._tok.id_to_token(token_id)

    # -- persistence --------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Save the tokenizer to *directory*.

        Creates ``tokenizer.json`` (full state) and ``special_tokens_map.json``
        for convenient inspection.

        Returns the path to ``tokenizer.json``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        tok_path = directory / "tokenizer.json"
        self._tok.save(str(tok_path))

        # Also dump a human-readable special-tokens map.
        special_map = {
            name: self._tok.token_to_id(name)
            for name in SPECIAL_TOKENS
            if self._tok.token_to_id(name) is not None
        }
        special_map_path = directory / "special_tokens_map.json"
        special_map_path.write_text(
            json.dumps(special_map, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        logger.info("Tokenizer saved to %s (vocab_size=%d)", tok_path, self.vocab_size)
        return tok_path

    @classmethod
    def load(cls, directory: str | Path) -> "AureliusTokenizer":
        """Load a previously-saved tokenizer from *directory*."""
        directory = Path(directory)
        tok_path = directory / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"No tokenizer.json found in {directory}")
        tokenizer = Tokenizer.from_file(str(tok_path))
        logger.info("Tokenizer loaded from %s", tok_path)
        return cls(tokenizer)

    # -- training -----------------------------------------------------------

    @classmethod
    def train(
        cls,
        iterator: Iterator[str],
        *,
        vocab_size: int = VOCAB_SIZE,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> "AureliusTokenizer":
        """Train a new byte-level BPE tokenizer from an iterator of strings.

        Parameters
        ----------
        iterator:
            Yields raw text samples (one document per yield).
        vocab_size:
            Target vocabulary size **including** reserved special tokens.
        min_frequency:
            Minimum merge frequency during BPE training.
        show_progress:
            Display a progress bar during training.
        """
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))

        # Normalizer: NFC unicode normalisation (preserves whitespace).
        tokenizer.normalizer = NFC()

        # Pre-tokenizer: byte-level (GPT-2 style) — handles all UTF-8.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Decoder must match the pre-tokenizer.
        tokenizer.decoder = decoders.ByteLevel()

        # Post-processor: byte-level offset trimming.
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

        # Trainer ----------------------------------------------------------------
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=RESERVED_TOKENS,  # IDs 0..511
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        logger.info(
            "Starting BPE training (target vocab_size=%d, reserved=%d)...",
            vocab_size,
            NUM_RESERVED_SPECIAL,
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer)

        # Add explicit multi-space tokens if not already present.
        for sp_tok in MULTI_SPACE_TOKENS:
            if tokenizer.token_to_id(sp_tok) is None:
                tokenizer.add_tokens([sp_tok])

        instance = cls(tokenizer)
        logger.info(
            "Training complete. Final vocab_size=%d", instance.vocab_size,
        )
        return instance

    # -- internals ----------------------------------------------------------

    def _build_special_id_map(self) -> None:
        """Cache name -> id for named special tokens."""
        self._special_map: dict[str, int] = {}
        for name in SPECIAL_TOKENS:
            tid = self._tok.token_to_id(name)
            if tid is not None:
                self._special_map[name] = tid

    def _special_id(self, name: str) -> int:
        """Return the id for a named special token, raising on miss."""
        try:
            return self._special_map[name]
        except KeyError:
            raise ValueError(
                f"Special token {name!r} not found in vocabulary"
            ) from None

    def __repr__(self) -> str:
        return f"AureliusTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# Streaming dataset helper
# ---------------------------------------------------------------------------

def stream_hf_dataset(
    dataset_name: str = "allenai/dolma",
    split: str = "train",
    text_field: str = "text",
    max_samples: int | None = None,
) -> Iterator[str]:
    """Yield text samples from a HuggingFace dataset in streaming mode.

    This avoids downloading the entire dataset to disk.

    Parameters
    ----------
    dataset_name:
        HuggingFace dataset identifier (e.g. ``"allenai/dolma"``).
    split:
        Dataset split to stream.
    text_field:
        Name of the column containing text.
    max_samples:
        Stop after this many samples (``None`` = exhaust the stream).
    """
    from datasets import load_dataset

    logger.info(
        "Streaming dataset %s (split=%s, max_samples=%s)",
        dataset_name,
        split,
        max_samples,
    )
    ds = load_dataset(dataset_name, split=split, streaming=True)

    count = 0
    for sample in ds:
        text = sample.get(text_field)
        if text:
            yield text
            count += 1
            if max_samples is not None and count >= max_samples:
                break

    logger.info("Streamed %d samples from %s", count, dataset_name)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Train the Aurelius tokenizer from the command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train the Aurelius BPE tokenizer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="allenai/dolma",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Name of the text column in the dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (None = use all)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help="Target vocabulary size",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum BPE merge frequency",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tokenizer",
        help="Directory to save the trained tokenizer",
    )
    args = parser.parse_args()

    iterator = stream_hf_dataset(
        dataset_name=args.dataset,
        split=args.split,
        text_field=args.text_field,
        max_samples=args.max_samples,
    )

    tokenizer = AureliusTokenizer.train(
        iterator,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    tokenizer.save(args.output_dir)

    # Quick sanity check.
    test_text = "Hello, world! def main():\n    pass"
    ids = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(ids)
    logger.info("Sanity check — encode/decode round-trip:")
    logger.info("  input:   %r", test_text)
    logger.info("  ids:     %s", ids[:20])
    logger.info("  decoded: %r", decoded)


if __name__ == "__main__":
    main()
