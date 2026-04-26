"""BPE tokenizer trainer -- pure stdlib implementation.

Reference: Sennrich et al. 2015 "Neural Machine Translation of Rare Words with
Subword Units" (arXiv:1508.07909); GPT-2 byte-level BPE.

Learns a byte-level BPE merge table from a text corpus up to a target vocabulary
size. No external dependencies (no HF tokenizers, sentencepiece, etc.).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

# --- byte-level mapping (GPT-2 style) -----------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """Reversible mapping from bytes (0..255) to unicode code points that are
    printable / non-whitespace. Matches GPT-2 byte-level BPE.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {v: k for k, v in _BYTE_ENCODER.items()}


# --- config -------------------------------------------------------------------


@dataclass
class BPEConfig:
    vocab_size: int = 8000
    special_tokens: list[str] = field(default_factory=list)
    pretokenize_regex: str = r"\S+"
    byte_level: bool = True


# --- trainer ------------------------------------------------------------------


class BPETrainer:
    def __init__(self, config: BPEConfig):
        if config.special_tokens is None:
            config.special_tokens = []
        self.config = config

    def _pretokenize(self, text: str) -> list[str]:
        return re.findall(self.config.pretokenize_regex, text)

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        if self.config.byte_level:
            return tuple(_BYTE_ENCODER[b] for b in word.encode("utf-8"))
        return tuple(word)

    def train(self, texts: list[str]) -> dict:
        cfg = self.config
        specials = list(cfg.special_tokens or [])

        if not texts or all(not t for t in texts):
            raise ValueError("empty corpus: cannot train BPE")

        # minimum vocab: 256 base bytes (if byte_level) + specials
        base_size = 256 if cfg.byte_level else 0
        if cfg.vocab_size < base_size + len(specials):
            raise ValueError(
                f"vocab_size={cfg.vocab_size} too small; need >= "
                f"{base_size + len(specials)} for base bytes + specials"
            )

        # strip special tokens from the training text (they shouldn't be
        # merged apart) then pretokenize.
        word_counts: Counter[tuple[str, ...]] = Counter()
        special_pat = None
        if specials:
            special_pat = re.compile("|".join(re.escape(s) for s in specials))
        for text in texts:
            if not text:
                continue
            if special_pat is not None:
                parts = special_pat.split(text)
            else:
                parts = [text]
            for part in parts:
                for tok in self._pretokenize(part):
                    word_counts[self._word_to_symbols(tok)] += 1

        if not word_counts:
            raise ValueError("empty corpus after pretokenization")

        # initialize vocab with specials first, then base bytes, then merges
        vocab: dict[str, int] = {}
        for s in specials:
            if s not in vocab:
                vocab[s] = len(vocab)

        if cfg.byte_level:
            for byte_val in range(256):
                tok = _BYTE_ENCODER[byte_val]
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        else:
            # seed with observed characters (sorted for determinism)
            seen = sorted({sym for word in word_counts for sym in word})
            for sym in seen:
                if sym not in vocab:
                    vocab[sym] = len(vocab)

        merges: list[tuple[str, str]] = []
        words: list[list[str]] = [list(w) for w in word_counts.keys()]
        freqs: list[int] = list(word_counts.values())

        target_merges = cfg.vocab_size - len(vocab)
        for _ in range(max(0, target_merges)):
            # count adjacent pairs
            pair_counts: Counter[tuple[str, str]] = Counter()
            for w, f in zip(words, freqs):
                for i in range(len(w) - 1):
                    pair_counts[(w[i], w[i + 1])] += f
            if not pair_counts:
                break
            # tie-break deterministically: highest count, then lex order
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            pair, cnt = best
            if cnt < 1:
                break
            merges.append(pair)
            new_tok = pair[0] + pair[1]
            if new_tok not in vocab:
                vocab[new_tok] = len(vocab)
            # apply merge across all words
            a, b = pair
            new_words: list[list[str]] = []
            for w in words:
                if len(w) < 2:
                    new_words.append(w)
                    continue
                out: list[str] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new_words.append(out)
            words = new_words
            if len(vocab) >= cfg.vocab_size:
                break

        return {
            "vocab": vocab,
            "merges": merges,
            "special_tokens": specials,
            "byte_level": cfg.byte_level,
            "pretokenize_regex": cfg.pretokenize_regex,
        }

    def save(self, tokenizer: dict, path: str) -> None:
        ser = {
            "vocab": tokenizer["vocab"],
            "merges": [list(p) for p in tokenizer["merges"]],
            "special_tokens": tokenizer.get("special_tokens", []),
            "byte_level": tokenizer.get("byte_level", True),
            "pretokenize_regex": tokenizer.get("pretokenize_regex", r"\S+"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ser, f, ensure_ascii=False)

    def load(self, path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["merges"] = [tuple(p) for p in data["merges"]]
        return data


# --- tokenizer (encode/decode) ------------------------------------------------


class BPETokenizer:
    def __init__(
        self,
        vocab: dict,
        merges: list,
        byte_level: bool = True,
        special_tokens: list[str] | None = None,
        pretokenize_regex: str = r"\S+",
    ):
        self.vocab: dict[str, int] = dict(vocab)
        self.inv_vocab: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: list[tuple[str, str]] = [tuple(p) for p in merges]
        self.bpe_ranks: dict[tuple[str, str], int] = {p: i for i, p in enumerate(self.merges)}
        self.byte_level = byte_level
        self.special_tokens = list(special_tokens or [])
        self.pretokenize_regex = pretokenize_regex

    def _word_to_symbols(self, word: str) -> list[str]:
        if self.byte_level:
            return [_BYTE_ENCODER[b] for b in word.encode("utf-8")]
        return list(word)

    def _bpe(self, symbols: list[str]) -> list[str]:
        if len(symbols) < 2:
            return symbols
        word = list(symbols)
        while True:
            # find best-ranked pair
            best_rank = None
            best_i = -1
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                r = self.bpe_ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            a, b = word[best_i], word[best_i + 1]
            merged = a + b
            # merge all non-overlapping occurrences
            new_word: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) == 1:
                break
        return word

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        ids: list[int] = []
        # split on special tokens first
        if self.special_tokens:
            pat = re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
            parts = pat.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.vocab[part])
                continue
            for tok in re.findall(self.pretokenize_regex, part):
                syms = self._word_to_symbols(tok)
                pieces = self._bpe(syms)
                for p in pieces:
                    if p not in self.vocab:
                        # byte-level fallback: emit each byte-symbol
                        if self.byte_level:
                            for ch in p:
                                if ch in self.vocab:
                                    ids.append(self.vocab[ch])
                                else:
                                    raise KeyError(f"symbol {ch!r} not in vocab")
                        else:
                            raise KeyError(f"token {p!r} not in vocab")
                    else:
                        ids.append(self.vocab[p])
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces: list[str] = []
        for i in ids:
            if i not in self.inv_vocab:
                raise KeyError(f"id {i} not in vocab")
            pieces.append(self.inv_vocab[i])
        if self.byte_level:
            # strip specials (keep literal), convert byte-symbols back
            out_bytes = bytearray()
            result_parts: list[str] = []
            for p in pieces:
                if p in self.special_tokens:
                    if out_bytes:
                        result_parts.append(out_bytes.decode("utf-8", errors="replace"))
                        out_bytes = bytearray()
                    result_parts.append(p)
                else:
                    for ch in p:
                        if ch in _BYTE_DECODER:
                            out_bytes.append(_BYTE_DECODER[ch])
                        else:
                            # unknown char -- utf-8 encode it
                            out_bytes.extend(ch.encode("utf-8"))
            if out_bytes:
                result_parts.append(out_bytes.decode("utf-8", errors="replace"))
            return "".join(result_parts)
        return "".join(pieces)
