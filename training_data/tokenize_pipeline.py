from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class TokenizePipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._tokenizer: Callable | None = None
        self._vocab_size: int = config.get("tokenizer", {}).get("vocab_size", 128_000)
        self._max_length: int = config.get("tokenize", {}).get("max_length", 8192)
        self._num_workers: int = config.get("tokenize", {}).get("num_workers", 4)
        self._verify: bool = config.get("tokenize", {}).get("verify_integrity", True)

    def load_tokenizer(self) -> Callable:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from src.data.tokenizer import AureliusTokenizer

            tok = AureliusTokenizer.load(
                self.config.get("tokenizer", {}).get("path", "tokenizers/aurelius-128k")
            )
            self._vocab_size = tok.vocab_size
            self._tokenizer = tok
            return tok
        except Exception:
            logger.warning("AureliusTokenizer not available, using fallback BPE tokenizer")
            tok = self._build_fallback_tokenizer()
            self._tokenizer = tok
            return tok

    def _build_fallback_tokenizer(self) -> Callable:
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel

            tok = HFTokenizer(BPE(unk_token="<|unk|>"))
            tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self._vocab_size = 128_000
            return tok
        except ImportError:
            logger.warning("tokenizers package not installed, using simple whitespace splitter")

            class SimpleTokenizer:
                @property
                def vocab_size(self) -> int:
                    return 128_000

                def encode(self, text: str) -> list[int]:
                    return [min(ord(c), 127_999) for c in text]

                def encode_batch(self, texts: list[str]) -> list[list[int]]:
                    return [self.encode(t) for t in texts]

            return SimpleTokenizer()  # type: ignore

    def _tokenize_text(self, tokenizer: Callable, text: str) -> list[int]:
        try:
            ids = tokenizer.encode(text)
        except AttributeError:
            ids = tokenizer.encode_batch([text])[0]
        if self._max_length:
            ids = ids[: self._max_length]
        ids = [max(0, min(i, self._vocab_size - 1)) for i in ids]
        return ids

    def tokenize_texts(
        self, texts: list[str], output_dir: str, shard_size: int = 16384
    ) -> list[str]:
        tokenizer = self.load_tokenizer()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_workers = min(self._num_workers, multiprocessing.cpu_count())

        if n_workers > 1 and len(texts) > shard_size:
            with multiprocessing.Pool(n_workers) as pool:
                all_ids = pool.map(self._tokenize_text, [tokenizer] * len(texts), texts)
        else:
            all_ids = []
            for t in texts:
                try:
                    ids = self._tokenize_text(tokenizer, t)
                except Exception:
                    ids = [0]
                all_ids.append(ids)

        shard_paths: list[str] = []
        for shard_idx in range(0, len(all_ids), shard_size):
            shard_data = all_ids[shard_idx : shard_idx + shard_size]
            max_len = max((len(s) for s in shard_data), default=0)
            arr = np.zeros((len(shard_data), max_len), dtype=np.uint16)
            for i, seq in enumerate(shard_data):
                for j, tid in enumerate(seq):
                    arr[i, j] = tid

            shard_file = output_path / f"shard_{shard_idx // shard_size:06d}.npy"
            np.save(str(shard_file), arr)
            shard_paths.append(str(shard_file))

            if self._verify:
                if not self.validate_shard(str(shard_file)):
                    logger.warning("Shard %s failed integrity check", shard_file)

        logger.info("Tokenized %d texts into %d shards at %s", len(texts), len(shard_paths), output_dir)
        return shard_paths

    def tokenize_jsonl(
        self,
        input_path: str,
        output_dir: str,
        text_field: str = "text",
        shard_size: int = 16384,
    ) -> int:
        texts: list[str] = []
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get(text_field)
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

        if not texts:
            logger.warning("No texts found in %s", input_path)
            return 0

        self.tokenize_texts(texts, output_dir, shard_size=shard_size)
        return len(texts)

    def create_shard_manifest(self, shard_dir: str, output_path: str) -> dict:
        shard_dir_p = Path(shard_dir)
        shard_files = sorted(shard_dir_p.glob("shard_*.npy"))

        total_tokens = 0
        shards_info = []
        for sf in shard_files:
            arr = np.load(str(sf))
            total_tokens += int(arr.sum())
            try:
                h = self.compute_hash(str(sf))
            except Exception:
                h = ""
            shards_info.append({"path": str(sf), "samples": int(arr.shape[0]), "hash": h})

        manifest = {
            "shard_count": len(shard_files),
            "total_tokens": total_tokens,
            "shards": shards_info,
            "hash": hashlib.sha256(str(sorted(shard_files)).encode()).hexdigest(),
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        return manifest

    def split_train_val(
        self, shard_paths: list[str], val_ratio: float = 0.1, seed: int = 42
    ) -> tuple[list[str], list[str]]:
        rng = random.Random(seed)
        paths = list(shard_paths)
        rng.shuffle(paths)
        split_idx = max(1, int(len(paths) * (1 - val_ratio)))
        train_paths = paths[:split_idx]
        val_paths = paths[split_idx:]
        return train_paths, val_paths

    def interleave_datasets(
        self, shard_dirs: list[str], output_dir: str, weights: list[float]
    ) -> int:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]

        all_shards: list[tuple[str, float]] = []
        for shard_dir, weight in zip(shard_dirs, norm_weights):
            dir_p = Path(shard_dir)
            shards = sorted(dir_p.glob("shard_*.npy"))
            for s in shards:
                all_shards.append((str(s), weight))

        if not all_shards:
            logger.warning("No shards found to interleave")
            return 0

        rng = random.Random(42)
        rng.shuffle(all_shards)

        combined_tokens = 0
        for idx, (shard_path, _) in enumerate(all_shards):
            arr = np.load(shard_path)
            out_file = output_path / f"shard_{idx:06d}.npy"
            np.save(str(out_file), arr)
            combined_tokens += int(arr.sum())

        logger.info(
            "Interleaved %d shards from %d datasets into %s",
            len(all_shards),
            len(shard_dirs),
            output_dir,
        )
        return combined_tokens

    def validate_shard(self, shard_path: str) -> bool:
        try:
            arr = np.load(shard_path)
            if arr.dtype != np.uint16:
                logger.error("Shard %s has wrong dtype: %s", shard_path, arr.dtype)
                return False
            if arr.ndim != 2:
                logger.error("Shard %s has wrong ndim: %d", shard_path, arr.ndim)
                return False
            if arr.shape[0] == 0:
                logger.error("Shard %s has zero samples", shard_path)
                return False
            if np.any(np.isnan(arr.astype(float))):
                logger.error("Shard %s contains NaN values", shard_path)
                return False
            if arr.max() >= self._vocab_size:
                logger.warning(
                    "Shard %s has token IDs >= vocab_size (%d)", shard_path, self._vocab_size
                )
            return True
        except Exception as exc:
            logger.error("Shard validation failed for %s: %s", shard_path, exc)
            return False

    def compute_hash(self, shard_path: str) -> str:
        h = hashlib.sha256()
        with open(shard_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
