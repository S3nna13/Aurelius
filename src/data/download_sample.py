"""Download a small sample of FineWeb for local development and testing.

Uses the HuggingFace ``datasets`` streaming API so we never need to pull
the full 44 TB dataset.  The script streams rows, counts tokens with a
fast byte-level estimator, and writes Parquet shards to disk.

Designed to run comfortably on an M1 Pro with 16-32 GB RAM.

Usage
-----
::

    # Download ~1B tokens (default) into ./data/fineweb/sample/
    python -m src.data.download_sample

    # Download 500M tokens into a custom directory
    python -m src.data.download_sample --target-tokens 500_000_000 \\
                                       --output-dir ./data/sample_500m

    # Use a specific dataset / subset
    python -m src.data.download_sample --dataset HuggingFaceFW/fineweb-edu
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Rough bytes-per-token ratio for English text with a BPE tokenizer.
# GPT-NeoX / LLaMA-family tokenizers average ~3.5-4.0 bytes per token on
# English web text.  We use 3.7 as a conservative middle ground.
BYTES_PER_TOKEN_ESTIMATE: float = 3.7


def estimate_tokens(text: str) -> int:
    """Fast token count estimate without loading a tokenizer."""
    return max(1, int(len(text.encode("utf-8")) / BYTES_PER_TOKEN_ESTIMATE))


# ---------------------------------------------------------------------------
# Streaming downloader
# ---------------------------------------------------------------------------

def stream_dataset(
    dataset_id: str,
    subset: str | None = None,
    split: str = "train",
    text_key: str = "text",
    seed: int = 42,
) -> Iterator[str]:
    """Yield text documents from a HuggingFace dataset via streaming."""
    kwargs: dict[str, Any] = {
        "path": dataset_id,
        "split": split,
        "streaming": True,
        "trust_remote_code": False,
    }
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(**kwargs)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    for row in ds:
        text = row.get(text_key)
        if text:
            yield text


def download_sample(
    dataset_id: str = "HuggingFaceFW/fineweb",
    subset: str | None = None,
    split: str = "train",
    text_key: str = "text",
    target_tokens: int = 1_000_000_000,
    output_dir: str = "./data/fineweb/sample",
    rows_per_shard: int = 50_000,
    seed: int = 42,
) -> Path:
    """Stream *target_tokens* tokens from HuggingFace and save as Parquet.

    Returns the output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_docs = 0
    shard_idx = 0
    buffer: list[dict[str, str]] = []

    t0 = time.monotonic()
    log_interval = 100_000  # log every N docs

    logger.info(
        "Streaming from %s (target: %s tokens) -> %s",
        dataset_id,
        f"{target_tokens:,}",
        out,
    )

    for text in stream_dataset(
        dataset_id, subset=subset, split=split, text_key=text_key, seed=seed
    ):
        tokens = estimate_tokens(text)
        buffer.append({"text": text})
        total_tokens += tokens
        total_docs += 1

        # Flush shard to disk when buffer is full.
        if len(buffer) >= rows_per_shard:
            _write_shard(out, shard_idx, buffer)
            shard_idx += 1
            buffer = []

        # Progress logging.
        if total_docs % log_interval == 0:
            elapsed = time.monotonic() - t0
            docs_per_sec = total_docs / max(elapsed, 1e-6)
            logger.info(
                "Progress: %s docs | ~%sB tokens | %.0f docs/s",
                f"{total_docs:,}",
                f"{total_tokens / 1e9:.3f}",
                docs_per_sec,
            )

        if total_tokens >= target_tokens:
            break

    # Flush remaining buffer.
    if buffer:
        _write_shard(out, shard_idx, buffer)
        shard_idx += 1

    elapsed = time.monotonic() - t0
    logger.info(
        "Done: %s docs, ~%sB tokens, %d shards in %.1fs",
        f"{total_docs:,}",
        f"{total_tokens / 1e9:.3f}",
        shard_idx,
        elapsed,
    )

    # Write a metadata file for downstream consumers.
    meta_path = out / "metadata.txt"
    meta_path.write_text(
        f"dataset: {dataset_id}\n"
        f"subset: {subset or 'default'}\n"
        f"split: {split}\n"
        f"total_docs: {total_docs}\n"
        f"estimated_tokens: {total_tokens}\n"
        f"shards: {shard_idx}\n"
        f"seed: {seed}\n",
        encoding="utf-8",
    )

    return out


def _write_shard(
    output_dir: Path,
    shard_idx: int,
    rows: list[dict[str, str]],
) -> None:
    """Write a list of dicts to a Parquet file."""
    table = pa.Table.from_pylist(rows)
    shard_path = output_dir / f"shard_{shard_idx:05d}.parquet"
    pq.write_table(table, shard_path, compression="zstd")
    logger.debug("Wrote shard %s (%d rows)", shard_path.name, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a token-budget sample of FineWeb via streaming.",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset ID.",
    )
    parser.add_argument("--subset", default=None, help="Dataset subset/config.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=1_000_000_000,
        help="Approximate number of tokens to download (default 1B).",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/fineweb/sample",
        help="Directory to write Parquet shards.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=50_000,
        help="Documents per Parquet shard file.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    download_sample(
        dataset_id=args.dataset,
        subset=args.subset,
        split=args.split,
        text_key=args.text_key,
        target_tokens=args.target_tokens,
        output_dir=args.output_dir,
        rows_per_shard=args.rows_per_shard,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
