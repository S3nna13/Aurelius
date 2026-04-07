"""DataTrove processing pipeline for Project Aurelius 1.3B.

Runs a multi-stage pipeline over FineWeb (and optionally code data):

1. **Read** -- stream Parquet/JSONL from HuggingFace or local disk.
2. **Filter** -- LanguageFilter, GopherQualityFilter, C4QualityFilter,
   FineWebQualityFilter.
3. **Deduplicate** -- MinHash (5-gram, 14 buckets x 8 hashes per bucket).
4. **Write** -- output cleaned Parquet shards.

Resume is handled natively by DataTrove's marker-file mechanism: each
task writes a ``.completed`` marker after finishing, so re-running the
pipeline skips already-completed tasks automatically.

Usage
-----
::

    python -m src.data.pipeline --input-path /data/fineweb/raw \\
                                --output-path /data/fineweb/clean \\
                                --tasks 64

Or import ``build_pipeline`` / ``build_dedup_pipeline`` and compose your
own executor.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    LanguageFilter,
    LambdaFilter,
)
from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import ParquetWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """All tunables for the Aurelius data pipeline."""

    # I/O
    input_path: str = "hf://datasets/HuggingFaceFW/fineweb"
    output_path: str = "./data/fineweb/clean"
    minhash_work_dir: str = "./data/fineweb/minhash"
    input_format: str = "parquet"  # "parquet" or "jsonl"
    text_key: str = "text"

    # Language filter
    language: str = "en"
    language_threshold: float = 0.65

    # MinHash dedup
    minhash_ngram_size: int = 5
    minhash_num_buckets: int = 14
    minhash_hashes_per_bucket: int = 8

    # FineWeb-Edu quality score filter
    # Documents scored 0-5 by Llama-3-70B for educational quality.
    # >= 3.0 matches FineWeb-Edu's default threshold (8x token reduction).
    # Set to 0.0 to disable (e.g. for non-FineWeb-Edu data).
    edu_score_min: float = 3.0

    # Executor
    tasks: int = 64
    workers: int = -1  # -1 = auto (number of CPUs)
    logging_dir: str = "./logs/data_pipeline"

    @property
    def total_minhash_hashes(self) -> int:
        return self.minhash_num_buckets * self.minhash_hashes_per_bucket


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _get_reader(
    config: PipelineConfig,
    glob_pattern: str | None = None,
) -> ParquetReader | JsonlReader:
    """Instantiate the appropriate DataTrove reader."""
    kwargs: dict = {
        "data_folder": config.input_path,
        "text_key": config.text_key,
    }
    if glob_pattern:
        kwargs["glob_pattern"] = glob_pattern

    if config.input_format == "parquet":
        return ParquetReader(**kwargs)
    if config.input_format == "jsonl":
        return JsonlReader(**kwargs)
    raise ValueError(f"Unsupported input format: {config.input_format!r}")


def build_filter_pipeline(
    config: PipelineConfig,
    glob_pattern: str | None = None,
) -> LocalPipelineExecutor:
    """Stage 1 -- read, filter, and write intermediate Parquet.

    The output of this stage feeds into the MinHash dedup stages.
    """
    intermediate_output = f"{config.output_path}/filtered"

    pipeline = [
        _get_reader(config, glob_pattern),
        LanguageFilter(
            language_threshold=config.language_threshold,
            languages=[config.language],
        ),
        GopherQualityFilter(),
        C4QualityFilter(),
        FineWebQualityFilter(),
        LambdaFilter(
            filter_function=lambda doc: doc.metadata.get("int_score", doc.metadata.get("score", 0)) >= config.edu_score_min,
            name="EduScoreFilter",
        ),
        ParquetWriter(
            output_folder=intermediate_output,
            max_file_size=5 * 2**30,  # 5 GiB per shard
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=config.tasks,
        workers=config.workers,
        logging_dir=f"{config.logging_dir}/01_filter",
    )


def build_minhash_signature_pipeline(
    config: PipelineConfig,
) -> LocalPipelineExecutor:
    """Stage 2a -- compute MinHash signatures over filtered data."""
    filtered_path = f"{config.output_path}/filtered"

    pipeline = [
        ParquetReader(data_folder=filtered_path, text_key=config.text_key),
        MinhashDedupSignature(
            output_folder=f"{config.minhash_work_dir}/signatures",
            n_grams=config.minhash_ngram_size,
            num_buckets=config.minhash_num_buckets,
            hashes_per_bucket=config.minhash_hashes_per_bucket,
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=config.tasks,
        workers=config.workers,
        logging_dir=f"{config.logging_dir}/02a_minhash_sig",
    )


def build_minhash_buckets_pipeline(
    config: PipelineConfig,
) -> LocalPipelineExecutor:
    """Stage 2b -- bucket signatures for candidate-pair generation."""
    pipeline = [
        MinhashDedupBuckets(
            input_folder=f"{config.minhash_work_dir}/signatures",
            output_folder=f"{config.minhash_work_dir}/buckets",
            num_buckets=config.minhash_num_buckets,
            hashes_per_bucket=config.minhash_hashes_per_bucket,
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=config.minhash_num_buckets,
        workers=config.workers,
        logging_dir=f"{config.logging_dir}/02b_minhash_buckets",
    )


def build_minhash_cluster_pipeline(
    config: PipelineConfig,
) -> LocalPipelineExecutor:
    """Stage 2c -- cluster duplicate candidates via union-find."""
    pipeline = [
        MinhashDedupCluster(
            input_folder=f"{config.minhash_work_dir}/buckets",
            output_folder=f"{config.minhash_work_dir}/clusters",
            num_buckets=config.minhash_num_buckets,
            hashes_per_bucket=config.minhash_hashes_per_bucket,
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,  # clustering is single-task
        workers=1,
        logging_dir=f"{config.logging_dir}/02c_minhash_cluster",
    )


def build_minhash_filter_pipeline(
    config: PipelineConfig,
) -> LocalPipelineExecutor:
    """Stage 2d -- filter duplicates and write final clean Parquet."""
    filtered_path = f"{config.output_path}/filtered"
    final_output = f"{config.output_path}/deduped"

    pipeline = [
        ParquetReader(data_folder=filtered_path, text_key=config.text_key),
        MinhashDedupFilter(
            input_folder=f"{config.minhash_work_dir}/clusters",
            exclusion_writer=ParquetWriter(
                output_folder=f"{config.output_path}/duplicates",
            ),
        ),
        ParquetWriter(
            output_folder=final_output,
            max_file_size=5 * 2**30,
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=config.tasks,
        workers=config.workers,
        logging_dir=f"{config.logging_dir}/02d_minhash_filter",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(config: PipelineConfig) -> None:
    """Execute the complete pipeline: filter -> dedup (4-stage MinHash).

    Each stage is idempotent thanks to DataTrove's marker files.
    Re-running after a crash simply resumes from the last incomplete task.
    """
    stages: list[tuple[str, LocalPipelineExecutor]] = [
        ("01 Filter", build_filter_pipeline(config)),
        ("02a MinHash signatures", build_minhash_signature_pipeline(config)),
        ("02b MinHash buckets", build_minhash_buckets_pipeline(config)),
        ("02c MinHash clustering", build_minhash_cluster_pipeline(config)),
        ("02d MinHash filter", build_minhash_filter_pipeline(config)),
    ]

    for name, executor in stages:
        logger.info("Starting stage: %s", name)
        executor.run()
        logger.info("Completed stage: %s", name)

    logger.info(
        "Pipeline complete. Clean data at: %s/deduped", config.output_path
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Aurelius data processing pipeline (DataTrove)",
    )
    parser.add_argument(
        "--input-path",
        default=PipelineConfig.input_path,
        help="HuggingFace dataset path or local folder.",
    )
    parser.add_argument(
        "--output-path",
        default=PipelineConfig.output_path,
        help="Root output directory for processed data.",
    )
    parser.add_argument(
        "--minhash-work-dir",
        default=PipelineConfig.minhash_work_dir,
        help="Scratch directory for MinHash intermediate files.",
    )
    parser.add_argument(
        "--input-format",
        choices=("parquet", "jsonl"),
        default=PipelineConfig.input_format,
    )
    parser.add_argument("--tasks", type=int, default=PipelineConfig.tasks)
    parser.add_argument("--workers", type=int, default=PipelineConfig.workers)
    parser.add_argument(
        "--logging-dir",
        default=PipelineConfig.logging_dir,
    )
    parser.add_argument(
        "--language-threshold",
        type=float,
        default=PipelineConfig.language_threshold,
    )

    args = parser.parse_args(argv)
    return PipelineConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        minhash_work_dir=args.minhash_work_dir,
        input_format=args.input_format,
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=args.logging_dir,
        language_threshold=args.language_threshold,
    )


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    config = parse_args(argv)
    logger.info("Pipeline config: %s", config)
    run_full_pipeline(config)


if __name__ == "__main__":
    main()
