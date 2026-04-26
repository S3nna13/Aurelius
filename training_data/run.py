"""python -m training_data.run [--mode all|download|tokenize|synthetic|mix|status]
                               [--output ./data/aurelius-1.3b] [--samples N]
                               [--sources src1,src2] [--dry-run] [--config path]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("aurelius.run")


def _resolve_output(output: str) -> str:
    return str(Path(output).resolve())


def cmd_download(dm, output_dir: str, max_samples: int | None, sources: list[str] | None) -> None:
    logger.info("Download phase — output_dir=%s", output_dir)
    if sources:
        for src in sources:
            src_output = str(Path(output_dir) / "raw" / src)
            dm.run_source(src, src_output, max_samples=max_samples)
    else:
        dm.run_all(output_dir, max_samples={s: max_samples for s in dm.load_config().get("sources", {})} if max_samples else None)


def cmd_synthetic(output_dir: str, max_samples: int | None) -> None:
    logger.info("Synthetic data generation — output_dir=%s", output_dir)
    config = {"seed": 42}
    gen_dir = Path(output_dir) / "synthetic"

    try:
        from training_data.math_generator import MathDataGenerator
        gen = MathDataGenerator(config)
        gen.run(max_samples or 1000, str(gen_dir))
        logger.info("Math data generated")
    except Exception as e:
        logger.warning("Math generation skipped: %s", e)

    try:
        from training_data.code_generator import CodeDataGenerator
        gen = CodeDataGenerator(config)
        gen.run(max_samples or 1000, str(gen_dir))
        logger.info("Code data generated")
    except Exception as e:
        logger.warning("Code generation skipped: %s", e)

    try:
        from training_data.sft_generator import SFTDataGenerator
        gen = SFTDataGenerator(config)
        gen.run(max_samples or 1000, str(gen_dir))
        logger.info("SFT instruction data generated")
    except Exception as e:
        logger.warning("SFT generation skipped: %s", e)


def cmd_tokenize(output_dir: str, max_samples: int | None, config: dict) -> None:
    logger.info("Tokenization phase — output_dir=%s", output_dir)
    pipe_config = config if config else {}
    tokenize_cfg = config.get("tokenize", {}) if config else {}

    from training_data.tokenize_pipeline import TokenizePipeline

    pipeline = TokenizePipeline(pipe_config)
    shard_size = tokenize_cfg.get("shard_size", 16384)
    val_ratio = tokenize_cfg.get("val_ratio", 0.1)

    raw_dir = Path(output_dir) / "raw"
    shard_root = Path(output_dir) / "shards"
    train_shard_dir = shard_root / "train"
    val_shard_dir = shard_root / "val"

    all_shard_paths: list[str] = []

    for source_dir in raw_dir.iterdir():
        if not source_dir.is_dir():
            continue
        data_file = source_dir / "data.jsonl"
        if not data_file.exists():
            continue
        source_shard_dir = shard_root / "intermediate" / source_dir.name
        logger.info("Tokenizing %s", data_file)
        count = pipeline.tokenize_jsonl(
            str(data_file),
            str(source_shard_dir),
            text_field="text",
            shard_size=shard_size,
        )
        shards = sorted(Path(source_shard_dir).glob("shard_*.npy"))
        all_shard_paths.extend(str(s) for s in shards)
        logger.info("  -> %d texts, %d shards", count, len(shards))

    if all_shard_paths:
        train_paths, val_paths = pipeline.split_train_val(all_shard_paths, val_ratio=val_ratio)
        train_shard_dir.mkdir(parents=True, exist_ok=True)
        val_shard_dir.mkdir(parents=True, exist_ok=True)
        for i, sp in enumerate(train_paths):
            dest = train_shard_dir / f"shard_{i:06d}.npy"
            import shutil
            shutil.copy2(sp, dest)
        for i, sp in enumerate(val_paths):
            dest = val_shard_dir / f"shard_{i:06d}.npy"
            import shutil
            shutil.copy2(sp, dest)
        pipeline.create_shard_manifest(str(train_shard_dir), str(shard_root / "train_manifest.json"))
        pipeline.create_shard_manifest(str(val_shard_dir), str(shard_root / "val_manifest.json"))
        logger.info("Train shards: %d, Val shards: %d", len(train_paths), len(val_paths))


def cmd_mix(output_dir: str, config: dict) -> None:
    logger.info("Mixing phase — output_dir=%s", output_dir)
    data_mix = config.get("data_mix", {})
    if not data_mix:
        logger.warning("No data_mix config found, skipping")
        return

    pipe_config = config if config else {}
    from training_data.tokenize_pipeline import TokenizePipeline

    pipeline = TokenizePipeline(pipe_config)
    shard_root = Path(output_dir) / "shards"
    mixed_output = shard_root / "mixed"

    shard_dirs: list[str] = []
    weights: list[float] = []
    for source, weight in data_mix.items():
        source_shard_dir = shard_root / "intermediate" / source
        if source_shard_dir.exists() and list(source_shard_dir.glob("shard_*.npy")):
            shard_dirs.append(str(source_shard_dir))
            weights.append(weight)

    if shard_dirs:
        total_tokens = pipeline.interleave_datasets(shard_dirs, str(mixed_output), weights)
        logger.info("Mixed dataset: %d shards, %d tokens", len(list(mixed_output.glob("shard_*.npy"))), total_tokens)
    else:
        logger.warning("No sharded datasets found to mix")


def cmd_status(dm, output_dir: str) -> None:
    logger.info("Status report")
    print("=== Data Status ===")
    print(f"Output dir: {output_dir}")
    status = dm.status()
    for source, stat in status.items():
        print(f"  {source}: {stat}")
    print()
    manifest = dm.get_manifest()
    print("Data mix weights:")
    for source, weight in manifest.get("data_mix", {}).items():
        print(f"  {source}: {weight:.2%}")
    print()
    raw_dir = Path(output_dir) / "raw"
    shard_dir = Path(output_dir) / "shards"
    if raw_dir.exists():
        sizes = []
        for f in raw_dir.rglob("*"):
            if f.is_file():
                sizes.append(f.stat().st_size)
        print(f"Raw data size: {sum(sizes) / (1024**3):.2f} GB")
    if shard_dir.exists():
        shards = list(shard_dir.rglob("shard_*.npy"))
        print(f"Shard files: {len(shards)}")
        if shards:
            total_tokens = 0
            import numpy as np
            for s in shards:
                arr = np.load(str(s))
                total_tokens += arr.size
            print(f"Total tokens (approximate): {total_tokens:,}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aurelius training data pipeline")
    parser.add_argument("--mode", type=str, default="status",
                        choices=["all", "download", "tokenize", "synthetic", "mix", "status"],
                        help="Pipeline mode")
    parser.add_argument("--output", type=str, default="./data/aurelius-1.3b",
                        help="Output directory")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max samples per source (for testing)")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma-separated subset of sources")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done")
    parser.add_argument("--config", type=str, default="training_data/config.yaml",
                        help="Path to config file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = _resolve_output(args.output)

    if args.dry_run:
        print(f"Dry-run: mode={args.mode}, output={output_dir}, samples={args.samples}, sources={args.sources}")
        return 0

    config: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            import yaml
            raw = config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(raw) or {}
        except Exception as e:
            logger.warning("Could not load config %s: %s", args.config, e)

    from training_data.download_manager import DownloadManager
    dm = DownloadManager(args.config)

    source_list = args.sources.split(",") if args.sources else None

    if args.mode == "all":
        cmd_download(dm, output_dir, args.samples, source_list)
        cmd_synthetic(output_dir, args.samples)
        cmd_tokenize(output_dir, args.samples, config)
        cmd_mix(output_dir, config)
    elif args.mode == "download":
        cmd_download(dm, output_dir, args.samples, source_list)
    elif args.mode == "synthetic":
        cmd_synthetic(output_dir, args.samples)
    elif args.mode == "tokenize":
        cmd_tokenize(output_dir, args.samples, config)
    elif args.mode == "mix":
        cmd_mix(output_dir, config)
    elif args.mode == "status":
        cmd_status(dm, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
