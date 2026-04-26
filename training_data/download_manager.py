from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_DEFAULTS = {
    "data_mix": {
        "arxiv": 0.35,
        "reddit": 0.25,
        "synthetic_code": 0.15,
        "synthetic_instructions": 0.10,
        "synthetic_math": 0.08,
        "synthetic_safety": 0.05,
        "synthetic_agent": 0.02,
    },
    "tokenize": {
        "shard_size": 16384,
        "val_ratio": 0.1,
        "max_length": 8192,
        "num_workers": 4,
        "verify_integrity": True,
    },
    "sources": {
        "arxiv": {"enabled": True, "url": "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/arxiv/arxiv_0.jsonl"},
        "reddit": {"enabled": True, "url": "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T/resolve/main/reddit/reddit_0.jsonl"},
    },
}


class DownloadManager:
    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        self._config: dict[str, Any] | None = None
        self._status: dict[str, str] = {}

    def load_config(self) -> dict[str, Any]:
        if self._config is not None:
            return self._config
        config_path = self.config_path
        try:
            if config_path.exists():
                raw = config_path.read_text(encoding="utf-8")
                cfg: dict = yaml.safe_load(raw) or {}
            else:
                cfg = {}
        except Exception:
            cfg = {}
        for key, val in _CONFIG_DEFAULTS.items():
            cfg.setdefault(key, val)
        cfg.setdefault("sources", _CONFIG_DEFAULTS["sources"])
        cfg.setdefault("data_mix", _CONFIG_DEFAULTS["data_mix"])
        cfg.setdefault("tokenize", _CONFIG_DEFAULTS["tokenize"])
        self._config = cfg
        return cfg

    def run_all(self, output_dir: str, max_samples: dict[str, int] | None = None) -> dict[str, Any]:
        config = self.load_config()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        sources = config.get("sources", _CONFIG_DEFAULTS["sources"])
        max_samples = max_samples or {}

        agg_stats: dict[str, Any] = {
            "start_time": datetime.utcnow().isoformat(),
            "sources": [],
            "total_samples": 0,
        }

        for source_name, source_cfg in sources.items():
            if not source_cfg.get("enabled", True):
                continue
            try:
                result = self.run_source(
                    source_name,
                    str(output_path / source_name),
                    max_samples=max_samples.get(source_name),
                )
                agg_stats["sources"].append(result)
                agg_stats["total_samples"] += result.get("samples", 0)
                self._status[source_name] = "completed"
            except Exception as exc:
                logger.error("Source %s failed: %s", source_name, exc)
                agg_stats["sources"].append({"source": source_name, "status": "failed", "error": str(exc)})
                self._status[source_name] = "failed"

        agg_stats["end_time"] = datetime.utcnow().isoformat()
        return agg_stats

    def run_source(self, source: str, output_dir: str, **kwargs: Any) -> dict[str, Any]:
        config = self.load_config()
        sources = config.get("sources", {})
        source_cfg = sources.get(source, {})
        if not source_cfg:
            raise ValueError(f"Unknown source: {source}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        max_samples = kwargs.get("max_samples")
        url = source_cfg.get("url", "")

        result: dict[str, Any] = {
            "source": source,
            "output_dir": output_dir,
            "status": "pending",
            "samples": 0,
        }

        if source == "synthetic_code":
            from training_data.code_generator import CodeDataGenerator
            gen = CodeDataGenerator(config)
            gen.run(max_samples or 1000, str(output_path.parent))
            result["status"] = "completed"
            result["samples"] = max_samples or 1000
        elif source == "synthetic_math":
            from training_data.math_generator import MathDataGenerator
            gen = MathDataGenerator(config)
            gen.run(max_samples or 1000, str(output_path.parent))
            result["status"] = "completed"
            result["samples"] = max_samples or 1000
        elif source == "synthetic_instructions":
            from training_data.sft_generator import SFTDataGenerator
            gen = SFTDataGenerator(config)
            gen.run(max_samples or 1000, str(output_path.parent))
            result["status"] = "completed"
            result["samples"] = max_samples or 1000
        elif source in ("arxiv", "reddit"):
            count = self._download_jsonl(url, output_path, max_samples)
            result["status"] = "completed"
            result["samples"] = count
        else:
            result["status"] = "pending"
            logger.info("Source %s has no download handler, marking pending", source)

        return result

    def _download_jsonl(self, url: str, output_dir: Path, max_samples: int | None) -> int:
        import urllib.request

        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / "data.jsonl"
        temp_file = output_dir / ".data.jsonl.part"

        count = 0
        try:
            if url.startswith("http"):
                logger.info("Downloading %s to %s", url, output_file)
                urllib.request.urlretrieve(url, str(temp_file))
                os.rename(str(temp_file), str(output_file))
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            else:
                logger.info("No URL for %s, generating placeholder", output_dir)
                with open(output_file, "w", encoding="utf-8") as f:
                    for i in range(max_samples or 100):
                        f.write(json.dumps({"text": f"Sample document {i} from {output_dir.name}"}) + "\n")
                        count += 1
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise

        return count

    def status(self) -> dict[str, str]:
        if not self._status:
            config = self.load_config()
            sources = config.get("sources", {})
            for source_name in sources:
                self._status.setdefault(source_name, "pending")
        return dict(self._status)

    def get_manifest(self) -> dict[str, Any]:
        config = self.load_config()
        manifest: dict[str, Any] = {
            "created": datetime.utcnow().isoformat(),
            "sources": {},
            "data_mix": config.get("data_mix", _CONFIG_DEFAULTS["data_mix"]),
        }
        sources = config.get("sources", _CONFIG_DEFAULTS["sources"])
        for source_name, source_cfg in sources.items():
            manifest["sources"][source_name] = {
                "enabled": source_cfg.get("enabled", True),
                "status": self._status.get(source_name, "pending"),
            }
        return manifest

    def cleanup(self) -> None:
        config = self.load_config()
        sources = config.get("sources", {})
        for source_name in sources:
            partial_dir = Path(f".data/{source_name}.part")
            if partial_dir.exists():
                shutil.rmtree(partial_dir)
                logger.info("Cleaned up partial download %s", partial_dir)
        logger.info("Cleanup complete")
