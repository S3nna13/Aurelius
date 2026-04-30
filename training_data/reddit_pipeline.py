"""Reddit bulk download and preprocessing pipeline for Aurelius training data."""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

try:
    import zstandard as _zstd
except ImportError:
    _zstd = None


_PUSHSHIFT_BASE = "https://files.pushshift.io/reddit"
_ACADEMIC_TORRENT_BASE = "https://academictorrents.com"

_SUBREDDIT_CATEGORIES = {
    "technical": [
        "MachineLearning",
        "learnprogramming",
        "Python",
        "javascript",
        "golang",
        "rust",
        "compsci",
        "algorithms",
        "datascience",
        "typescript",
    ],
    "discussion": [
        "explainlikeimfive",
        "askscience",
        "askhistorians",
        "philosophy",
        "changemyview",
        "depthhub",
    ],
    "creative": [
        "WritingPrompts",
        "worldbuilding",
    ],
    "exclude": [
        "gonewild",
        "nsfw",
        "politic",
        "conspiracy",
    ],
}

_DEFAULT_CONFIG = {
    "subreddits": _SUBREDDIT_CATEGORIES,
    "min_score_submissions": 1,
    "min_score_comments": 2,
    "min_text_length": 50,
    "max_text_length": 10000,
    "include_conversations": True,
    "max_conversation_depth": 5,
    "state_file": "_state.json",
    "request_delay": 1.0,
}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r">\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def _is_bot(text: str) -> bool:
    lower = text.lower()
    bot_signals = [
        "i am a bot",
        "i'm a bot",
        "this is a bot",
        "beep boop",
        "i am an automated",
        "automated response",
        "^(bot",
    ]
    for signal in bot_signals:
        if signal in lower:
            return True
    return False


class RedditPipeline:
    """Production-grade Reddit data pipeline: download, parse, preprocess.

    Parameters
    ----------
    config : dict
        Configuration dictionary. See module-level ``_DEFAULT_CONFIG`` for
        the full schema.
    """

    def __init__(self, config: dict) -> None:
        self.config = dict(_DEFAULT_CONFIG)
        self.config.update(config)
        if "subreddits" in config:
            self.config["subreddits"] = {**_SUBREDDIT_CATEGORIES, **config["subreddits"]}

        self._allowlist: set[str] | None = None
        self._blocklist: set[str] = set()
        sub_cfg = self.config.get("subreddits", {})
        for cat in sub_cfg:
            if cat == "exclude":
                self._blocklist.update(sub_cfg[cat])
        self._blocklist = {s.lower() for s in self._blocklist}

    def _safe_name(self, name: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-")
        return cleaned or fallback

    def _resolve_within_root(
        self,
        root: str | Path,
        candidate: str | Path,
        *,
        must_exist: bool = False,
    ) -> Path:
        base = Path(root).expanduser().resolve()
        path = Path(candidate).expanduser()
        resolved = path.resolve(strict=must_exist) if path.is_absolute() else (base / path).resolve(strict=must_exist)
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Path escapes root: {candidate!r}") from exc
        return resolved

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------

    def _state_path(self, base_dir: str) -> Path:
        state_name = self._safe_name(self.config.get("state_file", "_state.json"), "_state.json")
        return Path(base_dir).expanduser().resolve() / state_name

    def _load_state(self, base_dir: str) -> dict[str, Any]:
        path = self._state_path(base_dir)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return {"processed_files": [], "downloaded_files": {}}
        return {"processed_files": [], "downloaded_files": {}}

    def _save_state(self, base_dir: str, state: dict[str, Any]) -> None:
        path = self._state_path(base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _mark_processed(self, base_dir: str, filepath: str, stats: dict[str, Any]) -> None:
        state = self._load_state(base_dir)
        state.setdefault("processed_files", [])
        state["processed_files"].append({"file": filepath, **stats})
        self._save_state(base_dir, state)

    def _is_processed(self, base_dir: str, filename: str) -> bool:
        state = self._load_state(base_dir)
        for entry in state.get("processed_files", []):
            if entry.get("file", "").endswith(filename):
                return True
        return False

    def _mark_downloaded(self, base_dir: str, url: str, local_path: str) -> None:
        state = self._load_state(base_dir)
        state.setdefault("downloaded_files", {})[url] = local_path
        self._save_state(base_dir, state)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _list_pushshift_files(self, file_types: list[str] | None) -> list[dict[str, str]]:
        """Scrape the Pushshift directory listing for monthly archives."""
        file_types = file_types or ["submissions", "comments"]
        files: list[dict[str, str]] = []
        for ftype in file_types:
            url = f"{_PUSHSHIFT_BASE}/{ftype}/"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Aurelius/1.0"})  # noqa: S310
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                    html = resp.read().decode("utf-8")
                pattern = re.compile(
                    r'href="([^"]*('
                    r"RS_\d{4}-\d{2}\.zst|"
                    r"RC_\d{4}-\d{2}\.zst"
                    r'))"'
                )
                for match in pattern.finditer(html):
                    fname = match.group(1)
                    files.append(
                        {
                            "name": fname,
                            "url": f"{url}{fname}",
                            "type": ftype,
                            "source": "pushshift",
                        }
                    )
            except Exception as exc:
                logger.warning("Failed to list Pushshift files for %s: %s", ftype, exc)
        files.sort(key=lambda f: f["name"])
        return files

    def download_bulk(
        self,
        target_dir: str,
        file_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Download Pushshift monthly files, returning list of downloaded paths."""
        target = Path(target_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        state = self._load_state(target_dir)
        downloaded_paths: list[str] = []

        available = self._list_pushshift_files(file_types)
        if limit:
            available = available[:limit]

        for entry in available:
            url = entry["url"]
            fname = self._safe_name(entry["name"], "reddit.zst")
            local_path = self._resolve_within_root(target, fname)

            existing = state.get("downloaded_files", {}).get(url)
            if existing:
                try:
                    existing_path = self._resolve_within_root(target, existing, must_exist=True)
                except (FileNotFoundError, ValueError):
                    existing_path = None
                if existing_path is not None and existing_path.exists():
                    logger.info("Already downloaded: %s", fname)
                    downloaded_paths.append(str(existing_path))
                    continue

            logger.info("Downloading %s ...", fname)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Aurelius/1.0"})  # noqa: S310
                with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    chunk_size = 65536
                    temp_path = local_path.with_name(f"{local_path.name}.tmp")
                    with open(temp_path, "wb") as f:
                        while True:
                            chunk = resp.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if _tqdm and total:
                                _tqdm.write(
                                    f"  {fname}: {downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MB"
                                )
                    shutil.move(str(temp_path), str(local_path))
                self._mark_downloaded(target_dir, url, str(local_path))
                downloaded_paths.append(str(local_path))
                if self.config.get("request_delay"):
                    time.sleep(self.config["request_delay"])
            except Exception as exc:
                logger.error("Failed to download %s: %s", url, exc)

        return downloaded_paths

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _open_compressed(
        self,
        filepath: str,
        *,
        input_root: str | Path | None = None,
    ) -> io.TextIOBase:
        """Open a potentially .zst or .gz file as a text stream."""
        if input_root is None:
            raise ValueError("input_root is required for file access")
        root = input_root
        path = self._resolve_within_root(root, filepath, must_exist=True)
        suffix = path.suffix
        if suffix == ".zst":
            if _zstd is None:
                raise ImportError("zstandard is required for .zst files; pip install zstandard")
            fh = open(path, "rb")
            dctx = _zstd.ZstdDecompressor()
            stream_reader = dctx.stream_reader(fh)
            return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
        elif suffix == ".gz":
            return gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            return open(path, encoding="utf-8", errors="replace")

    def process_file(
        self,
        filepath: str,
        output_dir: str,
        *,
        input_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Process a single Pushshift NDJSON file, writing raw JSONL lines."""
        fname = Path(filepath).name
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)

        safe_fname = self._safe_name(fname.replace(".zst", "").replace(".gz", ""), "reddit")
        raw_path = out / f"raw_{safe_fname}.jsonl"
        if raw_path.suffix != ".jsonl":
            raw_path = raw_path.with_suffix(".jsonl")

        total = 0
        kept = 0
        errors = 0

        with self._open_compressed(filepath, input_root=input_root) as stream, open(raw_path, "w") as out_f:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                processed = self._parse_item(item)
                if processed is None:
                    continue

                out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                kept += 1

        stats = {
            "file": fname,
            "total_items": total,
            "kept_items": kept,
            "parse_errors": errors,
            "output_path": str(raw_path),
        }
        self._mark_processed(
            str(out.parent),
            fname,
            {
                "total": total,
                "kept": kept,
                "errors": errors,
            },
        )
        return stats

    def _parse_item(self, item: dict) -> dict | None:
        """Parse a single Reddit item into our standard format. Returns None if filtered."""
        item_type = "submission" if "title" in item else "comment"

        subreddit = (item.get("subreddit") or "").strip().lower()
        if subreddit in self._blocklist:
            return None

        if item_type == "submission":
            title = (item.get("title") or "").strip()
            selftext = (item.get("selftext") or "").strip()
            text = f"{title}\n\n{selftext}" if selftext else title
            score = item.get("score", 0)
            created = item.get("created_utc", 0)
            item_id = str(item.get("id", ""))
            item.get("permalink", "")
        else:
            text = (item.get("body") or "").strip()
            score = item.get("score", 0)
            created = item.get("created_utc", 0)
            item_id = str(item.get("id", ""))
            item.get("permalink", "")

        text = _clean_text(text)

        min_score = (
            self.config.get("min_score_submissions", 1)
            if item_type == "submission"
            else self.config.get("min_score_comments", 2)
        )
        if score < min_score:
            return None

        min_len = self.config.get("min_text_length", 50)
        max_len = self.config.get("max_text_length", 10000)
        if len(text) < min_len or len(text) > max_len:
            return None

        lower_text = text.lower()
        if "[deleted]" in lower_text or "[removed]" in lower_text:
            return None

        if _is_bot(text):
            return None

        result: dict[str, Any] = {
            "id": item_id,
            "subreddit": subreddit,
            "title": title if item_type == "submission" else "",
            "text": text,
            "score": score,
            "created_utc": created,
            "type": item_type,
        }
        if item_type == "comment":
            result["parent_id"] = item.get("parent_id", "")
            result["link_id"] = item.get("link_id", "")

        return result

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def filter_by_quality(self, items: list[dict]) -> list[dict]:
        """Apply quality filters to a list of already-parsed items."""
        filtered: list[dict] = []
        for item in items:
            text = item.get("text", "")
            text = _clean_text(text)

            min_len = self.config.get("min_text_length", 50)
            max_len = self.config.get("max_text_length", 10000)
            if len(text) < min_len or len(text) > max_len:
                continue

            score = item.get("score", 0)
            is_submission = item.get("type") == "submission"
            min_score = (
                self.config.get("min_score_submissions", 1)
                if is_submission
                else self.config.get("min_score_comments", 2)
            )
            if score < min_score:
                continue

            lower_text = text.lower()
            if "[deleted]" in lower_text or "[removed]" in lower_text:
                continue

            if _is_bot(text):
                continue

            item["text"] = text
            filtered.append(item)

        return filtered

    # ------------------------------------------------------------------
    # Comment threading
    # ------------------------------------------------------------------

    def thread_comments(self, comments: list[dict]) -> list[dict]:
        """Thread flat comments into conversation trees.

        Returns a list of conversations, each a dict with an ``id``,
        ``subreddit``, ``conversations`` list, and ``score``.
        """
        depth_limit = self.config.get("max_conversation_depth", 5)

        comment_map: dict[str, dict] = {}
        for c in comments:
            cid = c.get("id", "")
            comment_map[cid] = c

        children_of: dict[str, list[dict]] = {}
        orphans: list[dict] = []
        for c in comments:
            raw_parent = c.get("parent_id", "")
            parent_id = raw_parent
            if raw_parent.startswith("t1_"):
                parent_id = raw_parent[3:]

            if parent_id in comment_map:
                children_of.setdefault(parent_id, []).append(c)
            else:
                orphans.append(c)

        conversations: list[dict] = []

        def _build_chain(comment: dict, depth: int = 0) -> list[dict[str, str]]:
            if depth >= depth_limit:
                return []
            cid = comment.get("id", "")
            chain = [{"from": "human", "value": comment.get("text", "")}]
            kids = children_of.get(cid, [])
            if kids:
                kids.sort(key=lambda k: k.get("created_utc", 0))
                for kid in kids:
                    reply_chain = _build_chain(kid, depth + 1)
                    if reply_chain:
                        chain.append({"from": "assistant", "value": kid.get("text", "")})
                        chain.extend(reply_chain[1:])
                    else:
                        chain.append({"from": "assistant", "value": kid.get("text", "")})
                    break
            return chain

        for comment in comments:
            parent_id_raw = comment.get("parent_id", "")
            parent_id = parent_id_raw
            if parent_id_raw.startswith("t1_"):
                parent_id = parent_id_raw[3:]

            if parent_id not in comment_map:
                chain = _build_chain(comment)
                if len(chain) >= 2:
                    conversations.append(
                        {
                            "id": comment.get("id", ""),
                            "subreddit": comment.get("subreddit", ""),
                            "conversations": chain,
                            "score": comment.get("score", 0),
                        }
                    )

        return conversations

    # ------------------------------------------------------------------
    # Output creation
    # ------------------------------------------------------------------

    def create_conversation_data(
        self,
        input_dir: str,
        output_path: str,
        max_examples: int | None = None,
    ) -> int:
        """Create threaded conversation JSONL from raw processed JSONL files."""
        in_dir = Path(input_dir)
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        all_items: list[dict] = []
        jsonl_files = sorted(in_dir.glob("raw_*.jsonl"))
        for jf in jsonl_files:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    all_items.append(item)

        comments = [i for i in all_items if i.get("type") == "comment"]
        conversations = self.thread_comments(comments)

        if max_examples and len(conversations) > max_examples:
            conversations = conversations[:max_examples]

        written = 0
        with open(out, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                written += 1

        logger.info("Wrote %d conversations to %s", written, output_path)
        return written

    def create_pretrain_data(self, input_dir: str, output_dir: str) -> int:
        """Create raw pretraining text from processed JSONL files."""
        in_dir = Path(input_dir)
        out_dir_p = Path(output_dir).expanduser().resolve()
        out_dir_p.mkdir(parents=True, exist_ok=True)

        total_written = 0
        jsonl_files = sorted(in_dir.glob("raw_*.jsonl"))
        shard_idx = 0
        shard_lines: list[str] = []
        shard_size = 10000

        for jf in jsonl_files:
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = item.get("text", "").strip()
                    if len(text) < self.config.get("min_text_length", 50):
                        continue

                    record = json.dumps(
                        {
                            "text": text,
                            "source": "reddit",
                            "subreddit": item.get("subreddit", ""),
                        },
                        ensure_ascii=False,
                    )
                    shard_lines.append(record)
                    total_written += 1

                    if len(shard_lines) >= shard_size:
                        shard_path = self._resolve_within_root(
                            out_dir_p, f"pretrain_reddit_{shard_idx:04d}.jsonl"
                        )
                        with open(shard_path, "w") as sf:
                            for sl in shard_lines:
                                sf.write(sl + "\n")
                        shard_lines = []
                        shard_idx += 1

        if shard_lines:
            shard_path = out_dir_p / f"pretrain_reddit_{shard_idx:04d}.jsonl"
            with open(shard_path, "w") as sf:
                for sl in shard_lines:
                    sf.write(sl + "\n")

        logger.info(
            "Wrote %d pretrain entries across %d shards",
            total_written,
            shard_idx + 1 if shard_lines or shard_idx else 0,
        )
        return total_written

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, output_dir: str, file_types: list[str] | None = None) -> dict[str, Any]:
        """Orchestrate full pipeline: download, process, and produce outputs."""
        base = Path(output_dir).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        raw_dir = base / "raw"
        raw_dir.mkdir(exist_ok=True)
        processed_dir = base / "processed"
        processed_dir.mkdir(exist_ok=True)
        pretrain_dir = base / "pretrain"
        pretrain_dir.mkdir(exist_ok=True)

        results: dict[str, Any] = {
            "downloaded": [],
            "processed": [],
            "conversations": 0,
            "pretrain_entries": 0,
        }

        downloaded = self.download_bulk(str(raw_dir), file_types=file_types)
        results["downloaded"] = [str(Path(p).name) for p in downloaded]

        for dl_path in downloaded:
            fname = Path(dl_path).name
            if self._is_processed(str(processed_dir), fname):
                logger.info("Skipping already-processed: %s", fname)
                continue
            stats = self.process_file(dl_path, str(processed_dir), input_root=str(raw_dir))
            results["processed"].append(stats)

        if self.config.get("include_conversations", True):
            conv_path = self._resolve_within_root(base, "conversations.jsonl")
            n_conv = self.create_conversation_data(str(processed_dir), str(conv_path))
            results["conversations"] = n_conv

        n_pretrain = self.create_pretrain_data(str(processed_dir), str(pretrain_dir))
        results["pretrain_entries"] = n_pretrain

        return results
