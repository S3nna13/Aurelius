"""Aurelius training data pipeline — arXiv papers + Reddit Q&A.
Downloads, processes, and converts to chat JSONL format.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are Aurelius, a helpful AI assistant trained on scientific literature and online discussions."


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _conversation(messages: list[dict], source: str = "") -> dict:
    return {"messages": [_msg("system", SYSTEM_PROMPT)] + messages, "source": source}


def _hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _write_jsonl(examples: list[dict], path: Path, append: bool = False) -> None:
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(examples)} examples → {path}")


def _deduplicate(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for ex in examples:
        content = json.dumps(ex["messages"], sort_keys=True)
        h = _hash(content)
        if h not in seen:
            seen.add(h)
            deduped.append(ex)
    return deduped


# ─── arXiv ────────────────────────────────────────────────────────────

def process_arxiv(output: Path, max_papers: int = 5000) -> int:
    """Download arXiv papers from HF and convert to chat format."""
    logger.info("Loading arXiv classification dataset...")
    from datasets import load_dataset

    ds = load_dataset("ccdv/arxiv-classification", split="train", streaming=True)
    examples: list[dict] = []
    count = 0
    for i, paper in enumerate(ds):
        if i >= max_papers:
            break
        try:
            text = paper.get("text", "").strip()
            if not text:
                continue
            # Split into title and abstract using common patterns
            lines = text.split("\n")
            title = lines[0].strip() if lines else ""
            abstract = " ".join(l.strip() for l in lines[1:] if l.strip())[:2000] if len(lines) > 1 else text[:500]
            if not title:
                title = f"ArXiv Paper {_hash(text[:100])}"
            examples.append(_conversation([
                _msg("user", f"Summarize the following paper:\n\nTitle: {title}"),
                _msg("assistant", abstract),
            ], source="arxiv"))
            count += 1
        except Exception as e:
            logger.warning(f"Error processing paper {i}: {e}")
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1} papers...")

    examples = _deduplicate(examples)
    _write_jsonl(examples, output)
    logger.info(f"arXiv: {len(examples)} examples from {count} papers processed")
    return len(examples)


# ─── Reddit ────────────────────────────────────────────────────────────

def process_reddit_qa(output: Path, max_posts: int = 5000) -> int:
    """Download Reddit Q&A from HF and convert to chat format."""
    logger.info("Loading Reddit Q&A dataset...")
    from datasets import load_dataset

    ds = load_dataset("nreimers/reddit_question_best_answers", split="train", streaming=True)
    examples: list[dict] = []
    count = 0
    for i, post in enumerate(ds):
        if i >= max_posts:
            break
        try:
            title = post.get("title", "").strip()
            body = post.get("body", "").strip()
            answers = post.get("answers", [])
            question = title + (" " + body[:500] if body else "")
            if not question or not answers:
                continue
            best = answers[0] if isinstance(answers, list) else answers
            answer_text = best.get("body", "") if isinstance(best, dict) else str(best)
            if len(question) < 20 or len(answer_text) < 50:
                continue
            examples.append(_conversation([
                _msg("user", question),
                _msg("assistant", answer_text[:3000]),
            ], source="reddit/qa"))
            count += 1
        except Exception as e:
            logger.warning(f"Error processing post {i}: {e}")
        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i+1} posts...")

    examples = _deduplicate(examples)
    _write_jsonl(examples, output, append=output.exists())
    logger.info(f"Reddit: {len(examples)} examples from {count} posts processed")
    return len(examples)


# ─── ArXiv ML Papers ─────────────────────────────────────────────────

def process_arxiv_ml(output: Path, max_papers: int = 3000) -> int:
    """ML-focused arXiv papers from CShorten dataset."""
    logger.info("Loading ML ArXiv papers...")
    from datasets import load_dataset

    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train", streaming=True)
    examples: list[dict] = []
    count = 0
    for i, paper in enumerate(ds):
        if i >= max_papers:
            break
        try:
            title = paper.get("title", "").strip()
            abstract = paper.get("abstract", "").strip()
            if not title or not abstract:
                continue
            examples.append(_conversation([
                _msg("user", f"Summarize the ML paper '{title}'"),
                _msg("assistant", abstract[:2000]),
            ], source="arxiv/ml"))
            count += 1
        except Exception as e:
            logger.warning(f"Error processing {i}: {e}")
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1} papers...")

    examples = _deduplicate(examples)
    _write_jsonl(examples, output, append=output.exists())
    logger.info(f"ML ArXiv: {len(examples)} examples")
    return len(examples)


# ─── Reddit Science Q&A ──────────────────────────────────────────────

def process_reddit_science(output: Path, max_posts: int = 2000) -> int:
    """Reddit science discussion — try multiple datasets."""
    logger.info("Loading Reddit discussions...")
    from datasets import load_dataset

    datasets_to_try = [
        ("DDSC/reddit-da", "text"),
    ]
    examples: list[dict] = []
    for ds_name, text_field in datasets_to_try:
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            count = 0
            for i, post in enumerate(ds):
                if i >= max_posts:
                    break
                try:
                    val = (post.get(text_field, "") or "")
                    if not isinstance(val, str) or len(val) < 150:
                        continue
                    truncated = val[:2500]
                    examples.append(_conversation([
                        _msg("user", "Tell me something interesting."),
                        _msg("assistant", truncated),
                    ], source=f"reddit/{ds_name}"))
                    count += 1
                except Exception:
                    continue
                if (i + 1) % 500 == 0:
                    logger.info(f"  {ds_name}: processed {i+1}...")
            logger.info(f"  {ds_name}: {count} examples")
        except Exception as e:
            logger.warning(f"  {ds_name}: skipped ({type(e).__name__}: {e})")

    examples = _deduplicate(examples)
    _write_jsonl(examples, output, append=output.exists())
    logger.info(f"Reddit discussions: {len(examples)} examples")
    return len(examples)


# ─── Main Pipeline ────────────────────────────────────────────────────

def main():
    data_dir = Path("training_data")
    data_dir.mkdir(exist_ok=True)
    output = data_dir / "arxiv_reddit_training_data.jsonl"

    logger.info("═" * 40)
    logger.info("Aurelius Training Data Pipeline")
    logger.info("═" * 40)

    total = 0
    with open(output, "w") as f:
        f.write("")  # clear file

    total += process_arxiv(output, max_papers=5000)
    total += process_reddit_qa(output, max_posts=5000)
    total += process_arxiv_ml(output, max_papers=3000)
    total += process_reddit_science(output, max_posts=3000)

    # Final dedup
    logger.info("═" * 40)
    logger.info("Final deduplication...")
    with open(output) as f:
        all_examples = [json.loads(line) for line in f if line.strip()]
    deduped = _deduplicate(all_examples)
    random.shuffle(deduped)
    _write_jsonl(deduped, output)

    total_chars = sum(len(json.dumps(ex)) for ex in deduped)
    logger.info(f"Total: {len(deduped)} unique examples → {output}")
    logger.info(f"  Size: {total_chars:,} chars (~{total_chars // 4:,} tokens)")
    logger.info(f"  Sources: arxiv, arxiv/summarization, reddit/r/*, reddit/r/tifu")

    # Count sources
    from collections import Counter
    sources = Counter(ex.get("source", "unknown") for ex in deduped)
    logger.info(f"  Source breakdown: {dict(sources.most_common(10))}")


if __name__ == "__main__":
    main()
