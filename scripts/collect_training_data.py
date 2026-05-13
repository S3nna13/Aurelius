from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

SYSTEM = "You are Aurelius, a helpful AI assistant."
def msg(r, c):
    return {"role": r, "content": c}


def add(msgs):
    return {"messages": [msg("system", SYSTEM)] + msgs}


# (hf_id, config, type, title_field, text_field)
SOURCES = [
    ("ccdv/arxiv-classification", None, "arxiv", "text", "text"),
    ("CShorten/ML-ArXiv-Papers", None, "arxiv", "title", "abstract"),
    ("nreimers/reddit_question_best_answers", None, "reddit", "title", "answers"),
    ("DDSC/reddit-da", None, "text", None, "text"),
    ("wikitext", "wikitext-103-v1", "text", None, "text"),
    ("pubmed_qa", "pqa_labeled", "qa", "question", "long_answer"),
]


def collect(src, tmp, limit=2000):
    from datasets import load_dataset

    hf_id, cfg, kind, title_f, text_f = src
    out = tmp / f"{hf_id.replace('/', '_')}.jsonl"
    count = 0
    kwa = {"split": "train", "streaming": True}
    if cfg:
        kwa["name"] = cfg
    hf_revision = os.environ.get("AURELIUS_HF_REVISION")
    if not hf_revision:
        raise ValueError("AURELIUS_HF_REVISION is required to pin Hugging Face dataset revisions")
    ds = load_dataset(hf_id, revision=hf_revision, **kwa)
    with open(out, "w") as f:
        for i, ex in enumerate(ds):
            if count >= limit:
                break
            try:
                if kind == "arxiv":
                    t = (ex.get("text") or ex.get("title") or "").strip()
                    a = (ex.get("abstract") or "").strip() or t[:2000]
                    if not a or len(a) < 50:
                        continue
                    title = t[:200] if t else "paper"
                    f.write(
                        json.dumps(
                            add([msg("user", f"Summarize: {title}"), msg("assistant", a[:2000])])
                        )
                        + "\n"
                    )
                    count += 1
                elif kind == "reddit":
                    title = (ex.get("title") or "").strip()
                    if not title or len(title) < 15:
                        continue
                    ans = ex.get("answers") or []
                    if not ans:
                        continue
                    a = (ans[0].get("body") if isinstance(ans[0], dict) else ans[0]) or ""
                    if len(a) < 50:
                        continue
                    f.write(
                        json.dumps(add([msg("user", title), msg("assistant", str(a)[:2500])]))
                        + "\n"
                    )
                    count += 1
                elif kind == "text":
                    t = (ex.get("text") or "").strip()
                    if len(t) < 300:
                        continue
                    f.write(
                        json.dumps(
                            add([msg("user", "Tell me about this."), msg("assistant", t[:2500])])
                        )
                        + "\n"
                    )
                    count += 1
                elif kind == "qa":
                    q = (ex.get(title_f) or "").strip() if title_f else ""
                    a = (ex.get(text_f) or "").strip() if text_f else ""
                    if not q:
                        q = (ex.get("question") or "").strip()
                    if not a:
                        a = (ex.get("long_answer") or ex.get("answer") or "").strip()
                    if not q or not a or len(a) < 50:
                        continue
                    f.write(
                        json.dumps(add([msg("user", q[:1000]), msg("assistant", a[:2500])])) + "\n"
                    )
                    count += 1
            except Exception:  # noqa: S110
                pass
            if (i + 1) % 500 == 0:
                print(f"  {hf_id.split('/')[-1]}: {i + 1}...", flush=True)
    print(f"  {hf_id.split('/')[-1]}: +{count}", flush=True)
    return out if count > 0 else None

def run_collection_loop(root: Path, rust: Path, desktop: Path) -> None:
    for r in range(1000):
        random.shuffle(SOURCES)
        src = SOURCES[0]
        print(f"Round {r + 1} — {src[0].split('/')[-1]}", flush=True)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            out = collect(src, tmp)
            if out is None or not out.exists():
                print("  no data, skip", flush=True)
                continue
            merged = tmp / "m.jsonl"
            subprocess.run([str(rust), str(merged), str(desktop), str(out)], capture_output=True)
            shutil.copy2(merged, desktop)
            lines = sum(1 for _ in open(desktop))
            print(f"  Desktop: {lines} ex, {desktop.stat().st_size / 1024 / 1024:.1f} MB", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect & convert training data from HuggingFace datasets")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Aurelius project root")
    parser.add_argument("--output", type=Path, default=Path.home() / "Desktop/Aurelius_Training_Data.jsonl", help="Final merged JSONL output")
    args = parser.parse_args()

    ROOT = args.root.expanduser().resolve()
    RUST = ROOT / "tools/jsonl_merge/target/release/jsonl-merge"
    DESKTOP = args.output.expanduser().resolve()
    run_collection_loop(ROOT, RUST, DESKTOP)


if __name__ == "__main__":
    main()
