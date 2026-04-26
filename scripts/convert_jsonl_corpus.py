#!/usr/bin/env python3
"""Convert Aurelius_Training_Data.jsonl (chat format) to flat text corpus for LM training."""
import json
import sys
from pathlib import Path

def main():
    jsonl_path = Path("/Users/christienantonio/Desktop/Aurelius_Training_Data.jsonl")
    out_path = Path("/Users/christienantonio/Desktop/Aurelius/data/reference_corpus/jsonl_corpus.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    total_chars = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            if not messages:
                continue

            parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    parts.append(f"### {role.capitalize()}\n{content}")

            if parts:
                text = "\n\n".join(parts) + "\n\n---\n\n"
                fout.write(text)
                total_lines += 1
                total_chars += len(text)

    print(f"Converted {total_lines:,} conversations -> {out_path}")
    print(f"Total text: {total_chars:,} chars ({total_chars/1e6:.1f}MB)")

if __name__ == "__main__":
    main()
