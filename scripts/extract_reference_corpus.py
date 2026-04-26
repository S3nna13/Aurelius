#!/usr/bin/env python3
"""
Extract all readable text from /Users/christienantonio/Desktop/Reference Models
into a single flat corpus file suitable for tokenizer training and LM pretraining.
Skips binaries, images, model weights, and known non-text formats.
"""

from pathlib import Path
import json

REF_DIR = Path("/Users/christienantonio/Desktop/Reference Models")
OUTPUT_DIR = Path("data/reference_corpus")
CORPUS_FILE = OUTPUT_DIR / "corpus.txt"
MANIFEST_FILE = OUTPUT_DIR / "manifest.jsonl"

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".rst", ".csv", ".html", ".htm", ".xml", ".sql", ".sh", ".bash", ".zsh",
    ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".kt", ".scala",
    ".rb", ".php", ".swift", ".m", ".mm", ".cs", ".fs", ".fsx",
    ".r", ".jl", ".lua", ".pl", ".pm", ".t", ".awk", ".sed",
    ".dockerfile", ".makefile", ".cmake", ".gradle", ".sbt",
    ".ini", ".cfg", ".conf", ".properties", ".env", ".toml",
    ".ipynb",  # JSON inside
}

SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".db",
    ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".onnx",
    ".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".ggml",
    ".o", ".a", ".lib", ".class", ".jar", ".war", ".ear",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".ico", ".icns",
    ".DS_Store",
}

SKIP_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv", "venv",
    "build", "dist", "target", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".terraform", ".idea", ".vscode", "*.egg-info",
}


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name.lower().startswith("dockerfile") or path.name.lower().startswith("makefile"):
        return True
    # Try reading first 4KB as text
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        chunk.decode("utf-8", errors="strict")
        return True
    except (UnicodeDecodeError, PermissionError, OSError):
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    total_files = 0
    skipped = 0

    with open(CORPUS_FILE, "w", encoding="utf-8") as out_f, \
         open(MANIFEST_FILE, "w", encoding="utf-8") as man_f:

        for path in REF_DIR.rglob("*"):
            if not path.is_file():
                continue
            if any(part.startswith(".") and part in SKIP_DIRS for part in path.parts):
                skipped += 1
                continue
            if not is_text_file(path):
                skipped += 1
                continue

            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if len(text.strip()) == 0:
                    continue
                # Add file path as header for context
                relative = path.relative_to(REF_DIR)
                out_f.write(f"\n{'='*60}\nFILE: {relative}\n{'='*60}\n\n")
                out_f.write(text)
                out_f.write("\n")

                total_chars += len(text)
                total_files += 1

                man_f.write(json.dumps({
                    "path": str(relative),
                    "chars": len(text),
                }) + "\n")

                if total_files % 1000 == 0:
                    print(f"Processed {total_files} files, {total_chars / 1e6:.1f} MB chars...")
            except Exception:
                skipped += 1
                continue

    print(f"\nDone!")
    print(f"Files extracted: {total_files}")
    print(f"Skipped:         {skipped}")
    print(f"Total chars:     {total_chars:,} ({total_chars / 1e6:.1f} MB)")
    print(f"Corpus saved:    {CORPUS_FILE}")
    print(f"Manifest saved:  {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
