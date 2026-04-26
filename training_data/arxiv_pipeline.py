"""arXiv bulk download and preprocessing pipeline for Aurelius training."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import unidecode
except ImportError:
    unidecode = None

try:
    from langdetect import detect as _lang_detect
except ImportError:
    _lang_detect = None

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "arxiv": {
        "categories": [
            "cs.AI", "cs.LG", "cs.CL", "cs.SE", "cs.PL",
            "stat.ML", "math.NA", "physics", "q-bio", "q-fin",
        ],
        "max_papers_per_category": None,
        "include_abstracts": True,
        "include_full_text": True,
        "include_code_blocks": True,
        "min_text_length": 1000,
        "max_text_length": 100000,
        "download_dir": "./data/arxiv/sources",
        "output_dir": "./data/arxiv/processed",
        "s3_bucket": "arxiv",
        "s3_prefix": "src/",
        "concurrency": 4,
    }
}

ARXIV_API_BASE = "http://export.arxiv.org/api/query"
ARXIV_S3_BASE = "https://arxiv.org/src/"

SECTION_HEADING_RE = re.compile(
    r'\\(?:section|subsection|subsubsection|paragraph|textbf|textit)\*?\s*\{([^}]+)\}',
    re.IGNORECASE,
)
LATEX_COMMAND_RE = re.compile(r'\\(?:[a-zA-Z]+|.)(?:\s*\{[^}]*\})?')
MATH_INLINE_RE = re.compile(r'\$[^$]+\$')
MATH_DISPLAY_RE = re.compile(r'\$\$[^$]+\$\$')
VERBATIM_RE = re.compile(r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}', re.DOTALL)
LISTINGS_RE = re.compile(r'\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}', re.DOTALL)
CITE_RE = re.compile(r'\\(?:cite|citet|citep)\s*(?:\[[^\]]*\])*\s*\{[^}]+\}')
REF_RE = re.compile(r'\\(?:ref|label|pageref)\s*\{[^}]+\}')
BIB_RE = re.compile(r'\\bibliography\s*\{[^}]+\}')
INCLUDE_RE = re.compile(r'\\(?:include|input)\s*\{[^}]+\}')
COMMENT_RE = re.compile(r'(?<!\\)%.*$', re.MULTILINE)
MULTI_LINE_COMMENT_RE = re.compile(
    r'\\begin\{comment\}.*?\\end\{comment\}', re.DOTALL
)
LATEX_ENV_RE = re.compile(r'\\(?:begin|end)\s*\{[a-zA-Z*]+\}')
LATEX_ESCAPED_CHARS_RE = re.compile(r'\\([{}_&^~#$])')

ARXIV_ID_RE = re.compile(
    r'(\d{4}\.\d{4,5}(?:v\d+)?'
    r'|[a-z\-]+(?:\.[a-z\-]+)?/\d{7}(?:v\d+)?)'
)


def _import_tqdm():
    return _tqdm


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    abstract: str
    year: int
    source_format: str = ""


@dataclass
class ParsedPaper:
    metadata: PaperMetadata
    sections: dict[str, str] = field(default_factory=dict)
    full_text: str = ""
    code_blocks: list[str] = field(default_factory=list)
    quality_score: float = 1.0


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------

def extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_ID_RE.search(text.strip())
    return m.group(1) if m else None


def strip_latex_commands(text: str) -> str:
    text = MATH_DISPLAY_RE.sub(" ", text)
    text = MATH_INLINE_RE.sub(" ", text)
    text = MULTI_LINE_COMMENT_RE.sub(" ", text)
    text = COMMENT_RE.sub("", text)
    text = LATEX_ESCAPED_CHARS_RE.sub(r"\1", text)
    text = LATEX_COMMAND_RE.sub(" ", text)
    text = LATEX_ENV_RE.sub(" ", text)
    text = CITE_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = BIB_RE.sub(" ", text)
    text = INCLUDE_RE.sub(" ", text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def strip_html_tags(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def detect_language(text: str) -> str:
    if _lang_detect is not None:
        try:
            return _lang_detect(text[:1000])
        except Exception:
            pass
    ascii_chars = sum(1 for c in text[:2000] if ord(c) < 128)
    ratio = ascii_chars / max(len(text[:2000]), 1)
    return "en" if ratio > 0.8 else "unknown"


def compute_math_ratio(text: str) -> float:
    if not text:
        return 0.0
    math_chars = sum(1 for c in text if c in '$\\{}[]_^=<>∑∫∏∂∇∃∀∈∉⊂⊃∪∩∧∨¬⇒⇔λμπσ')
    return math_chars / len(text)


def compute_quality_score(text: str) -> float:
    score = 1.0
    if len(text) < 1000:
        score -= 0.3
    math_ratio = compute_math_ratio(text)
    if math_ratio > 0.5:
        score -= 0.3
    if text.count("\\") > len(text) * 0.05:
        score -= 0.2
    return max(0.0, score)


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    lines = text.split("\n")
    current_heading = "abstract"
    current_lines: list[str] = []

    for line in lines:
        m = SECTION_HEADING_RE.search(line)
        if m:
            if current_lines:
                body = " ".join(current_lines).strip()
                if body:
                    sections[current_heading] = body
            current_heading = m.group(1).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        body = " ".join(current_lines).strip()
        if body:
            sections[current_heading] = body

    return sections


# ---------------------------------------------------------------------------
# ArxivPipeline
# ---------------------------------------------------------------------------

class ArxivPipeline:
    """Production-grade arXiv bulk download & preprocessing pipeline.

    Parameters
    ----------
    config : dict
        Configuration dictionary.  See module-level ``DEFAULT_CONFIG``
        for the full schema.
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("arxiv", config)
        cfg = {**DEFAULT_CONFIG["arxiv"], **cfg}
        self.config = cfg
        self._categories = cfg["categories"]
        self._max_per_cat = cfg["max_papers_per_category"]
        self._include_abstracts = cfg["include_abstracts"]
        self._include_full_text = cfg["include_full_text"]
        self._include_code_blocks = cfg["include_code_blocks"]
        self._min_text_length = cfg["min_text_length"]
        self._max_text_length = cfg["max_text_length"]
        self._download_dir = Path(cfg["download_dir"])
        self._output_dir = Path(cfg["output_dir"])
        self._s3_bucket = cfg["s3_bucket"]
        self._s3_prefix = cfg["s3_prefix"]
        self._concurrency = cfg["concurrency"]

        self._session: requests.Session | None = None
        if requests is not None:
            self._session = requests.Session()

    # -- helpers ---------------------------------------------------------

    def _get_session(self) -> requests.Session | None:
        if self._session is None and requests is not None:
            self._session = requests.Session()
        return self._session

    def _state_path(self, base_dir: Path) -> Path:
        return base_dir / "_state.json"

    def _load_state(self, base_dir: Path) -> dict[str, Any]:
        sp = self._state_path(base_dir)
        if sp.exists():
            try:
                return json.loads(sp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"processed_ids": [], "download_log": [], "last_run": None}

    def _save_state(self, base_dir: Path, state: dict) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        sp = self._state_path(base_dir)
        sp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _update_state(self, base_dir: Path, arxiv_id: str) -> None:
        state = self._load_state(base_dir)
        if arxiv_id not in state["processed_ids"]:
            state["processed_ids"].append(arxiv_id)
        state["last_run"] = datetime.now(timezone.utc).isoformat()
        self._save_state(base_dir, state)

    def _is_processed(self, base_dir: Path, arxiv_id: str) -> bool:
        state = self._load_state(base_dir)
        return arxiv_id in state["processed_ids"]

    def _fetch_url(self, url: str, timeout: int = 30) -> bytes | None:
        sess = self._get_session()
        if sess is None:
            logger.warning("requests not available; cannot fetch %s", url)
            return None
        for attempt in range(3):
            try:
                resp = sess.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as exc:
                logger.warning("Attempt %d/%d failed for %s: %s", attempt + 1, 3, url, exc)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def _download_s3(self, arxiv_id: str, target_dir: Path) -> Path | None:
        """Download a single paper from arXiv's S3 bucket."""
        url = f"{ARXIV_S3_BASE}{arxiv_id}"
        data = self._fetch_url(url)
        if data is None:
            return None
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / f"{arxiv_id}.tar.gz"
        try:
            dest.write_bytes(data)
        except OSError as exc:
            logger.error("Failed to write %s: %s", dest, exc)
            return None
        return dest

    def _download_api(
        self,
        category: str,
        max_results: int = 100,
        start: int = 0,
    ) -> list[dict[str, Any]]:
        """Query the arXiv API for papers in a category."""
        sess = self._get_session()
        if sess is None:
            return []
        params: dict[str, Any] = {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": min(max_results, 1000),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            resp = sess.get(ARXIV_API_BASE, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("API query failed for %s: %s", category, exc)
            return []
        return self._parse_atom_response(resp.text)

    @staticmethod
    def _parse_atom_response(xml_text: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for entry in re.finditer(
            r'<entry>(.*?)</entry>', xml_text, re.DOTALL
        ):
            entry_xml = entry.group(1)
            id_m = re.search(
                r'<id>\s*https?://arxiv\.org/abs/(\S+?)\s*</id>', entry_xml
            )
            title_m = re.search(r'<title>(.*?)</title>', entry_xml, re.DOTALL)
            summary_m = re.search(
                r'<summary>(.*?)</summary>', entry_xml, re.DOTALL
            )
            published_m = re.search(
                r'<published>(\d{4})', entry_xml
            )
            authors = re.findall(
                r'<name>(.*?)</name>', entry_xml
            )
            categories = re.findall(
                r'<category\s+term="([^"]+)"', entry_xml
            )

            results.append({
                "id": id_m.group(1).strip() if id_m else "",
                "title": _clean_atom_text(title_m.group(1) if title_m else ""),
                "abstract": _clean_atom_text(summary_m.group(1) if summary_m else ""),
                "year": int(published_m.group(1)) if published_m else 0,
                "authors": authors,
                "categories": categories,
            })
        return results

    # -- LaTeX extraction -----------------------------------------------

    def _extract_latex_text(self, source_path: Path) -> dict[str, Any]:
        """Extract text from a LaTeX source tarball."""
        result: dict[str, Any] = {
            "full_text": "",
            "sections": {},
            "code_blocks": [],
            "source_format": "latex",
        }

        try:
            with tarfile.open(source_path, "r:gz") as tar:
                tex_files: list[tarfile.TarInfo] = []
                other_files: list[tarfile.TarInfo] = []

                for member in tar.getmembers():
                    if member.isfile():
                        name = member.name.lower()
                        if name.endswith(".tex"):
                            tex_files.append(member)
                        elif name.endswith((".html", ".htm")):
                            other_files.append(member)
                        elif member.name.endswith(".pdf"):
                            other_files.append(member)

                if tex_files:
                    result["source_format"] = "latex"
                    full_text_parts: list[str] = []

                    for tf in tex_files:
                        try:
                            content = _read_tar_entry(tar, tf)
                        except (OSError, tarfile.TarError) as exc:
                            logger.warning("Failed to read %s: %s", tf.name, exc)
                            continue

                        if self._include_code_blocks:
                            code_blocks = _extract_code_blocks(content)
                            result["code_blocks"].extend(code_blocks)

                        cleaned = strip_latex_commands(content)
                        if cleaned:
                            full_text_parts.append(cleaned)

                    combined = "\n\n".join(full_text_parts)
                    result["full_text"] = combined
                    result["sections"] = split_into_sections(combined)

                elif other_files:
                    result["source_format"] = "html"
                    for of in other_files:
                        try:
                            content = _read_tar_entry(tar, of)
                        except (OSError, tarfile.TarError):
                            continue
                        if of.name.endswith(".pdf"):
                            result["full_text"] = ""
                            result["source_format"] = "pdf"
                        else:
                            cleaned = strip_html_tags(content)
                            if cleaned:
                                result["full_text"] = cleaned
                                result["sections"] = split_into_sections(cleaned)

        except (tarfile.TarError, OSError) as exc:
            logger.error("Failed to extract %s: %s", source_path, exc)

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_bulk(
        self,
        target_dir: str | Path,
        categories: list[str] | None = None,
        max_papers: int | None = None,
    ) -> list[tuple[str, str]]:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        result: list[tuple[str, str]] = []

        cats = categories or self._categories
        tqdm = _import_tqdm()

        for cat in cats:
            logger.info("Downloading category: %s", cat)
            papers = self._download_api(cat, max_results=min(max_papers or 5000, 1000))
            if tqdm is not None:
                papers = tqdm(papers, desc=f"Downloading {cat}")

            count = 0
            for paper in papers:
                if max_papers is not None and count >= max_papers:
                    break
                arxiv_id = paper["id"]
                if self._is_processed(target, arxiv_id):
                    continue
                dest = self._download_s3(arxiv_id, target)
                if dest is not None:
                    result.append((arxiv_id, str(dest)))
                    self._update_state(target, arxiv_id)
                    count += 1

        return result

    def extract_paper(
        self,
        arxiv_id: str,
        source_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        sp = Path(source_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if not sp.exists():
            logger.error("Source not found: %s", sp)
            return {}

        extracted = self._extract_latex_text(sp)
        full_text = extracted.get("full_text", "")

        if not full_text:
            return {}

        metadata = self._extract_metadata_from_text(arxiv_id, full_text)
        quality = compute_quality_score(full_text)

        if quality < 0.3:
            logger.debug("Skipping %s: quality too low (%.2f)", arxiv_id, quality)
            return {}

        if len(full_text) < self._min_text_length:
            return {}

        # trim
        if len(full_text) > self._max_text_length:
            full_text = full_text[:self._max_text_length]

        # language check
        lang = detect_language(full_text)
        if lang != "en":
            logger.debug("Skipping %s: language=%s", arxiv_id, lang)
            return {}

        # save raw text
        txt_path = out / f"{arxiv_id}.txt"
        txt_path.write_text(full_text, encoding="utf-8")

        sections = extracted.get("sections", {})
        code_blocks = extracted.get("code_blocks", [])

        paper_dict: dict[str, Any] = {
            "id": arxiv_id,
            "title": metadata.title,
            "authors": metadata.authors,
            "categories": metadata.categories,
            "year": metadata.year,
            "abstract": metadata.abstract,
            "full_text": full_text,
            "sections": sections,
            "code_blocks": code_blocks if self._include_code_blocks else [],
            "quality_score": quality,
            "source_format": extracted.get("source_format", ""),
        }

        self._update_state(out, arxiv_id)
        return paper_dict

    def _extract_metadata_from_text(
        self, arxiv_id: str, text: str
    ) -> PaperMetadata:
        title = ""
        authors: list[str] = []
        abstract = ""
        year = datetime.now().year

        lines = text.split("\n")
        for line in lines[:50]:
            if not title:
                tm = re.match(r'\\title\s*\{([^}]+)\}', line)
                if tm:
                    title = tm.group(1)
                    continue
            if not authors:
                am = re.match(r'\\author\s*\{([^}]+)\}', line)
                if am:
                    raw = am.group(1)
                    authors = [a.strip() for a in re.split(r'\\and|,', raw) if a.strip()]
                    continue

        # Extract year from arxiv ID pattern
        ym = re.match(r'(\d{4})', arxiv_id)
        if ym:
            year = int(ym.group(1))

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title or arxiv_id,
            authors=authors or ["Unknown"],
            categories=self._infer_categories(text),
            abstract=abstract,
            year=year,
        )

    @staticmethod
    def _infer_categories(text: str) -> list[str]:
        cats: list[str] = []
        for line in text.split("\n")[:30]:
            m = re.search(r'\\journal|\\acm|\\pacs|\\subjclass', line, re.IGNORECASE)
            if m:
                rest = line[m.end():]
                found = re.findall(r'[A-Za-z]+\.[A-Za-z]+', rest)
                cats.extend(found)
        return cats or ["unknown"]

    def process_papers(
        self,
        source_dir: str | Path,
        output_dir: str | Path,
        limit: int | None = None,
    ) -> int:
        src = Path(source_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tqdm = _import_tqdm()

        tar_files = sorted(src.glob("*.tar.gz"))
        if limit is not None:
            tar_files = tar_files[:limit]

        count = 0
        jsonl_path = out / "papers.jsonl"
        jfh: io.TextIOWrapper | None = None

        try:
            jfh = jsonl_path.open("w", encoding="utf-8") if self._include_full_text else None

            iterator = tqdm(tar_files, desc="Processing") if tqdm else tar_files
            for tf_path in iterator:
                arxiv_id = tf_path.stem.replace(".tar", "")
                if self._is_processed(out, arxiv_id):
                    continue

                paper = self.extract_paper(arxiv_id, tf_path, out)
                if not paper:
                    continue

                if jfh is not None:
                    jfh.write(json.dumps(paper, ensure_ascii=False) + "\n")

                count += 1
                if limit is not None and count >= limit:
                    break

        finally:
            if jfh is not None:
                jfh.close()

        logger.info("Processed %d papers -> %s", count, jsonl_path)
        return count

    def tokenize_papers(
        self,
        input_jsonl: str | Path,
        output_dir: str | Path,
        tokenizer_path: str,
    ) -> int:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        inp = Path(input_jsonl)

        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tokenizer_path)
        except ImportError:
            logger.error("tokenizers package not available")
            return 0

        count = 0
        shard_size = 16384
        current_shard: list[int] = []
        shard_idx = 0

        with inp.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = paper.get("full_text") or paper.get("abstract", "")
                if not text:
                    continue

                encoded = tok.encode(text)
                ids = encoded.ids
                current_shard.extend(ids)
                count += 1

                while len(current_shard) >= shard_size:
                    shard = current_shard[:shard_size]
                    current_shard = current_shard[shard_size:]
                    shard_path = out / f"shard_{shard_idx:06d}.npy"
                    import numpy as np
                    np.save(str(shard_path), np.array(shard, dtype=np.uint16))
                    shard_idx += 1

        if current_shard:
            shard_path = out / f"shard_{shard_idx:06d}.npy"
            import numpy as np
            np.save(str(shard_path), np.array(current_shard, dtype=np.uint16))

        logger.info("Tokenized %d papers -> %d shards in %s", count, shard_idx + 1, out)
        return count

    def create_instruction_data(
        self,
        input_jsonl: str | Path,
        output_jsonl: str | Path,
    ) -> int:
        inp = Path(input_jsonl)
        out = Path(output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with (
            inp.open("r", encoding="utf-8") as ifh,
            out.open("w", encoding="utf-8") as ofh,
        ):
            for line in ifh:
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    continue

                qa_pairs = self._generate_qa_pairs(paper)
                for qa in qa_pairs:
                    ofh.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    count += 1

        logger.info("Generated %d instruction pairs -> %s", count, out)
        return count

    def _generate_qa_pairs(self, paper: dict) -> list[dict]:
        pairs: list[dict] = []
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        sections = paper.get("sections", {})
        arxiv_id = paper.get("id", "")

        if title and abstract:
            pairs.append({
                "instruction": f"Summarize the paper titled '{title}'.",
                "response": abstract[:2000],
                "id": arxiv_id,
                "type": "title_summary",
            })

        for heading, content in sections.items():
            if len(content) > 100:
                pairs.append({
                    "instruction": f"Describe the {heading} of the paper titled '{title}'.",
                    "response": content[:2000],
                    "id": arxiv_id,
                    "type": "section",
                    "section": heading,
                })

        return pairs

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir: str | Path,
        max_papers: int | None = None,
    ) -> dict[str, Any]:
        out = Path(output_dir)
        sources = out / "sources"
        processed = out / "processed"
        sources.mkdir(parents=True, exist_ok=True)
        processed.mkdir(parents=True, exist_ok=True)

        stats: dict[str, Any] = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "downloaded": 0,
            "processed": 0,
            "tokenized": 0,
            "instruction_pairs": 0,
            "errors": [],
        }

        # 1. download
        logger.info("Step 1: Downloading bulk data...")
        try:
            downloaded = self.download_bulk(sources, max_papers=max_papers)
            stats["downloaded"] = len(downloaded)
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            stats["errors"].append(f"download: {exc}")

        # 2. process
        logger.info("Step 2: Processing papers...")
        try:
            processed_count = self.process_papers(sources, processed, limit=max_papers)
            stats["processed"] = processed_count
        except Exception as exc:
            logger.error("Processing failed: %s", exc)
            stats["errors"].append(f"process: {exc}")

        # 3. instruction data
        jsonl_path = processed / "papers.jsonl"
        if jsonl_path.exists():
            logger.info("Step 3: Creating instruction data...")
            try:
                inst_path = processed / "instructions.jsonl"
                inst_count = self.create_instruction_data(jsonl_path, inst_path)
                stats["instruction_pairs"] = inst_count
            except Exception as exc:
                logger.error("Instruction generation failed: %s", exc)
                stats["errors"].append(f"instruction: {exc}")

        stats["end_time"] = datetime.now(timezone.utc).isoformat()
        stats["output_dir"] = str(out)

        # save run stats
        stats_path = out / "_run_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Pipeline complete: %s", json.dumps(stats, indent=2))
        return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_atom_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    if unidecode is not None:
        text = unidecode.unidecode(text)
    return text


def _read_tar_entry(tar: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    fobj = tar.extractfile(member)
    if fobj is None:
        return ""
    raw = fobj.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_code_blocks(tex_content: str) -> list[str]:
    blocks: list[str] = []
    for m in VERBATIM_RE.finditer(tex_content):
        blocks.append(m.group(1).strip())
    for m in LISTINGS_RE.finditer(tex_content):
        blocks.append(m.group(1).strip())
    return blocks
