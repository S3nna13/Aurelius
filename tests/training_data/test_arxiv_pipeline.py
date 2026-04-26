"""Tests for the arXiv bulk download and preprocessing pipeline."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from training_data.arxiv_pipeline import (
    ArxivPipeline,
    DEFAULT_CONFIG,
    ARXIV_ID_RE,
    compute_math_ratio,
    compute_quality_score,
    detect_language,
    extract_arxiv_id,
    split_into_sections,
    strip_html_tags,
    strip_latex_commands,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline():
    return ArxivPipeline({})


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_tex():
    return (
        r"\title{Attention Is All You Need}"
        r"\author{Ashish Vaswani, Noam Shazeer}"
        r"\begin{abstract}"
        r"We propose a new simple network architecture, the Transformer."
        r"\end{abstract}"
        r"\section{Introduction}"
        r"The dominant sequence transduction models are based on"
        r"complex recurrent or convolutional neural networks."
        r"\section{Method}"
        r"Our model uses multi-head self-attention."
        r"\begin{equation}"
        r"Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"
        r"\end{equation}"
        r"\bibliography{refs}"
    )


@pytest.fixture
def sample_tar(temp_dir, sample_tex):
    tar_path = temp_dir / "1706.03762v1.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        info = tarfile.TarInfo(name="main.tex")
        info.size = len(sample_tex.encode("utf-8"))
        tar.addfile(info, io.BytesIO(sample_tex.encode("utf-8")))
    return tar_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Test: Config defaults
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config_structure(self):
        assert "arxiv" in DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG["arxiv"]
        assert "categories" in cfg
        assert "min_text_length" in cfg
        assert "max_text_length" in cfg
        assert "download_dir" in cfg
        assert "output_dir" in cfg

    def test_pipeline_default_categories(self, pipeline):
        assert len(pipeline.config["categories"]) >= 5
        assert "cs.LG" in pipeline.config["categories"]

    def test_pipeline_custom_config_overrides(self):
        custom = {"arxiv": {"categories": ["cs.AI"], "concurrency": 8}}
        p = ArxivPipeline(custom)
        assert p.config["categories"] == ["cs.AI"]
        assert p.config["concurrency"] == 8
        assert p.config["min_text_length"] == 1000  # inherits default

    def test_pipeline_min_text_length_default(self, pipeline):
        assert pipeline._min_text_length == 1000

    def test_pipeline_max_text_length_default(self, pipeline):
        assert pipeline._max_text_length == 100000


# ---------------------------------------------------------------------------
# Test: Paper ID extraction
# ---------------------------------------------------------------------------

class TestIDExtraction:
    def test_extract_standard_id(self):
        assert extract_arxiv_id("1706.03762") == "1706.03762"

    def test_extract_id_with_version(self):
        assert extract_arxiv_id("1706.03762v2") == "1706.03762v2"

    def test_extract_old_style_id(self):
        assert extract_arxiv_id("cs/0706100") == "cs/0706100"

    def test_extract_id_from_url(self):
        assert extract_arxiv_id("https://arxiv.org/abs/1706.03762v1") == "1706.03762v1"

    def test_extract_id_none(self):
        assert extract_arxiv_id("not an arxiv id") is None


# ---------------------------------------------------------------------------
# Test: Text cleaning
# ---------------------------------------------------------------------------

class TestTextCleaning:
    def test_strip_latex_commands(self):
        text = r"\textbf{bold} and \textit{italic} and \ref{label}"
        cleaned = strip_latex_commands(text)
        assert "bold" not in cleaned
        assert "italic" not in cleaned
        assert "label" not in cleaned

    def test_strip_math_inline(self):
        text = r"This is $E = mc^2$ a test."
        cleaned = strip_latex_commands(text)
        assert "$" not in cleaned

    def test_strip_math_display(self):
        text = r"Before $$\int_a^b f(x)dx$$ after."
        cleaned = strip_latex_commands(text)
        assert "$$" not in cleaned

    def test_strip_citations(self):
        text = r"Research shows \cite{smith2023} and \citep{jones2022}."
        cleaned = strip_latex_commands(text)
        assert "cite" not in cleaned

    def test_strip_comments(self):
        text = "Visible text\n% This is a comment\nMore text"
        cleaned = strip_latex_commands(text)
        assert "%" not in cleaned

    def test_strip_html_tags(self):
        text = "<p>Hello <b>world</b></p>"
        assert strip_html_tags(text) == "Hello world"

    def test_strip_html_entities(self):
        text = "foo &amp; bar &lt; baz"
        cleaned = strip_html_tags(text)
        assert "&amp;" not in cleaned
        assert "&lt;" not in cleaned

    def test_preserve_regular_text(self):
        text = "Regular English text with no LaTeX."
        assert strip_latex_commands(text) == "Regular English text with no LaTeX."


# ---------------------------------------------------------------------------
# Test: Quality filtering
# ---------------------------------------------------------------------------

class TestQualityFilters:
    def test_math_ratio_low(self):
        text = "This is a normal paper with only a few symbols like x."
        assert compute_math_ratio(text) < 0.5

    def test_math_ratio_high(self):
        text = "$\\alpha \\beta \\gamma \\delta \\epsilon \\phi \\sum \\int$"
        assert compute_math_ratio(text) > 0.3

    def test_quality_score_short_text(self):
        text = "Short"
        score = compute_quality_score(text)
        assert score < 0.8

    def test_quality_score_good_text(self):
        text = "A " * 2000
        score = compute_quality_score(text)
        assert score > 0.5

    def test_detect_language_english(self):
        text = "This is a normal English paragraph about machine learning."
        assert detect_language(text) == "en"

    def test_detect_language_non_english_fallback(self):
        text = "这是中文文本" * 100
        lang = detect_language(text)
        assert lang != "en"


# ---------------------------------------------------------------------------
# Test: Section splitting
# ---------------------------------------------------------------------------

class TestSectionSplitting:
    def test_split_basic_sections(self):
        text = (
            "Abstract text here.\n"
            r"\section{Introduction}"
            "\nIntro content.\n"
            r"\section{Method}"
            "\nMethod content."
        )
        sections = split_into_sections(text)
        assert "introduction" in sections
        assert "method" in sections
        assert "abstract" in sections

    def test_split_empty_text(self):
        sections = split_into_sections("")
        assert sections == {}

    def test_split_single_section(self):
        sections = split_into_sections(r"\section{Results}We got good results.")
        assert "results" in sections


# ---------------------------------------------------------------------------
# Test: JSONL output format
# ---------------------------------------------------------------------------

class TestJSONLOutput:
    def test_extract_paper_saves_txt(self, pipeline, sample_tar, temp_dir):
        result = pipeline.extract_paper("1706.03762v1", sample_tar, temp_dir)
        txt_path = temp_dir / "1706.03762v1.txt"
        assert txt_path.exists()
        assert txt_path.stat().st_size > 0

    def test_extract_paper_dict_keys(self, pipeline, sample_tar, temp_dir):
        result = pipeline.extract_paper("1706.03762v1", sample_tar, temp_dir)
        assert "id" in result
        assert "full_text" in result
        assert "categories" in result
        assert "year" in result
        assert "quality_score" in result

    def test_extract_paper_id_match(self, pipeline, sample_tar, temp_dir):
        result = pipeline.extract_paper("1706.03762v1", sample_tar, temp_dir)
        assert result["id"] == "1706.03762v1"

    def test_process_papers_creates_jsonl(self, pipeline, temp_dir):
        count = pipeline.process_papers(temp_dir, temp_dir, limit=0)
        assert count == 0

    def test_instruction_pairs_format(self, pipeline):
        paper = {
            "id": "1706.03762",
            "title": "Test Paper",
            "abstract": "This is a test abstract.",
            "sections": {"method": "Our method is great."},
        }
        pairs = pipeline._generate_qa_pairs(paper)
        assert len(pairs) >= 1
        assert "instruction" in pairs[0]
        assert "response" in pairs[0]
        assert "id" in pairs[0]


# ---------------------------------------------------------------------------
# Test: State tracking
# ---------------------------------------------------------------------------

class TestStateTracking:
    def test_state_initialization(self, pipeline, temp_dir):
        state = pipeline._load_state(temp_dir)
        assert "processed_ids" in state
        assert state["processed_ids"] == []

    def test_state_persistence(self, pipeline, temp_dir):
        pipeline._save_state(temp_dir, {"processed_ids": ["1706.03762"]})
        state = pipeline._load_state(temp_dir)
        assert "1706.03762" in state["processed_ids"]

    def test_is_processed_true(self, pipeline, temp_dir):
        pipeline._save_state(temp_dir, {"processed_ids": ["1706.03762"]})
        assert pipeline._is_processed(temp_dir, "1706.03762") is True

    def test_is_processed_false(self, pipeline, temp_dir):
        pipeline._save_state(temp_dir, {"processed_ids": ["0704.0001"]})
        assert pipeline._is_processed(temp_dir, "1706.03762") is False

    def test_update_state_adds_id(self, pipeline, temp_dir):
        pipeline._update_state(temp_dir, "1706.03762")
        state = pipeline._load_state(temp_dir)
        assert "1706.03762" in state["processed_ids"]

    def test_update_state_no_duplicates(self, pipeline, temp_dir):
        pipeline._update_state(temp_dir, "1706.03762")
        pipeline._update_state(temp_dir, "1706.03762")
        state = pipeline._load_state(temp_dir)
        assert state["processed_ids"].count("1706.03762") == 1


# ---------------------------------------------------------------------------
# Test: Incremental / resume logic
# ---------------------------------------------------------------------------

class TestIncremental:
    def test_skip_processed_papers(self, pipeline, temp_dir):
        pipeline._save_state(temp_dir, {"processed_ids": ["1706.03762"]})
        assert pipeline._is_processed(temp_dir, "1706.03762") is True

    def test_process_new_paper_only(self, pipeline, temp_dir, sample_tar):
        pipeline._save_state(temp_dir, {"processed_ids": []})
        result = pipeline.extract_paper("1706.03762v1", sample_tar, temp_dir)
        assert result is not None and result.get("id") == "1706.03762v1"

    def test_state_has_last_run(self, pipeline, temp_dir):
        pipeline._update_state(temp_dir, "1706.03762")
        state = pipeline._load_state(temp_dir)
        assert "last_run" in state
        assert state["last_run"] is not None

    def test_state_tracking_after_extract(self, pipeline, temp_dir, sample_tar):
        pipeline.extract_paper("1706.03762v1", sample_tar, temp_dir)
        state = pipeline._load_state(temp_dir)
        assert "1706.03762v1" in state["processed_ids"]


# ---------------------------------------------------------------------------
# Test: Pipeline instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_pipeline_loads(self):
        p = ArxivPipeline({})
        assert p is not None
        assert len(p.config["categories"]) > 0

    def test_pipeline_config_key_access(self):
        p = ArxivPipeline({})
        assert p.config["categories"][:3] == ["cs.AI", "cs.LG", "cs.CL"]

    def test_arxiv_id_regex_matches_new_format(self):
        assert ARXIV_ID_RE.match("1706.03762") is not None

    def test_arxiv_id_regex_matches_old_format(self):
        assert ARXIV_ID_RE.match("cs/0706100") is not None
