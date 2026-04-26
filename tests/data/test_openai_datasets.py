"""Tests for OpenAI dataset loaders — gsm8k, mmmlu, mrcr, graphwalks,
frontierscience, coval, gdpval, healthbench.

All tests use mock data; no network requests are made.
"""

from src.data.openai_datasets import (
    FrontierScienceDataset,
    # FrontierScience
    FrontierScienceSample,
    GDPvalDataset,
    # GDPval
    GDPvalSample,
    # GraphWalks
    GraphWalkSample,
    GraphWalksDataset,
    GSM8KDataset,
    # GSM8K
    GSM8KSample,
    HealthBenchDataset,
    # HealthBench
    HealthBenchSample,
    # MMMLU
    MMLUSample,
    MMMLUDataset,
    MRCRDataset,
    # MRCR
    MRCRSample,
    # CoVal
    coval_rankings_to_pairs,
    coval_to_dpo_format,
    graphwalks_f1_score,
    parse_gsm8k_answer,
    parse_gsm8k_final_number,
)

# ── GSM8K ─────────────────────────────────────────────────────────────────────


def test_parse_gsm8k_answer_splits_correctly():
    raw = "He had 3 apples.\nThen he got 5 more.\n#### 8"
    reasoning, final = parse_gsm8k_answer(raw)
    assert "#### 8" not in reasoning
    assert final == "8"


def test_parse_gsm8k_strips_calculator_annotations():
    raw = "48 / 2 = <<48/2=24>>24\n#### 24"
    reasoning, final = parse_gsm8k_answer(raw)
    assert "<<" not in reasoning
    assert ">>" not in reasoning
    assert final == "24"


def test_parse_gsm8k_final_number():
    raw = "Some steps here.\n#### 72"
    assert parse_gsm8k_final_number(raw) == 72.0


def test_parse_gsm8k_final_number_none():
    raw = "No answer delimeter here"
    assert parse_gsm8k_final_number(raw) is None


def test_gsm8k_dataset_len():
    rows = [
        {"question": "What is 2+2?", "answer": "2 + 2 = <<2+2=4>>4\n#### 4"},
        {"question": "What is 3+3?", "answer": "3 + 3 = <<3+3=6>>6\n#### 6"},
        {"question": "What is 5-1?", "answer": "5 - 1 = <<5-1=4>>4\n#### 4"},
    ]
    ds = GSM8KDataset(rows)
    assert len(ds) == 3


def test_gsm8k_sample_fields():
    rows = [{"question": "What is 7+5?", "answer": "7 + 5 = <<7+5=12>>12\n#### 12"}]
    ds = GSM8KDataset(rows)
    sample = ds[0]
    assert isinstance(sample, GSM8KSample)
    assert sample.question == "What is 7+5?"
    assert sample.final_answer == "12"
    assert sample.final_number == 12.0
    assert "<<" not in sample.reasoning
    assert sample.answer_raw == "7 + 5 = <<7+5=12>>12\n#### 12"


# ── MMMLU ─────────────────────────────────────────────────────────────────────


def test_mmlu_sample_choices_dict():
    rows = [
        {
            "Question": "What is the capital of France?",
            "A": "Berlin",
            "B": "Paris",
            "C": "Madrid",
            "D": "Rome",
            "Answer": "B",
            "Subject": "geography",
        }
    ]
    ds = MMMLUDataset(rows, language="default")
    sample = ds[0]
    assert isinstance(sample, MMLUSample)
    assert set(sample.choices.keys()) == {"A", "B", "C", "D"}
    assert sample.choices["B"] == "Paris"
    assert sample.correct_answer == "B"
    assert sample.subject == "geography"
    assert sample.language == "default"


def test_mmlu_dataset_len():
    rows = [
        {
            "Question": f"Question {i}",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "Answer": "A",
            "Subject": "math",
        }
        for i in range(5)
    ]
    ds = MMMLUDataset(rows, language="FR_FR")
    assert len(ds) == 5


# ── MRCR ──────────────────────────────────────────────────────────────────────


def test_mrcr_sample_fields():
    rows = [
        {
            "prompt": "Long conversation text here...",
            "answer": "42",
            "random_string_to_prepend": "abcdefghij",
            "n_needles": 4,
            "desired_msg_index": 2,
            "total_messages": 100,
            "n_chars": 50000,
            "date_added": "2024-01-01",
        }
    ]
    ds = MRCRDataset(rows)
    sample = ds[0]
    assert isinstance(sample, MRCRSample)
    assert sample.prompt == "Long conversation text here..."
    assert sample.answer == "42"
    assert sample.prepend_hash == "abcdefghij"
    assert sample.n_needles == 4
    assert sample.desired_index == 2
    assert sample.total_messages == 100
    assert sample.n_chars == 50000


# ── GraphWalks ────────────────────────────────────────────────────────────────


def test_graphwalks_f1_perfect():
    predicted = ["A", "B", "C"]
    gold = ["A", "B", "C"]
    assert graphwalks_f1_score(predicted, gold) == 1.0


def test_graphwalks_f1_no_overlap():
    predicted = ["A", "B"]
    gold = ["C", "D"]
    assert graphwalks_f1_score(predicted, gold) == 0.0


def test_graphwalks_f1_partial():
    predicted = ["A", "B", "C"]
    gold = ["A", "B", "D"]
    # precision = 2/3, recall = 2/3, f1 = 2/3
    score = graphwalks_f1_score(predicted, gold)
    assert abs(score - (2 / 3)) < 1e-6


def test_graphwalks_both_empty():
    # Both empty → vacuously correct → f1 = 1.0
    assert graphwalks_f1_score([], []) == 1.0


def test_graphwalks_dataset_len():
    rows = [
        {
            "prompt": "Graph problem",
            "answer_nodes": ["A", "B"],
            "problem_type": "bfs",
            "date_added": "2024-01-01",
        }
    ]
    ds = GraphWalksDataset(rows)
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample, GraphWalkSample)
    assert sample.problem_type == "bfs"


# ── CoVal ─────────────────────────────────────────────────────────────────────


def test_coval_rankings_to_pairs():
    assessments = [
        {"ranking": "B>A>C=D"},
    ]
    pairs = coval_rankings_to_pairs(assessments)
    # B > A, B > C, B > D, A > C, A > D are all valid
    assert ("B", "A") in pairs
    assert ("B", "C") in pairs
    assert ("B", "D") in pairs
    assert ("A", "C") in pairs
    assert ("A", "D") in pairs
    # C=D means no pair between C and D
    assert ("C", "D") not in pairs
    assert ("D", "C") not in pairs


def test_coval_to_dpo_format_keys():
    comparisons = [
        {
            "prompt_id": "p1",
            "prompt": [{"role": "user", "content": "Hello"}],
            "responses": {"A": "Good response", "B": "Bad response", "C": "OK", "D": "Meh"},
            "metadata": {
                "assessments": [
                    {"ranking": "A>B>C>D"},
                    {"ranking": "A>B>C>D"},
                    {"ranking": "A>B>D>C"},
                ]
            },
        }
    ]
    dpo_data = coval_to_dpo_format(comparisons)
    assert len(dpo_data) > 0
    for item in dpo_data:
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item


# ── FrontierScience ───────────────────────────────────────────────────────────


def test_frontier_science_sample_fields():
    rows = [
        {
            "problem": "Calculate the binding energy...",
            "answer": "13.6 eV",
            "subject": "Physics",
            "task_group_id": "123e4567-e89b-12d3-a456-426614174000",
        }
    ]
    ds = FrontierScienceDataset(rows)
    sample = ds[0]
    assert isinstance(sample, FrontierScienceSample)
    assert sample.problem == "Calculate the binding energy..."
    assert sample.answer == "13.6 eV"
    assert sample.subject == "Physics"
    assert sample.task_group_id == "123e4567-e89b-12d3-a456-426614174000"


# ── GDPval ────────────────────────────────────────────────────────────────────


def test_gdpval_sample_fields():
    rows = [
        {
            "task_id": "task_001",
            "sector": "Finance",
            "occupation": "Analyst",
            "prompt": "Analyze this financial report...",
            "rubric_json": '{"criteria": []}',
            "rubric_pretty": "Evaluation criteria: ...",
        }
    ]
    ds = GDPvalDataset(rows)
    sample = ds[0]
    assert isinstance(sample, GDPvalSample)
    assert sample.task_id == "task_001"
    assert sample.sector == "Finance"
    assert sample.occupation == "Analyst"
    assert sample.prompt == "Analyze this financial report..."
    assert sample.rubric_pretty == "Evaluation criteria: ..."


# ── HealthBench ───────────────────────────────────────────────────────────────


def test_health_bench_sample_fields():
    rows = [
        {
            "prompt": [{"role": "user", "content": "What is aspirin used for?"}],
            "prompt_id": "hb_001",
            "completion": "Aspirin is used for pain relief...",
            "completion_id": "c_001",
            "category": "medication",
            "rubrics": [{"criterion": "accuracy", "points": 5, "tags": ["factual"]}],
            "example_tags": ["pharmacology", "OTC"],
        }
    ]
    ds = HealthBenchDataset(rows)
    sample = ds[0]
    assert isinstance(sample, HealthBenchSample)
    assert sample.prompt_messages == [{"role": "user", "content": "What is aspirin used for?"}]
    assert sample.completion == "Aspirin is used for pain relief..."
    assert sample.category == "medication"
    assert sample.rubrics == [{"criterion": "accuracy", "points": 5, "tags": ["factual"]}]
    assert "pharmacology" in sample.example_tags
    assert sample.prompt_id == "hb_001"
