"""Tests for instruction_following evaluation module."""

import pytest

from src.eval.instruction_following import (
    ComplianceResult,
    Instruction,
    InstructionConfig,
    InstructionFollowingEvaluator,
    check_format_compliance,
    check_keyword_compliance,
    check_length_compliance,
    check_structure_compliance,
    create_instruction_test_suite,
)


# 1. InstructionConfig defaults
def test_instruction_config_defaults():
    config = InstructionConfig()
    assert config.eval_format is True
    assert config.eval_length is True
    assert config.eval_keywords is True
    assert config.eval_structure is True
    assert config.case_sensitive is False


# 2. check_format_compliance: valid JSON -> 1.0
def test_check_format_json_valid():
    response = '{"name": "aurelius", "version": 1}'
    assert check_format_compliance(response, "json") == 1.0


# 3. check_format_compliance: invalid JSON -> 0.0
def test_check_format_json_invalid():
    response = "not valid json at all"
    assert check_format_compliance(response, "json") == 0.0


# 4. check_format_compliance: list format "- item" -> 1.0
def test_check_format_list():
    response = "- first item\n- second item"
    assert check_format_compliance(response, "list") == 1.0


# 5. check_length_compliance: word count in range -> 1.0
def test_check_length_in_range():
    response = " ".join(["word"] * 50)
    assert check_length_compliance(response, min_words=10, max_words=100) == 1.0


# 6. check_length_compliance: too short -> 0.0
def test_check_length_too_short():
    response = "only three words"
    assert check_length_compliance(response, min_words=10, max_words=None) == 0.0


# 7. check_keyword_compliance: all required present -> score > 0
def test_check_keyword_all_required():
    response = "This response contains the important keyword and also relevant details."
    score = check_keyword_compliance(response, required=["important", "relevant"], forbidden=[])
    assert score > 0


# 8. check_keyword_compliance: forbidden keyword present -> 0.0
def test_check_keyword_forbidden_present():
    response = "This response mentions something forbidden."
    score = check_keyword_compliance(response, required=[], forbidden=["forbidden"])
    assert score == 0.0


# 9. check_structure_compliance: numbered "1. a\n2. b\n3. c" -> 1.0
def test_check_structure_numbered():
    response = "1. first\n2. second\n3. third"
    assert check_structure_compliance(response, "numbered") == 1.0


# 10. evaluator ComplianceResult has all expected fields
def test_evaluator_compliance_result_fields():
    config = InstructionConfig()
    evaluator = InstructionFollowingEvaluator(config)
    instruction = Instruction(
        prompt="Describe something.",
        required_format="paragraph",
        min_words=5,
        max_words=200,
        required_keywords=["example"],
        forbidden_keywords=[],
    )
    response = (
        "This is an example of a well-formed paragraph. "
        "It contains multiple sentences and enough words to pass constraints."
    )
    result = evaluator.evaluate(instruction, response)

    assert isinstance(result, ComplianceResult)
    assert hasattr(result, "format_score")
    assert hasattr(result, "length_score")
    assert hasattr(result, "keyword_score")
    assert hasattr(result, "structure_score")
    assert hasattr(result, "overall_score")
    assert 0.0 <= result.overall_score <= 1.0


# 11. aggregate_results returns correct keys
def test_evaluator_aggregate_keys():
    config = InstructionConfig()
    evaluator = InstructionFollowingEvaluator(config)
    instructions = [
        Instruction(prompt="Say hello.", required_format=None),
        Instruction(prompt="List items.", required_format="list"),
    ]
    responses = [
        "Hello there, this is a response with enough words for the test.",
        "- item one\n- item two\n- item three",
    ]
    results = evaluator.evaluate_batch(instructions, responses)
    agg = evaluator.aggregate_results(results)

    assert set(agg.keys()) == {"mean_overall", "format_rate", "length_rate", "keyword_rate", "structure_rate"}
    for v in agg.values():
        assert isinstance(v, float)


# 12. create_instruction_test_suite returns n items
def test_create_instruction_test_suite_count():
    suite = create_instruction_test_suite(n=20, seed=42)
    assert len(suite) == 20
    for instruction, response in suite:
        assert isinstance(instruction, Instruction)
        assert isinstance(response, str)
