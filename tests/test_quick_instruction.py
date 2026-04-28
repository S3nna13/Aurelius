"""Tests for Quick Instruction system."""

from src.serving.quick_instruction import (
    QuickInstructionConfig,
    QuickInstructionManager,
    QuickInstructionToken,
)


def test_quick_instruction_token_enum():
    assert QuickInstructionToken.ACTION.value == "<|action|>"
    assert QuickInstructionToken.QUERY.value == "<|query|>"


def test_quick_instruction_config_defaults():
    config = QuickInstructionConfig()
    assert config.enabled is False
    assert config.search_action is True
    assert config.domain_classification is True


def test_build_prompt_suffix():
    config = QuickInstructionConfig(enabled=True)
    token_ids = {t.value: 0 for t in QuickInstructionToken}
    manager = QuickInstructionManager(config, token_ids)
    suffix = manager.build_prompt_suffix("Hello")
    assert "<|action|>" in suffix
    assert "<|query|>" in suffix


def test_build_prompt_suffix_disabled():
    config = QuickInstructionConfig(enabled=False)
    token_ids = {t.value: 0 for t in QuickInstructionToken}
    manager = QuickInstructionManager(config, token_ids)
    assert manager.build_prompt_suffix("Hello") == ""


def test_should_search():
    config = QuickInstructionConfig(enabled=True)
    manager = QuickInstructionManager(config, {})
    assert manager.should_search([0.8]) is True
    assert manager.should_search([0.3]) is False


def test_classify_domain():
    config = QuickInstructionConfig(enabled=True)
    manager = QuickInstructionManager(config, {})
    domain = manager.classify_domain([0.1, 0.8, 0.1], ["math", "coding", "general"])
    assert domain == "coding"


def test_build_title_suffix_disabled():
    config = QuickInstructionConfig(enabled=False)
    manager = QuickInstructionManager(config, {})
    assert manager.build_title_suffix() == ""
