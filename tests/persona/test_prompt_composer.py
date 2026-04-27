"""Tests for src/persona/prompt_composer.py."""

import pytest

from src.persona.prompt_composer import (
    PromptComposer,
    SystemPromptFragment,
    SystemPromptPriority,
)
from src.persona.unified_persona import (
    Guardrail,
    GuardrailScope,
    GuardrailSeverity,
    IntentMapping,
    PersonaDomain,
    PersonaFacet,
    UnifiedPersona,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def composer():
    return PromptComposer()


@pytest.fixture
def basic_persona():
    return UnifiedPersona(
        id="test",
        name="Test",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="Be helpful.",
    )


# ---------------------------------------------------------------------------
# PromptComposer init
# ---------------------------------------------------------------------------


def test_init_default():
    c = PromptComposer()
    assert c.max_total_chars == 16_000
    assert c.separator == "\n---\n"


def test_init_custom():
    c = PromptComposer(max_total_chars=1000, separator=" | ")
    assert c.max_total_chars == 1000
    assert c.separator == " | "


def test_init_invalid_max():
    with pytest.raises(ValueError, match="max_total_chars must be positive"):
        PromptComposer(max_total_chars=0)
    with pytest.raises(ValueError, match="max_total_chars must be positive"):
        PromptComposer(max_total_chars=-1)


# ---------------------------------------------------------------------------
# compose — basic
# ---------------------------------------------------------------------------


def test_compose_basic(composer, basic_persona):
    result = composer.compose(basic_persona)
    assert "Be helpful." in result
    assert "[SOURCE:test" in result
    assert "PRIORITY:MODEL]" in result


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_basic(composer, basic_persona):
    messages = composer.build_messages(basic_persona, "Hello")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Be helpful." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_build_messages_with_history(composer, basic_persona):
    history = [{"role": "user", "content": "Prev"}, {"role": "assistant", "content": "Ans"}]
    messages = composer.build_messages(basic_persona, "Hello", history=history)
    assert len(messages) == 4
    assert messages[1] == {"role": "user", "content": "Prev"}
    assert messages[2] == {"role": "assistant", "content": "Ans"}


# ---------------------------------------------------------------------------
# Priority encoding / sorting
# ---------------------------------------------------------------------------


def test_priority_sorting(composer, basic_persona):
    fragment = SystemPromptFragment(
        priority=SystemPromptPriority.DEVELOPER,
        content="Developer override",
        source_id="dev",
    )
    result = composer.compose(basic_persona, context_fragments=[fragment])
    lines = result.split("\n")
    # DEVELOPER (0) should appear before MODEL (4)
    dev_idx = next(i for i, l in enumerate(lines) if "DEVELOPER" in l)
    model_idx = next(i for i, l in enumerate(lines) if "MODEL" in l)
    assert dev_idx < model_idx


def test_empty_fragments_skipped(composer, basic_persona):
    fragment = SystemPromptFragment(
        priority=SystemPromptPriority.DEVELOPER,
        content="   ",
        source_id="empty",
    )
    result = composer.compose(basic_persona, context_fragments=[fragment])
    assert "empty" not in result


# ---------------------------------------------------------------------------
# Facet rendering
# ---------------------------------------------------------------------------


def test_security_facet_rendered(composer):
    persona = UnifiedPersona(
        id="sec",
        name="Sec",
        domain=PersonaDomain.SECURITY,
        description="desc",
        system_prompt="Secure.",
        facets=(PersonaFacet("security", {"mode": "offensive", "scope": "internal"}),),
    )
    result = composer.compose(persona)
    assert "Security mode: offensive" in result
    assert "Scope: internal assets only" in result


def test_threat_intel_facet_rendered(composer):
    persona = UnifiedPersona(
        id="ti",
        name="TI",
        domain=PersonaDomain.THREAT_INTEL,
        description="desc",
        system_prompt="Intel.",
        facets=(PersonaFacet("threat_intel", {"query_classifiers": ["cve", "mitre"]}),),
    )
    result = composer.compose(persona)
    assert "cve, mitre" in result


def test_agent_mode_facet_rendered(composer):
    persona = UnifiedPersona(
        id="agent",
        name="Agent",
        domain=PersonaDomain.AGENT,
        description="desc",
        system_prompt="Agent.",
        facets=(PersonaFacet("agent_mode", {"mode": "code"}),),
    )
    result = composer.compose(persona)
    assert "Agent mode: code." in result


def test_constitution_facet_rendered(composer):
    persona = UnifiedPersona(
        id="c",
        name="C",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="Const.",
        facets=(PersonaFacet("constitution", {"dimensions": ["honesty", "helpfulness"]}),),
    )
    result = composer.compose(persona)
    assert "Constitutional dimensions: honesty, helpfulness." in result


def test_harm_filter_facet_rendered(composer):
    persona = UnifiedPersona(
        id="hf",
        name="HF",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="HF.",
        facets=(PersonaFacet("harm_filter", {"categories": ["malicious_code"], "action": "block"}),),
    )
    result = composer.compose(persona)
    assert "Harm filter: malicious_code" in result
    assert "action: block" in result


def test_personality_facet_rendered(composer):
    persona = UnifiedPersona(
        id="p",
        name="P",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="P.",
        facets=(PersonaFacet("personality", {"traits": ["witty", "precise"]}),),
    )
    result = composer.compose(persona)
    assert "Personality traits: witty, precise." in result


def test_personality_facet_empty_traits_omitted(composer):
    persona = UnifiedPersona(
        id="p",
        name="P",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="P.",
        facets=(PersonaFacet("personality", {"traits": []}),),
    )
    result = composer.compose(persona)
    assert "Personality traits" not in result


def test_dialogue_facet_rendered(composer):
    persona = UnifiedPersona(
        id="d",
        name="D",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="D.",
        facets=(PersonaFacet("dialogue", {}),),
    )
    result = composer.compose(persona)
    assert "dialogue state machine" in result


def test_unknown_facet_returns_empty(composer):
    persona = UnifiedPersona(
        id="u",
        name="U",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="U.",
        facets=(PersonaFacet("unknown", {"key": "val"}),),
    )
    result = composer.compose(persona)
    assert "unknown" not in result.lower()


# ---------------------------------------------------------------------------
# Guardrail rendering
# ---------------------------------------------------------------------------


def test_critical_guardrail_high_priority(composer):
    persona = UnifiedPersona(
        id="g",
        name="G",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="G.",
        guardrails=(
            Guardrail("g1", "Never do X.", GuardrailSeverity.CRITICAL, GuardrailScope.ALL),
        ),
    )
    result = composer.compose(persona)
    assert "MUST: Never do X." in result
    assert "DEVELOPER" in result


def test_high_guardrail_high_priority(composer):
    persona = UnifiedPersona(
        id="g",
        name="G",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="G.",
        guardrails=(
            Guardrail("g1", "Be careful.", GuardrailSeverity.HIGH, GuardrailScope.ALL),
        ),
    )
    result = composer.compose(persona)
    assert "MUST: Be careful." in result
    assert "DEVELOPER" in result


def test_medium_guardrail_operator_priority(composer):
    persona = UnifiedPersona(
        id="g",
        name="G",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="G.",
        guardrails=(
            Guardrail("g1", "Prefer this.", GuardrailSeverity.MEDIUM, GuardrailScope.ALL),
        ),
    )
    result = composer.compose(persona)
    assert "SHOULD: Prefer this." in result
    assert "OPERATOR" in result


def test_low_guardrail_operator_priority(composer):
    persona = UnifiedPersona(
        id="g",
        name="G",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="G.",
        guardrails=(
            Guardrail("g1", "Consider this.", GuardrailSeverity.LOW, GuardrailScope.ALL),
        ),
    )
    result = composer.compose(persona)
    assert "SHOULD: Consider this." in result


# ---------------------------------------------------------------------------
# Intent hint resolution
# ---------------------------------------------------------------------------


def test_intent_hint_with_contract(composer):
    persona = UnifiedPersona(
        id="i",
        name="I",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="I.",
        intent_mappings=(IntentMapping("question", "answer", "general_qa"),),
    )
    result = composer.compose(persona, user_input="I have a question about AI")
    assert "[intent=question]" in result
    assert "general_qa" in result


def test_intent_hint_without_contract(composer):
    persona = UnifiedPersona(
        id="i",
        name="I",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="I.",
        intent_mappings=(IntentMapping("explain", "provide explanation"),),
    )
    result = composer.compose(persona, user_input="Explain this")
    assert "[intent=explain]" in result
    assert "provide explanation" in result


def test_intent_hint_default(composer):
    persona = UnifiedPersona(
        id="i",
        name="I",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt="I.",
        intent_mappings=(IntentMapping("question", "answer", "general_qa"),),
    )
    result = composer.compose(persona, user_input="Something unrelated")
    assert "[intent=detect]" in result
    assert "structured-output contract" in result


def test_intent_hint_no_mappings(composer, basic_persona):
    result = composer.compose(basic_persona, user_input="What?")
    assert "[intent=" not in result


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_evicts_low_priority(composer):
    long_prompt = "x" * 20_000
    persona = UnifiedPersona(
        id="long",
        name="Long",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt=long_prompt,
        facets=(PersonaFacet("personality", {"traits": ["witty"]}),),
    )
    result = composer.compose(persona)
    assert len(result) <= composer.max_total_chars


def test_truncation_preserves_immutable(composer):
    long_prompt = "x" * 20_000
    persona = UnifiedPersona(
        id="long",
        name="Long",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt=long_prompt,
        immutable_prompt=True,
    )
    result = composer.compose(persona)
    assert "x" * 100 in result
    assert "[SOURCE:long" in result


def test_truncation_preserves_security_facet_immutable(composer):
    long_prompt = "x" * 20_000
    persona = UnifiedPersona(
        id="long",
        name="Long",
        domain=PersonaDomain.GENERAL,
        description="desc",
        system_prompt=long_prompt,
        facets=(PersonaFacet("security", {"mode": "defensive"}),),
    )
    result = composer.compose(persona)
    assert "Security mode: defensive" in result


def test_no_truncation_when_under_limit(composer, basic_persona):
    result = composer.compose(basic_persona)
    assert "Be helpful." in result


# ---------------------------------------------------------------------------
# Context fragments merging
# ---------------------------------------------------------------------------


def test_context_fragments_merged(composer, basic_persona):
    fragment = SystemPromptFragment(
        priority=SystemPromptPriority.OPERATOR,
        content="External context.",
        source_id="external",
    )
    result = composer.compose(basic_persona, context_fragments=[fragment])
    assert "External context." in result
    assert "[SOURCE:external" in result


def test_multiple_context_fragments(composer, basic_persona):
    fragments = [
        SystemPromptFragment(
            priority=SystemPromptPriority.OPERATOR,
            content="First.",
            source_id="f1",
        ),
        SystemPromptFragment(
            priority=SystemPromptPriority.TOOL,
            content="Second.",
            source_id="f2",
        ),
    ]
    result = composer.compose(basic_persona, context_fragments=fragments)
    assert "First." in result
    assert "Second." in result


# ---------------------------------------------------------------------------
# SystemPromptFragment immutability
# ---------------------------------------------------------------------------


def test_fragment_is_frozen():
    fragment = SystemPromptFragment(
        priority=SystemPromptPriority.DEVELOPER,
        content="text",
        source_id="src",
    )
    with pytest.raises(Exception):
        fragment.content = "new"


def test_system_prompt_priority_values():
    assert SystemPromptPriority.DEVELOPER == 0
    assert SystemPromptPriority.OPERATOR == 1
    assert SystemPromptPriority.USER == 2
    assert SystemPromptPriority.TOOL == 3
    assert SystemPromptPriority.MODEL == 4
