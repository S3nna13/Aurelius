"""Tests for src/memory/semantic_memory.py — ~50 tests."""

import pytest
from src.memory.semantic_memory import (
    RelationType,
    Concept,
    Relation,
    SemanticMemory,
)
from src.memory import MEMORY_REGISTRY


# ---------------------------------------------------------------------------
# RelationType enum
# ---------------------------------------------------------------------------

def test_relation_type_is_a():
    assert RelationType.IS_A == "is_a"

def test_relation_type_has_property():
    assert RelationType.HAS_PROPERTY == "has_property"

def test_relation_type_related_to():
    assert RelationType.RELATED_TO == "related_to"

def test_relation_type_part_of():
    assert RelationType.PART_OF == "part_of"

def test_relation_type_causes():
    assert RelationType.CAUSES == "causes"


# ---------------------------------------------------------------------------
# Concept dataclass
# ---------------------------------------------------------------------------

def test_concept_name():
    c = Concept(name="dog")
    assert c.name == "dog"

def test_concept_description_default():
    c = Concept(name="dog")
    assert c.description == ""

def test_concept_attributes_default():
    c = Concept(name="dog")
    assert c.attributes == {}

def test_concept_description_set():
    c = Concept(name="dog", description="a pet")
    assert c.description == "a pet"

def test_concept_attributes_set():
    c = Concept(name="dog", attributes={"legs": 4})
    assert c.attributes["legs"] == 4

def test_concept_attributes_independent():
    c1 = Concept(name="a")
    c2 = Concept(name="b")
    c1.attributes["x"] = 1
    assert "x" not in c2.attributes


# ---------------------------------------------------------------------------
# SemanticMemory — add_concept / get_concept
# ---------------------------------------------------------------------------

def test_add_concept_increases_count():
    sm = SemanticMemory()
    sm.add_concept("cat")
    assert sm.concept_count() == 1

def test_add_concept_returns_concept():
    sm = SemanticMemory()
    c = sm.add_concept("cat", description="feline")
    assert isinstance(c, Concept)
    assert c.name == "cat"

def test_add_concept_get_concept_roundtrip():
    sm = SemanticMemory()
    sm.add_concept("cat", description="feline")
    c = sm.get_concept("cat")
    assert c is not None
    assert c.name == "cat"
    assert c.description == "feline"

def test_add_concept_with_attributes():
    sm = SemanticMemory()
    c = sm.add_concept("bird", legs=2, wings=True)
    assert c.attributes["legs"] == 2
    assert c.attributes["wings"] is True

def test_get_concept_unknown_returns_none():
    sm = SemanticMemory()
    assert sm.get_concept("unknown") is None

def test_add_concept_idempotent():
    sm = SemanticMemory()
    sm.add_concept("cat")
    sm.add_concept("cat")
    assert sm.concept_count() == 1

def test_add_multiple_concepts():
    sm = SemanticMemory()
    sm.add_concept("cat")
    sm.add_concept("dog")
    assert sm.concept_count() == 2

def test_concept_count_starts_zero():
    sm = SemanticMemory()
    assert sm.concept_count() == 0


# ---------------------------------------------------------------------------
# SemanticMemory — add_relation
# ---------------------------------------------------------------------------

def test_add_relation_returns_relation():
    sm = SemanticMemory()
    sm.add_concept("cat")
    sm.add_concept("animal")
    r = sm.add_relation("cat", RelationType.IS_A, "animal")
    assert isinstance(r, Relation)

def test_add_relation_has_auto_id():
    sm = SemanticMemory()
    sm.add_concept("cat")
    sm.add_concept("animal")
    r = sm.add_relation("cat", RelationType.IS_A, "animal")
    assert r.id and len(r.id) == 8

def test_add_relation_unknown_source_raises():
    sm = SemanticMemory()
    sm.add_concept("animal")
    with pytest.raises(ValueError, match="source"):
        sm.add_relation("ghost", RelationType.IS_A, "animal")

def test_add_relation_unknown_target_raises():
    sm = SemanticMemory()
    sm.add_concept("cat")
    with pytest.raises(ValueError, match="target"):
        sm.add_relation("cat", RelationType.IS_A, "ghost")

def test_add_relation_stores_fields():
    sm = SemanticMemory()
    sm.add_concept("fire")
    sm.add_concept("smoke")
    r = sm.add_relation("fire", RelationType.CAUSES, "smoke", weight=0.9)
    assert r.source == "fire"
    assert r.target == "smoke"
    assert r.relation_type == RelationType.CAUSES
    assert r.weight == pytest.approx(0.9)

def test_relation_count_increments():
    sm = SemanticMemory()
    sm.add_concept("a")
    sm.add_concept("b")
    sm.add_concept("c")
    assert sm.relation_count() == 0
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    assert sm.relation_count() == 1
    sm.add_relation("b", RelationType.PART_OF, "c")
    assert sm.relation_count() == 2

def test_relation_count_starts_zero():
    sm = SemanticMemory()
    assert sm.relation_count() == 0

def test_add_relation_ids_are_unique():
    sm = SemanticMemory()
    sm.add_concept("a")
    sm.add_concept("b")
    r1 = sm.add_relation("a", RelationType.RELATED_TO, "b")
    r2 = sm.add_relation("a", RelationType.RELATED_TO, "b")
    assert r1.id != r2.id


# ---------------------------------------------------------------------------
# SemanticMemory — get_relations
# ---------------------------------------------------------------------------

def test_get_relations_as_source():
    sm = SemanticMemory()
    sm.add_concept("cat"); sm.add_concept("animal")
    sm.add_relation("cat", RelationType.IS_A, "animal")
    rels = sm.get_relations("cat")
    assert len(rels) == 1
    assert rels[0].source == "cat"

def test_get_relations_as_target():
    sm = SemanticMemory()
    sm.add_concept("cat"); sm.add_concept("animal")
    sm.add_relation("cat", RelationType.IS_A, "animal")
    rels = sm.get_relations("animal")
    assert len(rels) == 1
    assert rels[0].target == "animal"

def test_get_relations_both_directions():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("b"); sm.add_concept("c")
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    sm.add_relation("c", RelationType.RELATED_TO, "a")
    rels = sm.get_relations("a")
    assert len(rels) == 2

def test_get_relations_empty():
    sm = SemanticMemory()
    sm.add_concept("loner")
    assert sm.get_relations("loner") == []


# ---------------------------------------------------------------------------
# SemanticMemory — neighbors
# ---------------------------------------------------------------------------

def test_neighbors_direct():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("b")
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    assert "b" in sm.neighbors("a")

def test_neighbors_reverse():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("b")
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    assert "a" in sm.neighbors("b")

def test_neighbors_empty():
    sm = SemanticMemory()
    sm.add_concept("alone")
    assert sm.neighbors("alone") == set()

def test_neighbors_multiple():
    sm = SemanticMemory()
    sm.add_concept("hub"); sm.add_concept("x"); sm.add_concept("y")
    sm.add_relation("hub", RelationType.RELATED_TO, "x")
    sm.add_relation("hub", RelationType.RELATED_TO, "y")
    assert sm.neighbors("hub") == {"x", "y"}


# ---------------------------------------------------------------------------
# SemanticMemory — find_path
# ---------------------------------------------------------------------------

def test_find_path_direct_neighbor():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("b")
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    path = sm.find_path("a", "b")
    assert path == ["a", "b"]

def test_find_path_same_node():
    sm = SemanticMemory()
    sm.add_concept("a")
    assert sm.find_path("a", "a") == ["a"]

def test_find_path_two_hops():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("b"); sm.add_concept("c")
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    sm.add_relation("b", RelationType.RELATED_TO, "c")
    path = sm.find_path("a", "c")
    assert path == ["a", "b", "c"]

def test_find_path_unreachable_returns_none():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("z")
    assert sm.find_path("a", "z") is None

def test_find_path_respects_max_depth():
    sm = SemanticMemory()
    nodes = ["n0", "n1", "n2", "n3", "n4", "n5", "n6"]
    for n in nodes:
        sm.add_concept(n)
    for i in range(len(nodes) - 1):
        sm.add_relation(nodes[i], RelationType.RELATED_TO, nodes[i + 1])
    # max_depth=3 means path length ≤ 3 hops; n0→n6 needs 6 hops → None
    assert sm.find_path("n0", "n6", max_depth=3) is None

def test_find_path_bidirectional():
    sm = SemanticMemory()
    sm.add_concept("a"); sm.add_concept("b")
    sm.add_relation("a", RelationType.RELATED_TO, "b")
    # b→a should work via reverse traversal
    path = sm.find_path("b", "a")
    assert path is not None
    assert path[0] == "b" and path[-1] == "a"


# ---------------------------------------------------------------------------
# MEMORY_REGISTRY
# ---------------------------------------------------------------------------

def test_memory_registry_semantic_exists():
    assert "semantic" in MEMORY_REGISTRY

def test_memory_registry_semantic_is_semantic_memory():
    assert isinstance(MEMORY_REGISTRY["semantic"], SemanticMemory)
