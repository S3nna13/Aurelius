"""
test_element_locator.py
Tests for element_locator.py  (>=28 tests)
"""

import pytest

from src.computer_use.element_locator import (
    ELEMENT_LOCATOR_REGISTRY,
    REGISTRY,
    ElementLocator,
    ElementQuery,
    LocatedElement,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_element(
    element_id: str = "e1",
    selector: str = "#button",
    x: int = 10,
    y: int = 20,
    width: int = 100,
    height: int = 50,
    text: str = "Click me",
    visible: bool = True,
) -> LocatedElement:
    return LocatedElement(
        element_id=element_id,
        selector=selector,
        x=x,
        y=y,
        width=width,
        height=height,
        text=text,
        visible=visible,
    )


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in ELEMENT_LOCATOR_REGISTRY


def test_registry_alias():
    assert REGISTRY is ELEMENT_LOCATOR_REGISTRY


def test_registry_default_is_class():
    assert ELEMENT_LOCATOR_REGISTRY["default"] is ElementLocator


# ---------------------------------------------------------------------------
# LocatedElement — frozen / properties
# ---------------------------------------------------------------------------


def test_located_element_frozen():
    el = make_element()
    with pytest.raises((AttributeError, TypeError)):
        el.x = 999  # type: ignore[misc]


def test_located_element_center_basic():
    el = make_element(x=10, y=20, width=100, height=50)
    assert el.center() == (60, 45)


def test_located_element_center_zero_size():
    el = make_element(x=5, y=5, width=0, height=0)
    assert el.center() == (5, 5)


def test_located_element_area_basic():
    el = make_element(width=100, height=50)
    assert el.area() == 5000


def test_located_element_area_zero():
    el = make_element(width=0, height=50)
    assert el.area() == 0


def test_located_element_defaults():
    el = LocatedElement(element_id="x", selector=".foo", x=0, y=0, width=10, height=10)
    assert el.text == ""
    assert el.visible is True


# ---------------------------------------------------------------------------
# ElementQuery — frozen
# ---------------------------------------------------------------------------


def test_element_query_frozen():
    q = ElementQuery(selector="#btn")
    with pytest.raises((AttributeError, TypeError)):
        q.selector = "other"  # type: ignore[misc]


def test_element_query_defaults():
    q = ElementQuery(selector="div")
    assert q.selector_type == "css"
    assert q.region is None


def test_element_query_with_region():
    q = ElementQuery(selector="div", region=(0, 0, 200, 200))
    assert q.region == (0, 0, 200, 200)


# ---------------------------------------------------------------------------
# ElementLocator — register and find
# ---------------------------------------------------------------------------


def test_register_and_find_by_selector():
    loc = ElementLocator()
    el = make_element(selector="#submit-btn")
    loc.register(el)
    results = loc.find(ElementQuery(selector="submit"))
    assert el in results


def test_find_case_insensitive():
    loc = ElementLocator()
    el = make_element(selector="#SubmitButton")
    loc.register(el)
    results = loc.find(ElementQuery(selector="submitbutton"))
    assert el in results


def test_find_case_insensitive_upper_query():
    loc = ElementLocator()
    el = make_element(selector="#submit-btn")
    loc.register(el)
    results = loc.find(ElementQuery(selector="SUBMIT"))
    assert el in results


def test_find_no_match():
    loc = ElementLocator()
    loc.register(make_element(selector="#btn"))
    assert loc.find(ElementQuery(selector="nonexistent")) == []


def test_find_multiple_matches():
    loc = ElementLocator()
    e1 = make_element("e1", "#btn-a")
    e2 = make_element("e2", "#btn-b")
    loc.register(e1)
    loc.register(e2)
    results = loc.find(ElementQuery(selector="btn"))
    assert e1 in results and e2 in results


def test_find_with_region_inside():
    loc = ElementLocator()
    # center = (60, 45) — within region (0,0,200,200)
    el = make_element(x=10, y=20, width=100, height=50)
    loc.register(el)
    results = loc.find(ElementQuery(selector="#button", region=(0, 0, 200, 200)))
    assert el in results


def test_find_with_region_outside():
    loc = ElementLocator()
    # center = (60, 45) — outside region (200,200,100,100)
    el = make_element(x=10, y=20, width=100, height=50)
    loc.register(el)
    results = loc.find(ElementQuery(selector="#button", region=(200, 200, 100, 100)))
    assert el not in results


def test_find_region_boundary_inclusive():
    loc = ElementLocator()
    # center = (10, 10)
    el = make_element(x=10, y=10, width=0, height=0)
    loc.register(el)
    # Region exactly at center
    results = loc.find(ElementQuery(selector="#button", region=(10, 10, 0, 0)))
    assert el in results


# ---------------------------------------------------------------------------
# ElementLocator — find_at
# ---------------------------------------------------------------------------


def test_find_at_inside():
    loc = ElementLocator()
    el = make_element(x=10, y=10, width=80, height=40)
    loc.register(el)
    assert loc.find_at(50, 30) is el


def test_find_at_on_edge():
    loc = ElementLocator()
    el = make_element(x=10, y=10, width=80, height=40)
    loc.register(el)
    assert loc.find_at(10, 10) is el  # top-left corner
    assert loc.find_at(90, 50) is el  # bottom-right corner


def test_find_at_outside_returns_none():
    loc = ElementLocator()
    el = make_element(x=10, y=10, width=80, height=40)
    loc.register(el)
    assert loc.find_at(9, 10) is None
    assert loc.find_at(10, 9) is None


def test_find_at_empty_registry():
    loc = ElementLocator()
    assert loc.find_at(50, 50) is None


def test_find_at_returns_first_match():
    loc = ElementLocator()
    e1 = make_element("e1", "#a", x=0, y=0, width=100, height=100)
    e2 = make_element("e2", "#b", x=0, y=0, width=100, height=100)
    loc.register(e1)
    loc.register(e2)
    assert loc.find_at(50, 50) is e1


# ---------------------------------------------------------------------------
# ElementLocator — visible_elements
# ---------------------------------------------------------------------------


def test_visible_elements_filters_invisible():
    loc = ElementLocator()
    visible = make_element("v1", "#visible", visible=True)
    hidden = make_element("h1", "#hidden", visible=False)
    loc.register(visible)
    loc.register(hidden)
    vis = loc.visible_elements()
    assert visible in vis
    assert hidden not in vis


def test_visible_elements_all_hidden():
    loc = ElementLocator()
    loc.register(make_element(visible=False))
    assert loc.visible_elements() == []


def test_visible_elements_empty():
    loc = ElementLocator()
    assert loc.visible_elements() == []


# ---------------------------------------------------------------------------
# ElementLocator — clear
# ---------------------------------------------------------------------------


def test_clear_empties_registry():
    loc = ElementLocator()
    loc.register(make_element())
    loc.clear()
    assert loc.find(ElementQuery(selector="#button")) == []


def test_clear_then_register():
    loc = ElementLocator()
    loc.register(make_element("e1"))
    loc.clear()
    new_el = make_element("e2", "#new")
    loc.register(new_el)
    assert loc.find(ElementQuery(selector="#new")) == [new_el]
