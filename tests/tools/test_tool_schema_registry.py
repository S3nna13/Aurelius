from __future__ import annotations

import pytest

from src.tools.tool_schema_registry import (
    ParameterSchema,
    ToolSchema,
    ToolSchemaRegistry,
    TOOL_SCHEMA_REGISTRY,
)


def _make_schema(name: str = "search", tags: list[str] | None = None) -> ToolSchema:
    return ToolSchema(
        name=name,
        description="Search the web",
        parameters=[
            ParameterSchema(name="query", type="string"),
            ParameterSchema(name="limit", type="integer", required=False, default=10),
        ],
        returns="list of results",
        tags=tags or [],
    )


def fresh_registry() -> ToolSchemaRegistry:
    return ToolSchemaRegistry()


class TestParameterSchema:
    def test_required_defaults_true(self):
        p = ParameterSchema(name="x", type="string")
        assert p.required is True

    def test_optional_default_none(self):
        p = ParameterSchema(name="x", type="string", required=False)
        assert p.default is None

    def test_enum_stored(self):
        p = ParameterSchema(name="mode", type="string", enum=["fast", "slow"])
        assert "fast" in p.enum

    def test_description_stored(self):
        p = ParameterSchema(name="x", type="integer", description="a count")
        assert p.description == "a count"


class TestToolSchema:
    def test_version_default(self):
        s = _make_schema()
        assert s.version == "1.0"

    def test_tags_default_empty(self):
        s = ToolSchema(name="t", description="d", parameters=[])
        assert s.tags == []

    def test_returns_default_empty(self):
        s = ToolSchema(name="t", description="d", parameters=[])
        assert s.returns == ""


class TestToolSchemaRegistry:
    def test_register_and_get(self):
        reg = fresh_registry()
        schema = _make_schema("find")
        reg.register(schema)
        assert reg.get("find") is schema

    def test_get_missing_returns_none(self):
        reg = fresh_registry()
        assert reg.get("nonexistent") is None

    def test_list_tools_empty(self):
        reg = fresh_registry()
        assert reg.list_tools() == []

    def test_list_tools_after_register(self):
        reg = fresh_registry()
        reg.register(_make_schema("a"))
        reg.register(_make_schema("b"))
        assert set(reg.list_tools()) == {"a", "b"}

    def test_unregister_existing(self):
        reg = fresh_registry()
        reg.register(_make_schema("del_me"))
        assert reg.unregister("del_me") is True
        assert reg.get("del_me") is None

    def test_unregister_missing_returns_false(self):
        reg = fresh_registry()
        assert reg.unregister("ghost") is False

    def test_list_by_tag(self):
        reg = fresh_registry()
        reg.register(_make_schema("t1", tags=["io", "net"]))
        reg.register(_make_schema("t2", tags=["io"]))
        reg.register(_make_schema("t3", tags=["compute"]))
        io_tools = reg.list_by_tag("io")
        assert len(io_tools) == 2
        names = {s.name for s in io_tools}
        assert "t1" in names and "t2" in names

    def test_list_by_tag_no_match(self):
        reg = fresh_registry()
        reg.register(_make_schema("t1", tags=["net"]))
        assert reg.list_by_tag("missing") == []

    def test_to_dict_returns_none_for_unknown(self):
        reg = fresh_registry()
        assert reg.to_dict("nope") is None

    def test_to_dict_structure(self):
        reg = fresh_registry()
        reg.register(_make_schema("search"))
        d = reg.to_dict("search")
        assert d["name"] == "search"
        assert d["description"] == "Search the web"
        assert isinstance(d["parameters"], list)
        assert d["version"] == "1.0"
        assert d["returns"] == "list of results"

    def test_to_dict_parameter_fields(self):
        reg = fresh_registry()
        reg.register(_make_schema("search"))
        d = reg.to_dict("search")
        param_names = [p["name"] for p in d["parameters"]]
        assert "query" in param_names and "limit" in param_names

    def test_to_dict_enum_included(self):
        reg = fresh_registry()
        schema = ToolSchema(
            name="mode_tool",
            description="choose mode",
            parameters=[
                ParameterSchema(name="mode", type="string", enum=["a", "b"])
            ],
        )
        reg.register(schema)
        d = reg.to_dict("mode_tool")
        p = d["parameters"][0]
        assert p["enum"] == ["a", "b"]

    def test_to_dict_no_enum_key_when_none(self):
        reg = fresh_registry()
        reg.register(_make_schema("search"))
        d = reg.to_dict("search")
        p = next(p for p in d["parameters"] if p["name"] == "query")
        assert "enum" not in p

    def test_all_to_dict_count(self):
        reg = fresh_registry()
        reg.register(_make_schema("a"))
        reg.register(_make_schema("b"))
        result = reg.all_to_dict()
        assert len(result) == 2

    def test_all_to_dict_empty(self):
        reg = fresh_registry()
        assert reg.all_to_dict() == []

    def test_global_registry_is_instance(self):
        assert isinstance(TOOL_SCHEMA_REGISTRY, ToolSchemaRegistry)

    def test_overwrite_registration(self):
        reg = fresh_registry()
        reg.register(_make_schema("dup"))
        new_schema = ToolSchema(name="dup", description="updated", parameters=[])
        reg.register(new_schema)
        assert reg.get("dup").description == "updated"
