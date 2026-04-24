import pytest
from src.tools.json_tool import JSONTool, JSONToolConfig, JSON_TOOL_REGISTRY


def test_validate_valid():
    tool = JSONTool()
    ok, err = tool.validate('{"a": 1}')
    assert ok is True
    assert err == ""


def test_validate_invalid():
    tool = JSONTool()
    ok, err = tool.validate("{bad json}")
    assert ok is False
    assert len(err) > 0


def test_format_indents():
    tool = JSONTool()
    result = tool.format('{"b":2,"a":1}')
    assert '"b": 2' in result
    assert "\n" in result


def test_format_sort_keys():
    tool = JSONTool(JSONToolConfig(sort_keys=True))
    result = tool.format('{"b":2,"a":1}')
    assert result.index('"a"') < result.index('"b"')


def test_extract_path_simple():
    tool = JSONTool()
    data = {"a": {"b": {"c": 42}}}
    assert tool.extract_path(data, "a.b.c") == 42


def test_extract_path_list_index():
    tool = JSONTool()
    data = {"items": [10, 20, 30]}
    assert tool.extract_path(data, "items.1") == 20


def test_extract_path_missing_raises():
    tool = JSONTool()
    with pytest.raises((KeyError, IndexError, TypeError)):
        tool.extract_path({"a": 1}, "a.b.c")


def test_merge_deep():
    tool = JSONTool()
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 0}, "c": 4}
    result = tool.merge(base, override)
    assert result == {"a": {"x": 1, "y": 99, "z": 0}, "b": 3, "c": 4}


def test_merge_shallow():
    tool = JSONTool()
    base = {"a": {"x": 1}, "b": 2}
    override = {"a": {"z": 9}}
    result = tool.merge(base, override, deep=False)
    assert result["a"] == {"z": 9}
    assert result["b"] == 2


def test_diff_keys():
    tool = JSONTool()
    a = {"x": 1, "y": 2, "z": 3}
    b = {"y": 99, "z": 3, "w": 4}
    diff = tool.diff_keys(a, b)
    assert "x" in diff["removed"]
    assert "w" in diff["added"]
    assert "y" in diff["changed"]
    assert "z" not in diff["changed"]


def test_registry_key():
    assert "default" in JSON_TOOL_REGISTRY
    assert JSON_TOOL_REGISTRY["default"] is JSONTool
