"""Tests for src/ui/context_menu.py — ~50 tests."""

from __future__ import annotations

import pytest

from src.ui.context_menu import (
    MenuItemKind,
    MenuItem,
    ContextMenu,
    ContextMenuRegistry,
    CONTEXT_MENU_REGISTRY,
)


# ---------------------------------------------------------------------------
# MenuItemKind enum
# ---------------------------------------------------------------------------


class TestMenuItemKind:
    def test_action_value(self):
        assert MenuItemKind.ACTION == "action"

    def test_separator_value(self):
        assert MenuItemKind.SEPARATOR == "separator"

    def test_submenu_value(self):
        assert MenuItemKind.SUBMENU == "submenu"

    def test_is_str_subclass(self):
        assert isinstance(MenuItemKind.ACTION, str)

    def test_enum_members(self):
        members = {k.value for k in MenuItemKind}
        assert members == {"action", "separator", "submenu"}


# ---------------------------------------------------------------------------
# MenuItem dataclass
# ---------------------------------------------------------------------------


class TestMenuItem:
    def test_label_stored(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION)
        assert item.label == "Copy"

    def test_kind_stored(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION)
        assert item.kind == MenuItemKind.ACTION

    def test_default_shortcut_empty(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION)
        assert item.shortcut == ""

    def test_default_enabled_true(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION)
        assert item.enabled is True

    def test_default_action_id_empty(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION)
        assert item.action_id == ""

    def test_custom_shortcut(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION, shortcut="Ctrl+C")
        assert item.shortcut == "Ctrl+C"

    def test_disabled_item(self):
        item = MenuItem(label="Paste", kind=MenuItemKind.ACTION, enabled=False)
        assert item.enabled is False

    def test_action_id_set(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION, action_id="copy")
        assert item.action_id == "copy"

    def test_separator_kind(self):
        item = MenuItem(label="", kind=MenuItemKind.SEPARATOR)
        assert item.kind == MenuItemKind.SEPARATOR

    def test_submenu_kind(self):
        item = MenuItem(label="More", kind=MenuItemKind.SUBMENU)
        assert item.kind == MenuItemKind.SUBMENU


# ---------------------------------------------------------------------------
# ContextMenu
# ---------------------------------------------------------------------------


class TestContextMenu:
    def setup_method(self):
        self.menu = ContextMenu(title="Test Menu")

    def test_add_item_increases_count(self):
        self.menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION))
        items = self.menu.enabled_items()
        assert len(items) == 1

    def test_add_separator_appended(self):
        self.menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION))
        self.menu.add_separator()
        self.menu.add_item(MenuItem(label="B", kind=MenuItemKind.ACTION))
        # separator is at index 1
        result = self.menu.select(1)
        assert result is None  # separator is not selectable

    def test_render_non_empty(self):
        self.menu.add_item(MenuItem(label="Copy", kind=MenuItemKind.ACTION))
        assert len(self.menu.render()) > 0

    def test_render_contains_label(self):
        self.menu.add_item(MenuItem(label="Copy", kind=MenuItemKind.ACTION))
        assert "Copy" in self.menu.render()

    def test_render_contains_title(self):
        result = self.menu.render()
        assert "Test Menu" in result

    def test_render_no_title(self):
        menu = ContextMenu()
        menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION))
        # Should not raise
        result = menu.render()
        assert "A" in result

    def test_select_valid_action(self):
        item = MenuItem(label="Copy", kind=MenuItemKind.ACTION, action_id="copy")
        self.menu.add_item(item)
        selected = self.menu.select(0)
        assert selected is item

    def test_select_separator_returns_none(self):
        self.menu.add_separator()
        assert self.menu.select(0) is None

    def test_select_disabled_returns_none(self):
        self.menu.add_item(MenuItem(label="Disabled", kind=MenuItemKind.ACTION, enabled=False))
        assert self.menu.select(0) is None

    def test_select_out_of_bounds_returns_none(self):
        assert self.menu.select(99) is None

    def test_select_negative_returns_none(self):
        self.menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION))
        assert self.menu.select(-1) is None

    def test_enabled_items_excludes_disabled(self):
        self.menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION, enabled=True))
        self.menu.add_item(MenuItem(label="B", kind=MenuItemKind.ACTION, enabled=False))
        pairs = self.menu.enabled_items()
        labels = [item.label for _, item in pairs]
        assert "A" in labels
        assert "B" not in labels

    def test_enabled_items_excludes_separators(self):
        self.menu.add_separator()
        self.menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION))
        pairs = self.menu.enabled_items()
        assert len(pairs) == 1

    def test_enabled_items_preserves_original_index(self):
        self.menu.add_separator()  # index 0
        self.menu.add_item(MenuItem(label="A", kind=MenuItemKind.ACTION))  # index 1
        pairs = self.menu.enabled_items()
        assert pairs[0][0] == 1  # original index

    def test_add_multiple_items(self):
        for label in ["A", "B", "C"]:
            self.menu.add_item(MenuItem(label=label, kind=MenuItemKind.ACTION))
        assert len(self.menu.enabled_items()) == 3

    def test_render_shows_shortcut(self):
        self.menu.add_item(MenuItem(label="Copy", kind=MenuItemKind.ACTION, shortcut="Ctrl+C"))
        assert "Ctrl+C" in self.menu.render()

    def test_render_returns_string(self):
        assert isinstance(self.menu.render(), str)


# ---------------------------------------------------------------------------
# ContextMenuRegistry
# ---------------------------------------------------------------------------


class TestContextMenuRegistry:
    def setup_method(self):
        self.registry = ContextMenuRegistry()

    def test_register_and_get(self):
        menu = ContextMenu(title="My Menu")
        self.registry.register("my", menu)
        assert self.registry.get("my") is menu

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            self.registry.get("ghost")

    def test_list_menus_empty(self):
        assert self.registry.list_menus() == []

    def test_list_menus_sorted(self):
        self.registry.register("z_menu", ContextMenu())
        self.registry.register("a_menu", ContextMenu())
        names = self.registry.list_menus()
        assert names == sorted(names)

    def test_overwrite(self):
        m1 = ContextMenu(title="One")
        m2 = ContextMenu(title="Two")
        self.registry.register("key", m1)
        self.registry.register("key", m2)
        assert self.registry.get("key") is m2


# ---------------------------------------------------------------------------
# CONTEXT_MENU_REGISTRY — pre-registered defaults
# ---------------------------------------------------------------------------


class TestDefaultContextMenuRegistry:
    def test_editor_registered(self):
        menu = CONTEXT_MENU_REGISTRY.get("editor")
        assert menu is not None

    def test_session_registered(self):
        menu = CONTEXT_MENU_REGISTRY.get("session")
        assert menu is not None

    def test_both_in_list(self):
        names = CONTEXT_MENU_REGISTRY.list_menus()
        assert "editor" in names
        assert "session" in names

    def test_editor_has_copy(self):
        menu = CONTEXT_MENU_REGISTRY.get("editor")
        action_labels = [item.label for _, item in menu.enabled_items()]
        assert "Copy" in action_labels

    def test_editor_has_paste(self):
        menu = CONTEXT_MENU_REGISTRY.get("editor")
        action_labels = [item.label for _, item in menu.enabled_items()]
        assert "Paste" in action_labels

    def test_editor_has_select_all(self):
        menu = CONTEXT_MENU_REGISTRY.get("editor")
        action_labels = [item.label for _, item in menu.enabled_items()]
        assert "Select All" in action_labels

    def test_editor_has_three_action_items(self):
        menu = CONTEXT_MENU_REGISTRY.get("editor")
        assert len(menu.enabled_items()) == 3

    def test_session_has_new_session(self):
        menu = CONTEXT_MENU_REGISTRY.get("session")
        action_labels = [item.label for _, item in menu.enabled_items()]
        assert "New Session" in action_labels

    def test_session_has_close_session(self):
        menu = CONTEXT_MENU_REGISTRY.get("session")
        action_labels = [item.label for _, item in menu.enabled_items()]
        assert "Close Session" in action_labels

    def test_session_has_two_action_items(self):
        menu = CONTEXT_MENU_REGISTRY.get("session")
        assert len(menu.enabled_items()) == 2

    def test_registry_is_correct_type(self):
        assert isinstance(CONTEXT_MENU_REGISTRY, ContextMenuRegistry)
