"""
Tests for src/workflow/workflow_visualizer.py
≥28 test cases covering NodeStyle, VisualizerConfig, WorkflowVisualizer, and REGISTRY.
"""

import dataclasses
import unittest

from src.workflow.workflow_visualizer import (
    WORKFLOW_VISUALIZER_REGISTRY,
    NodeStyle,
    VisualizerConfig,
    WorkflowVisualizer,
    _FILLCOLOR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_viz(indent: int = 2, show_metadata: bool = False) -> WorkflowVisualizer:
    return WorkflowVisualizer(VisualizerConfig(indent=indent, show_metadata=show_metadata))


SAMPLE_NODES = ["A", "B", "C"]
SAMPLE_EDGES = [("A", "B"), ("A", "C")]


# ---------------------------------------------------------------------------
# to_dot() tests
# ---------------------------------------------------------------------------

class TestToDot(unittest.TestCase):

    def setUp(self):
        self.viz = make_viz()

    def test_contains_digraph_keyword(self):
        dot = self.viz.to_dot(SAMPLE_NODES, SAMPLE_EDGES)
        self.assertIn("digraph", dot)

    def test_contains_all_nodes(self):
        dot = self.viz.to_dot(SAMPLE_NODES, SAMPLE_EDGES)
        for node in SAMPLE_NODES:
            self.assertIn(node, dot)

    def test_default_fillcolor_white(self):
        dot = self.viz.to_dot(["X"], [], node_styles={"X": NodeStyle.DEFAULT})
        self.assertIn("fillcolor=white", dot)

    def test_highlighted_fillcolor_yellow(self):
        dot = self.viz.to_dot(["X"], [], node_styles={"X": NodeStyle.HIGHLIGHTED})
        self.assertIn("fillcolor=yellow", dot)

    def test_failed_fillcolor_red(self):
        dot = self.viz.to_dot(["X"], [], node_styles={"X": NodeStyle.FAILED})
        self.assertIn("fillcolor=red", dot)

    def test_completed_fillcolor_green(self):
        dot = self.viz.to_dot(["X"], [], node_styles={"X": NodeStyle.COMPLETED})
        self.assertIn("fillcolor=green", dot)

    def test_node_has_label(self):
        dot = self.viz.to_dot(["MyNode"], [])
        self.assertIn('label="MyNode"', dot)

    def test_node_has_style_filled(self):
        dot = self.viz.to_dot(["N"], [])
        self.assertIn("style=filled", dot)

    def test_edge_arrow_present(self):
        dot = self.viz.to_dot(["A", "B"], [("A", "B")])
        self.assertIn("->", dot)

    def test_edge_correct_direction(self):
        dot = self.viz.to_dot(["A", "B"], [("A", "B")])
        self.assertIn('"A" -> "B"', dot)

    def test_empty_graph_no_crash(self):
        dot = self.viz.to_dot([], [])
        self.assertIn("digraph", dot)

    def test_no_node_styles_defaults_to_white(self):
        dot = self.viz.to_dot(["Z"], [], node_styles=None)
        self.assertIn("fillcolor=white", dot)

    def test_multiple_edges(self):
        dot = self.viz.to_dot(["A", "B", "C"], [("A", "B"), ("B", "C")])
        self.assertIn('"A" -> "B"', dot)
        self.assertIn('"B" -> "C"', dot)

    def test_closing_brace_present(self):
        dot = self.viz.to_dot([], [])
        self.assertIn("}", dot)


# ---------------------------------------------------------------------------
# to_ascii() tests
# ---------------------------------------------------------------------------

class TestToAscii(unittest.TestCase):

    def setUp(self):
        self.viz = make_viz(indent=2)

    def test_root_printed(self):
        output = self.viz.to_ascii(["root"], [])
        self.assertIn("root", output)

    def test_child_indented(self):
        output = self.viz.to_ascii(["A", "B"], [("A", "B")])
        lines = output.splitlines()
        # "B" should be indented (start with spaces)
        b_line = next(l for l in lines if "B" in l)
        self.assertTrue(b_line.startswith(" "))

    def test_root_not_indented(self):
        output = self.viz.to_ascii(["A", "B"], [("A", "B")])
        lines = output.splitlines()
        a_line = next(l for l in lines if l.strip() == "A")
        self.assertFalse(a_line.startswith(" "))

    def test_indent_respects_config(self):
        viz4 = make_viz(indent=4)
        output = viz4.to_ascii(["A", "B"], [("A", "B")])
        lines = output.splitlines()
        b_line = next(l for l in lines if "B" in l)
        self.assertTrue(b_line.startswith("    "))

    def test_empty_graph_returns_empty_string(self):
        output = self.viz.to_ascii([], [])
        self.assertEqual(output, "")

    def test_all_nodes_present(self):
        output = self.viz.to_ascii(SAMPLE_NODES, SAMPLE_EDGES)
        for node in SAMPLE_NODES:
            self.assertIn(node, output)

    def test_linear_chain_ordering(self):
        nodes = ["A", "B", "C"]
        edges = [("A", "B"), ("B", "C")]
        output = self.viz.to_ascii(nodes, edges)
        lines = [l.strip() for l in output.splitlines()]
        self.assertEqual(lines.index("A"), 0)
        self.assertLess(lines.index("A"), lines.index("B"))
        self.assertLess(lines.index("B"), lines.index("C"))

    def test_single_node(self):
        output = self.viz.to_ascii(["solo"], [])
        self.assertIn("solo", output)


# ---------------------------------------------------------------------------
# to_mermaid() tests
# ---------------------------------------------------------------------------

class TestToMermaid(unittest.TestCase):

    def setUp(self):
        self.viz = make_viz()

    def test_starts_with_flowchart_lr(self):
        mermaid = self.viz.to_mermaid(SAMPLE_NODES, SAMPLE_EDGES)
        self.assertTrue(mermaid.startswith("flowchart LR"))

    def test_contains_arrow_edges(self):
        mermaid = self.viz.to_mermaid(["A", "B"], [("A", "B")])
        self.assertIn("-->", mermaid)

    def test_node_format(self):
        mermaid = self.viz.to_mermaid(["myNode"], [])
        self.assertIn("myNode[myNode]", mermaid)

    def test_edge_format(self):
        mermaid = self.viz.to_mermaid(["A", "B"], [("A", "B")])
        self.assertIn("A --> B", mermaid)

    def test_all_nodes_in_output(self):
        mermaid = self.viz.to_mermaid(SAMPLE_NODES, SAMPLE_EDGES)
        for node in SAMPLE_NODES:
            self.assertIn(node, mermaid)

    def test_empty_graph_no_crash(self):
        mermaid = self.viz.to_mermaid([], [])
        self.assertIn("flowchart LR", mermaid)

    def test_multiple_edges(self):
        mermaid = self.viz.to_mermaid(["A", "B", "C"], [("A", "B"), ("B", "C")])
        self.assertIn("A --> B", mermaid)
        self.assertIn("B --> C", mermaid)


# ---------------------------------------------------------------------------
# VisualizerConfig frozen dataclass test
# ---------------------------------------------------------------------------

class TestVisualizerConfig(unittest.TestCase):

    def test_config_is_frozen(self):
        cfg = VisualizerConfig()
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.indent = 10  # type: ignore[misc]

    def test_config_defaults(self):
        cfg = VisualizerConfig()
        self.assertEqual(cfg.indent, 2)
        self.assertFalse(cfg.show_metadata)

    def test_config_custom_values(self):
        cfg = VisualizerConfig(indent=4, show_metadata=True)
        self.assertEqual(cfg.indent, 4)
        self.assertTrue(cfg.show_metadata)


# ---------------------------------------------------------------------------
# NodeStyle tests
# ---------------------------------------------------------------------------

class TestNodeStyle(unittest.TestCase):

    def test_all_styles_in_fillcolor_map(self):
        for style in NodeStyle:
            self.assertIn(style, _FILLCOLOR)

    def test_default_color(self):
        self.assertEqual(_FILLCOLOR[NodeStyle.DEFAULT], "white")

    def test_failed_color(self):
        self.assertEqual(_FILLCOLOR[NodeStyle.FAILED], "red")

    def test_completed_color(self):
        self.assertEqual(_FILLCOLOR[NodeStyle.COMPLETED], "green")

    def test_highlighted_color(self):
        self.assertEqual(_FILLCOLOR[NodeStyle.HIGHLIGHTED], "yellow")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):

    def test_registry_has_default_key(self):
        self.assertIn("default", WORKFLOW_VISUALIZER_REGISTRY)

    def test_registry_default_is_visualizer_class(self):
        self.assertIs(WORKFLOW_VISUALIZER_REGISTRY["default"], WorkflowVisualizer)

    def test_registry_value_is_instantiable(self):
        cls = WORKFLOW_VISUALIZER_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, WorkflowVisualizer)


if __name__ == "__main__":
    unittest.main()
