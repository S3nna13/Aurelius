import argparse

import pytest

from aurelius_cli.metrics_commands import _run_demo, build_metrics_parser

pytest.importorskip("src.monitoring.sre_metrics")




def build_parser() -> argparse.ArgumentParser:
    """Helper: build top-level parser with metrics subparser attached."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    build_metrics_parser(sub)
    return parser


class TestMetricsParser:
    """Parsing and dispatch for `aurelius metrics` command."""

    def test_metrics_subcommand_exists(self):
        parser = build_parser()
        assert "metrics" in parser._subparsers._group_actions[0].choices

    def test_metrics_demo_parses_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["metrics", "demo"])
        assert args.command == "metrics"
        assert args.metrics_cmd == "demo"
        assert args.requests == 100
        assert args.error_rate == pytest.approx(0.05)
        assert args.latency_mean == pytest.approx(120.0)

    def test_metrics_demo_parses_custom_values(self):
        parser = build_parser()
        args = parser.parse_args([
            "metrics", "demo",
            "--requests", "500",
            "--error-rate", "0.1",
            "--latency-mean", "200",
            "--latency-std", "30"
        ])
        assert args.requests == 500
        assert args.error_rate == pytest.approx(0.1)
        assert args.latency_mean == pytest.approx(200.0)
        assert args.latency_std == pytest.approx(30.0)


class TestMetricsDemoOutput:
    """Integration-like checks for `_run_demo` output format."""

    def test_demo_returns_zero_and_prints_report(self, capsys):
        """_run_demo should exit 0 and print a formatted stats block."""
        ns = argparse.Namespace(
            metrics_cmd="demo",
            requests=20,
            error_rate=0.0,
            latency_mean=100.0,
            latency_std=10.0,
        )
        rc = _run_demo(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total requests : 20" in out
        assert "Error count    : 0" in out
        assert "Error rate     : 0.00%" in out
        assert "Latency p50" in out
        assert "Latency p90" in out
        assert "Latency p99" in out
        assert "Health score" in out
        assert "Saturation" in out

    def test_demo_handles_errors_and_nonzero_rate(self, capsys):
        ns = argparse.Namespace(
            metrics_cmd="demo",
            requests=100,
            error_rate=0.2,
            latency_mean=150.0,
            latency_std=50.0,
        )
        rc = _run_demo(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total requests : 100" in out
        # error count roughly around 20 but due to random may vary; just check line exists
        assert "Error count" in out
        assert "Health score" in out

    def test_demo_zero_requests_does_not_divide_by_zero(self, capsys):
        ns = argparse.Namespace(
            metrics_cmd="demo",
            requests=0,
            error_rate=0.0,
            latency_mean=0.0,
            latency_std=0.0,
        )
        rc = _run_demo(ns)
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total requests : 0" in out
        assert "Error rate     : N/A" in out
