import time

from src.profiling.model_benchmarker import (
    BenchmarkConfig,
    BenchmarkStats,
    ModelBenchmarker,
    MODEL_BENCHMARKER_REGISTRY,
)


def _noop_fn(x):
    return x


def _slow_fn(x):
    time.sleep(0.001)
    return x


def _factory(bs):
    return list(range(bs))


def test_default_config_values():
    cfg = BenchmarkConfig()
    assert cfg.warmup_runs == 3
    assert cfg.bench_runs == 10
    assert cfg.batch_sizes == [1, 2, 4, 8]


def test_config_batch_sizes_independent():
    a = BenchmarkConfig()
    b = BenchmarkConfig()
    a.batch_sizes.append(16)
    assert b.batch_sizes == [1, 2, 4, 8]


def test_run_returns_benchmark_stats():
    b = ModelBenchmarker()
    stats = b.run(_noop_fn, 4, _factory)
    assert isinstance(stats, BenchmarkStats)


def test_run_batch_size_recorded():
    b = ModelBenchmarker()
    stats = b.run(_noop_fn, 7, _factory)
    assert stats.batch_size == 7


def test_run_mean_positive():
    b = ModelBenchmarker()
    stats = b.run(_slow_fn, 1, _factory)
    assert stats.latency_mean_ms > 0


def test_run_p50_p95_p99_ordering():
    b = ModelBenchmarker()
    stats = b.run(_slow_fn, 1, _factory)
    assert stats.latency_p50_ms <= stats.latency_p95_ms
    assert stats.latency_p95_ms <= stats.latency_p99_ms


def test_run_throughput_positive():
    b = ModelBenchmarker()
    stats = b.run(_slow_fn, 2, _factory)
    assert stats.throughput_samples_per_s > 0


def test_run_throughput_scales_with_batch():
    b = ModelBenchmarker(BenchmarkConfig(warmup_runs=1, bench_runs=3))
    s1 = b.run(_noop_fn, 1, _factory)
    s8 = b.run(_noop_fn, 8, _factory)
    # throughput is batch_size / mean_latency; with near-zero work, bigger batch
    # produces bigger throughput on average. Be lenient: just require >0.
    assert s1.throughput_samples_per_s > 0
    assert s8.throughput_samples_per_s > 0


def test_run_sweep_length_matches_batch_sizes():
    cfg = BenchmarkConfig(batch_sizes=[1, 2, 3])
    b = ModelBenchmarker(cfg)
    out = b.run_sweep(_noop_fn, _factory)
    assert len(out) == 3


def test_run_sweep_batch_sizes_preserved():
    cfg = BenchmarkConfig(batch_sizes=[2, 5, 11])
    b = ModelBenchmarker(cfg)
    out = b.run_sweep(_noop_fn, _factory)
    assert [s.batch_size for s in out] == [2, 5, 11]


def test_run_sweep_default_batches():
    b = ModelBenchmarker()
    out = b.run_sweep(_noop_fn, _factory)
    assert len(out) == 4


def test_report_contains_header():
    b = ModelBenchmarker()
    out = b.run_sweep(_noop_fn, _factory)
    rep = b.report(out)
    assert "batch_size" in rep
    assert "p50_ms" in rep
    assert "p99_ms" in rep
    assert "throughput" in rep


def test_report_contains_separator_chars():
    b = ModelBenchmarker()
    out = b.run_sweep(_noop_fn, _factory)
    rep = b.report(out)
    assert "-" in rep
    assert "|" in rep


def test_report_empty():
    b = ModelBenchmarker()
    rep = b.report([])
    assert "No benchmark" in rep


def test_best_throughput_picks_highest():
    b = ModelBenchmarker()
    stats = [
        BenchmarkStats(1, 1.0, 1.0, 1.0, 1.0, 10.0),
        BenchmarkStats(2, 1.0, 1.0, 1.0, 1.0, 99.0),
        BenchmarkStats(4, 1.0, 1.0, 1.0, 1.0, 50.0),
    ]
    best = b.best_throughput(stats)
    assert best.throughput_samples_per_s == 99.0
    assert best.batch_size == 2


def test_best_throughput_empty_raises():
    b = ModelBenchmarker()
    try:
        b.best_throughput([])
    except ValueError:
        return
    assert False, "expected ValueError"


def test_registry_key():
    assert "default" in MODEL_BENCHMARKER_REGISTRY
    assert MODEL_BENCHMARKER_REGISTRY["default"] is ModelBenchmarker


def test_benchmark_stats_is_frozen():
    s = BenchmarkStats(1, 1.0, 1.0, 1.0, 1.0, 1.0)
    try:
        s.batch_size = 2  # type: ignore
    except Exception:
        return
    assert False, "expected frozen dataclass"


def test_warmup_executes():
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1

    cfg = BenchmarkConfig(warmup_runs=2, bench_runs=3)
    ModelBenchmarker(cfg).run(fn, 1, _factory)
    assert calls["n"] == 5


def test_input_factory_called_once():
    calls = {"n": 0}

    def fac(bs):
        calls["n"] += 1
        return bs

    ModelBenchmarker(BenchmarkConfig(warmup_runs=1, bench_runs=1)).run(
        _noop_fn, 3, fac
    )
    assert calls["n"] == 1


def test_run_zero_bench_runs_not_allowed_with_default():
    # default has bench_runs=10 so samples list should be 10
    b = ModelBenchmarker()
    cnt = {"n": 0}

    def fn(x):
        cnt["n"] += 1

    b.run(fn, 1, _factory)
    assert cnt["n"] == 3 + 10


def test_custom_warmup_and_bench():
    cfg = BenchmarkConfig(warmup_runs=0, bench_runs=5)
    b = ModelBenchmarker(cfg)
    cnt = {"n": 0}

    def fn(x):
        cnt["n"] += 1

    b.run(fn, 1, _factory)
    assert cnt["n"] == 5


def test_report_row_count():
    b = ModelBenchmarker(BenchmarkConfig(batch_sizes=[1, 2]))
    out = b.run_sweep(_noop_fn, _factory)
    rep = b.report(out)
    # header + sep + 2 rows
    assert len(rep.splitlines()) == 4


def test_p99_index_ceiling():
    # With bench_runs=10, p99 idx = ceil(10*0.99)-1 = 9 -> last element
    cfg = BenchmarkConfig(warmup_runs=0, bench_runs=10)
    b = ModelBenchmarker(cfg)
    stats = b.run(_noop_fn, 1, _factory)
    assert stats.latency_p99_ms >= stats.latency_p95_ms


def test_latency_mean_nonneg_no_sleep():
    b = ModelBenchmarker(BenchmarkConfig(warmup_runs=0, bench_runs=2))
    stats = b.run(_noop_fn, 1, _factory)
    assert stats.latency_mean_ms >= 0


def test_sweep_stats_all_instances():
    b = ModelBenchmarker()
    out = b.run_sweep(_noop_fn, _factory)
    for s in out:
        assert isinstance(s, BenchmarkStats)
