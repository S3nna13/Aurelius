import pytest

from src.training.early_stopping import EarlyStopping, EarlyStoppingConfig, Mode, ReduceOnPlateau


def test_no_stop_while_improving():
    es = EarlyStopping(EarlyStoppingConfig(patience=3))
    for loss in [1.0, 0.9, 0.8, 0.7]:
        assert not es.step(loss)
    assert not es.stopped


def test_stops_after_patience():
    es = EarlyStopping(EarlyStoppingConfig(patience=3, min_delta=0.0))
    es.step(1.0)  # set best
    for _ in range(3):
        result = es.step(1.0)  # no improvement
    assert result is True
    assert es.stopped


def test_reset_on_improvement():
    es = EarlyStopping(EarlyStoppingConfig(patience=3, min_delta=0.0))
    es.step(1.0)
    es.step(1.0)  # 1 bad step
    es.step(0.5)  # improvement — reset counter
    es.step(1.0)  # 1 bad step again
    assert not es.stopped


def test_max_mode():
    es = EarlyStopping(EarlyStoppingConfig(patience=2, mode=Mode.MAX, min_delta=0.0))
    es.step(0.5)
    es.step(0.4)  # worse
    result = es.step(0.4)  # still no improvement
    assert result is True


def test_history_recorded():
    es = EarlyStopping()
    for v in [1.0, 0.9, 0.8]:
        es.step(v)
    assert es.history == [1.0, 0.9, 0.8]


def test_reduce_on_plateau_normal():
    rop = ReduceOnPlateau(EarlyStoppingConfig(reduce_patience=3))
    scale = rop.step(1.0)  # first step
    assert scale == 1.0


def test_reduce_on_plateau_triggers():
    rop = ReduceOnPlateau(EarlyStoppingConfig(reduce_patience=2, reduce_factor=0.5, min_delta=0.0))
    rop.step(1.0)  # set best
    rop.step(1.0)  # 1 bad
    scale = rop.step(1.0)  # 2 bad → trigger
    assert scale == pytest.approx(0.5)


def test_reduce_resets_after_trigger():
    rop = ReduceOnPlateau(EarlyStoppingConfig(reduce_patience=2, reduce_factor=0.5, min_delta=0.0))
    rop.step(1.0)
    rop.step(1.0)
    rop.step(1.0)  # triggers, resets counter
    scale = rop.step(1.0)  # counter=1 now, not yet triggered
    assert scale == 1.0
