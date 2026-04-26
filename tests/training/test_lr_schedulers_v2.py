"""
Tests for src/training/lr_schedulers_v2.py

15 tests covering all scheduler classes and the factory.
Tiny model: nn.Linear(8, 8), SGD lr=0.1.
Every test performs at least one forward/backward pass.
"""

import torch
import torch.nn as nn

from src.training.lr_schedulers_v2 import (
    CosineWithWarmup,
    CyclicLRScheduler,
    InverseSquareRootScheduler,
    LRSchedulerFactory,
    WarmupScheduler,
    WSDScheduler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_and_opt(lr: float = 0.1):
    """Return (model, optimizer) with a tiny nn.Linear(8, 8)."""
    model = nn.Linear(8, 8)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return model, opt


def _fwd_bwd(model, opt):
    """One forward + backward pass; does NOT call opt.step()."""
    x = torch.randn(2, 8)
    loss = model(x).sum()
    opt.zero_grad()
    loss.backward()


# ---------------------------------------------------------------------------
# WarmupScheduler tests
# ---------------------------------------------------------------------------


def test_warmup_lr_zero_at_step_zero():
    """LR should start at 0 (before any step)."""
    model, opt = _make_model_and_opt()
    sched = WarmupScheduler(opt, warmup_steps=10, base_lr=0.05)
    _fwd_bwd(model, opt)
    assert sched.get_lr() == 0.0, f"Expected 0.0, got {sched.get_lr()}"


def test_warmup_lr_equals_base_lr_at_warmup_steps():
    """After exactly warmup_steps steps, LR == base_lr."""
    model, opt = _make_model_and_opt()
    warmup = 8
    base = 0.05
    sched = WarmupScheduler(opt, warmup_steps=warmup, base_lr=base)
    for _ in range(warmup):
        _fwd_bwd(model, opt)
        sched.step()
    assert abs(sched.get_lr() - base) < 1e-9, f"Expected {base}, got {sched.get_lr()}"


def test_warmup_linear_increase():
    """LR must strictly increase during warmup and be evenly spaced."""
    model, opt = _make_model_and_opt()
    warmup = 6
    base = 0.06
    sched = WarmupScheduler(opt, warmup_steps=warmup, base_lr=base)
    lrs = []
    for _ in range(warmup):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())
    # strictly increasing
    for i in range(1, len(lrs)):
        assert lrs[i] > lrs[i - 1], "LR not monotonically increasing during warmup"
    # equal spacing
    diffs = [lrs[i] - lrs[i - 1] for i in range(1, len(lrs))]
    assert all(abs(d - diffs[0]) < 1e-9 for d in diffs), "LR spacing is not uniform"


def test_warmup_state_dict_round_trip():
    """state_dict / load_state_dict must restore scheduler state exactly."""
    model, opt = _make_model_and_opt()
    sched = WarmupScheduler(opt, warmup_steps=10, base_lr=0.05)
    for _ in range(4):
        _fwd_bwd(model, opt)
        sched.step()

    state = sched.state_dict()

    # create a fresh scheduler and load saved state
    model2, opt2 = _make_model_and_opt()
    sched2 = WarmupScheduler(opt2, warmup_steps=1, base_lr=0.01)  # different params
    sched2.load_state_dict(state)

    assert sched2._step == sched._step
    assert abs(sched2.get_lr() - sched.get_lr()) < 1e-9
    assert sched2.base_lr == sched.base_lr
    assert sched2.warmup_steps == sched.warmup_steps


# ---------------------------------------------------------------------------
# CosineWithWarmup tests
# ---------------------------------------------------------------------------


def test_cosine_warmup_lr_increases_during_warmup():
    """LR must strictly increase across warmup phase."""
    model, opt = _make_model_and_opt(lr=0.1)
    warmup = 5
    sched = CosineWithWarmup(opt, warmup_steps=warmup, total_steps=20)
    lrs = []
    for _ in range(warmup):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())
    for i in range(1, len(lrs)):
        assert lrs[i] > lrs[i - 1], "LR not increasing during warmup"


def test_cosine_warmup_lr_decreases_after_warmup():
    """LR must strictly decrease across cosine-decay phase."""
    model, opt = _make_model_and_opt(lr=0.1)
    warmup = 4
    total = 16
    sched = CosineWithWarmup(opt, warmup_steps=warmup, total_steps=total)
    # run through warmup
    for _ in range(warmup):
        _fwd_bwd(model, opt)
        sched.step()
    # now collect decay-phase LRs
    lrs = []
    for _ in range(total - warmup):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())
    for i in range(1, len(lrs)):
        assert lrs[i] < lrs[i - 1], f"LR not decreasing at decay step {i}"


def test_cosine_warmup_lr_at_end_approx_min_lr():
    """LR at total_steps should be approximately min_lr."""
    model, opt = _make_model_and_opt(lr=0.1)
    warmup = 4
    total = 16
    min_ratio = 0.1
    sched = CosineWithWarmup(opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=min_ratio)
    for _ in range(total):
        _fwd_bwd(model, opt)
        sched.step()
    expected_min = 0.1 * min_ratio
    assert abs(sched.get_lr() - expected_min) < 1e-6, (
        f"Expected ~{expected_min}, got {sched.get_lr()}"
    )


def test_cosine_warmup_n_cycles_resets_midway():
    """With n_cycles=2, LR should reset (rise) at the midpoint of the decay phase."""
    model, opt = _make_model_and_opt(lr=0.1)
    warmup = 2
    total = 12
    sched = CosineWithWarmup(
        opt, warmup_steps=warmup, total_steps=total, min_lr_ratio=0.1, n_cycles=2
    )
    decay_steps = total - warmup  # 10
    half = decay_steps // 2  # 5 — end of first cycle

    # run warmup
    for _ in range(warmup):
        _fwd_bwd(model, opt)
        sched.step()

    # collect decay LRs
    lrs = []
    for _ in range(decay_steps):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())

    # LR at start of second cycle (index `half`) should be higher than at end of
    # first cycle (index `half - 1`)
    assert lrs[half] > lrs[half - 1], (
        f"Expected LR reset at cycle boundary; "
        f"lrs[{half - 1}]={lrs[half - 1]:.6f}, lrs[{half}]={lrs[half]:.6f}"
    )


# ---------------------------------------------------------------------------
# WSDScheduler tests
# ---------------------------------------------------------------------------


def test_wsd_current_phase_correct():
    """current_phase() must return the right string at every step boundary."""
    model, opt = _make_model_and_opt()
    sched = WSDScheduler(
        opt, warmup_steps=3, stable_steps=4, decay_steps=3, peak_lr=0.05, min_lr=0.0
    )
    # step 0 — before any step — still warmup
    assert sched.current_phase() == "warmup"

    for _ in range(3):
        _fwd_bwd(model, opt)
        sched.step()
    assert sched.current_phase() == "warmup"  # at step 3

    _fwd_bwd(model, opt)
    sched.step()  # step 4
    assert sched.current_phase() == "stable"

    for _ in range(3):
        _fwd_bwd(model, opt)
        sched.step()
    assert sched.current_phase() == "stable"  # step 7

    _fwd_bwd(model, opt)
    sched.step()  # step 8 → decay
    assert sched.current_phase() == "decay"


def test_wsd_stable_phase_constant_lr():
    """LR must be exactly peak_lr throughout the stable phase."""
    model, opt = _make_model_and_opt()
    peak = 0.05
    sched = WSDScheduler(
        opt, warmup_steps=3, stable_steps=5, decay_steps=3, peak_lr=peak, min_lr=0.0
    )
    # skip warmup
    for _ in range(3):
        _fwd_bwd(model, opt)
        sched.step()
    # stable phase
    for _ in range(5):
        _fwd_bwd(model, opt)
        sched.step()
        assert abs(opt.param_groups[0]["lr"] - peak) < 1e-9, (
            f"LR deviated from peak during stable: {opt.param_groups[0]['lr']}"
        )


def test_wsd_lr_at_end_of_decay_approx_min_lr():
    """LR at the end of the decay phase must be ≈ min_lr."""
    model, opt = _make_model_and_opt()
    min_lr = 1e-4
    warmup, stable, decay = 3, 4, 5
    sched = WSDScheduler(
        opt,
        warmup_steps=warmup,
        stable_steps=stable,
        decay_steps=decay,
        peak_lr=0.05,
        min_lr=min_lr,
    )
    total = warmup + stable + decay
    for _ in range(total):
        _fwd_bwd(model, opt)
        sched.step()
    assert abs(sched._current_lr - min_lr) < 1e-6, f"Expected ~{min_lr}, got {sched._current_lr}"


# ---------------------------------------------------------------------------
# InverseSquareRootScheduler tests
# ---------------------------------------------------------------------------


def test_inv_sqrt_lr_peaks_around_warmup_steps():
    """LR should peak at or very near warmup_steps."""
    model, opt = _make_model_and_opt()
    warmup = 10
    sched = InverseSquareRootScheduler(opt, d_model=16, warmup_steps=warmup)
    lrs = []
    for _ in range(warmup * 3):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())
    peak_step = lrs.index(max(lrs)) + 1  # 1-indexed
    assert peak_step <= warmup + 2, (
        f"Peak LR should be near warmup_steps={warmup}, got peak at step {peak_step}"
    )


def test_inv_sqrt_lr_decreases_after_warmup():
    """After warmup, LR must strictly decrease (inverse square-root decay)."""
    model, opt = _make_model_and_opt()
    warmup = 8
    sched = InverseSquareRootScheduler(opt, d_model=16, warmup_steps=warmup)
    # run through warmup
    for _ in range(warmup):
        _fwd_bwd(model, opt)
        sched.step()
    # collect a few post-warmup LRs
    lrs = []
    for _ in range(6):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())
    for i in range(1, len(lrs)):
        assert lrs[i] < lrs[i - 1], f"LR not decreasing at post-warmup step {i}"


# ---------------------------------------------------------------------------
# CyclicLRScheduler tests
# ---------------------------------------------------------------------------


def test_cyclic_lr_oscillates_between_base_and_max():
    """LR must stay in [base_lr, max_lr] and reach max_lr within one cycle."""
    model, opt = _make_model_and_opt()
    base, maximum = 0.01, 0.1
    step_size = 4
    sched = CyclicLRScheduler(opt, base_lr=base, max_lr=maximum, step_size=step_size)
    lrs = []
    for _ in range(step_size * 4):
        _fwd_bwd(model, opt)
        sched.step()
        lrs.append(sched.get_lr())
    assert all(base - 1e-9 <= lr <= maximum + 1e-9 for lr in lrs), (
        "Some LR values are outside [base_lr, max_lr]"
    )
    assert max(lrs) >= maximum - 1e-6, "LR never reached max_lr"


def test_cyclic_lr_triangular2_halves_amplitude():
    """In triangular2 mode, peak LR should halve each cycle."""
    model, opt = _make_model_and_opt()
    base, maximum = 0.01, 0.1
    step_size = 4
    sched = CyclicLRScheduler(
        opt, base_lr=base, max_lr=maximum, step_size=step_size, mode="triangular2"
    )
    # collect peak of cycle 1 and cycle 2
    cycle_len = step_size * 2
    lrs_c1, lrs_c2 = [], []
    for i in range(cycle_len * 2):
        _fwd_bwd(model, opt)
        sched.step()
        if i < cycle_len:
            lrs_c1.append(sched.get_lr())
        else:
            lrs_c2.append(sched.get_lr())
    peak1 = max(lrs_c1)
    peak2 = max(lrs_c2)
    amplitude1 = peak1 - base
    amplitude2 = peak2 - base
    assert abs(amplitude2 - amplitude1 / 2.0) < 1e-5, (
        f"Amplitude did not halve: cycle1={amplitude1:.6f}, cycle2={amplitude2:.6f}"
    )


# ---------------------------------------------------------------------------
# LRSchedulerFactory tests
# ---------------------------------------------------------------------------


def test_factory_creates_correct_types():
    """create() must return the right scheduler class for every registered name."""
    factory = LRSchedulerFactory()

    model, opt = _make_model_and_opt(lr=0.05)
    s = factory.create("warmup", opt, warmup_steps=5, base_lr=0.05)
    assert isinstance(s, WarmupScheduler)

    model, opt = _make_model_and_opt(lr=0.05)
    s = factory.create("cosine_warmup", opt, warmup_steps=5, total_steps=20)
    assert isinstance(s, CosineWithWarmup)

    model, opt = _make_model_and_opt(lr=0.05)
    s = factory.create(
        "wsd", opt, warmup_steps=3, stable_steps=4, decay_steps=3, peak_lr=0.05, min_lr=0.0
    )
    assert isinstance(s, WSDScheduler)

    model, opt = _make_model_and_opt(lr=0.05)
    s = factory.create("inv_sqrt", opt, d_model=16, warmup_steps=10)
    assert isinstance(s, InverseSquareRootScheduler)

    model, opt = _make_model_and_opt(lr=0.05)
    s = factory.create("cyclic", opt, base_lr=0.01, max_lr=0.1, step_size=4)
    assert isinstance(s, CyclicLRScheduler)

    # forward/backward to satisfy test requirement
    _fwd_bwd(model, opt)
    s.step()


def test_factory_plot_schedule_length_and_non_negative():
    """plot_schedule must return exactly n_steps values, all >= 0."""
    factory = LRSchedulerFactory()
    model, opt = _make_model_and_opt(lr=0.05)
    sched = factory.create("cosine_warmup", opt, warmup_steps=3, total_steps=10)

    n = 10
    lrs = factory.plot_schedule(sched, n)
    assert len(lrs) == n, f"Expected {n} LR values, got {len(lrs)}"
    assert all(lr >= 0 for lr in lrs), f"Negative LR found: {lrs}"

    # perform a forward/backward pass to satisfy the requirement
    _fwd_bwd(model, opt)
