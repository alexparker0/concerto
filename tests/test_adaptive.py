"""Milestone-3 adaptive controller tests: signal-adaptive stereo_beta."""

import numpy as np
import pytest

from concerto.adaptive import AdaptiveBetaConfig, block_directness, compute_beta_schedule
from concerto.audio.buffer import measure
from concerto.audio.layouts import INTERNAL_7_1
from concerto.dsp.upmix import UpmixRenderer, load_config
from concerto.native import NativeUpmixRenderer, library_available

SR = 48000
N = SR * 4

BCFG = AdaptiveBetaConfig()


def _render_with_schedule(src, schedule, native=False):
    left, right = src
    cfg = load_config("configs/default.toml")
    if native:
        r = NativeUpmixRenderer(SR, cfg)
    else:
        r = UpmixRenderer(SR, cfg)
    return r.render(left, right, beta_schedule=schedule)


def _surround(out):
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    return out.channels[idx["SL"]], out.channels[idx["SR"]]


def test_directness_hard_panned_is_high():
    rng = np.random.default_rng(3)
    noise = (0.3 * rng.standard_normal(N)).astype(np.float32)
    d = block_directness(noise, np.zeros(N, dtype=np.float32), SR, BCFG)
    assert d > 0.9, f"hard-panned directness {d:.3f}"


def test_directness_diffuse_is_low():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(N) * 0.2
    b = rng.standard_normal(N) * 0.2
    d = block_directness(a.astype(np.float32), b.astype(np.float32), SR, BCFG)
    assert d < 0.3, f"diffuse directness {d:.3f}"


def test_directness_dual_mono_has_no_side_energy():
    # Dual mono: side content is zero; the metric must not claim direction.
    t = np.arange(N) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    d = block_directness(x, x.copy(), SR, BCFG)
    assert d == 0.0  # no side energy -> no steering evidence


def test_beta_schedule_bounds_and_mapping():
    rng = np.random.default_rng(5)
    noise = (0.3 * rng.standard_normal(N)).astype(np.float32)
    # Hard-panned -> beta near minimum (strong directional extraction).
    s_panned = compute_beta_schedule(noise, np.zeros(N, dtype=np.float32), SR, BCFG)
    assert s_panned[-1] < BCFG.beta_min + 0.1, s_panned[-1]
    # Diffuse -> beta near maximum (envelopment, no front-lock).
    s_diffuse = compute_beta_schedule(
        (rng.standard_normal(N) * 0.2).astype(np.float32),
        (rng.standard_normal(N) * 0.2).astype(np.float32), SR, BCFG)
    assert s_diffuse[-1] > BCFG.beta_max - 0.1, s_diffuse[-1]
    # Dual mono -> beta at maximum (side energy zero: surrounds carry little).
    t = np.arange(N) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    s_mono = compute_beta_schedule(x, x.copy(), SR, BCFG)
    assert s_mono[-1] > BCFG.beta_max - 0.1, s_mono[-1]


def test_beta_schedule_is_smooth():
    # Exponential smoothing: no block-to-block jumps beyond the step bound.
    rng = np.random.default_rng(7)
    l = (rng.standard_normal(N) * 0.2).astype(np.float32)
    r = l.copy()
    r[: N // 2] = (rng.standard_normal(N // 2) * 0.2).astype(np.float32)
    s = compute_beta_schedule(l, r, SR, BCFG)
    dt = 1.0 / SR
    chunk_s = len(l) // len(s) / SR if len(s) else 1
    alpha = 1.0 - np.exp(-(len(l) / len(s)) / SR / BCFG.tau_s)
    max_step = (BCFG.beta_max - BCFG.beta_min) * alpha * 1.5 + 1e-6
    steps = np.abs(np.diff(s))
    assert np.all(steps <= max_step), f"largest beta step {steps.max():.4f}"


def test_adaptive_render_keeps_directionality():
    rng = np.random.default_rng(9)
    noise = (0.3 * rng.standard_normal(N)).astype(np.float32)
    src = (noise, np.zeros(N, dtype=np.float32))
    schedule = compute_beta_schedule(*src, SR, BCFG)
    out = _render_with_schedule(src, schedule)
    sl, sr = _surround(out)
    sl_rms, sr_rms = measure(sl)[1], measure(sr)[1]
    assert sl_rms > 3 * sr_rms, f"SL={sl_rms:.4f} SR={sr_rms:.4f}"


def test_adaptive_render_dual_mono_quiet_surrounds():
    t = np.arange(N) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    schedule = compute_beta_schedule(x, x.copy(), SR, BCFG)
    out = _render_with_schedule((x, x.copy()), schedule)
    sl, sr = _surround(out)
    front_rms = measure(out.channels[0])[1]
    assert measure(sl)[1] < 0.2 * front_rms


@pytest.mark.skipif(not library_available(), reason="native core not built")
def test_native_adaptive_equivalence():
    rng = np.random.default_rng(11)
    l = (0.3 * rng.standard_normal(N)).astype(np.float32)
    r = l.copy()
    r[N // 2:] = (0.3 * rng.standard_normal(N // 2)).astype(np.float32)
    schedule = compute_beta_schedule(l, r, SR, BCFG)
    a = _render_with_schedule((l, r), schedule, native=False)
    b = _render_with_schedule((l, r), schedule, native=True)
    worst = max(
        float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        for x, y in zip(a.channels, b.channels)
    )
    assert worst < 1e-4, f"adaptive beta render deviation {worst:.3e}"


@pytest.mark.skipif(not library_available(), reason="native core not built")
def test_native_adaptive_differs_from_static():
    rng = np.random.default_rng(13)
    l = (0.3 * rng.standard_normal(N)).astype(np.float32)
    r = (0.3 * rng.standard_normal(N)).astype(np.float32)
    schedule = compute_beta_schedule(l, r, SR, BCFG)
    cfg = load_config("configs/default.toml")
    static = NativeUpmixRenderer(SR, cfg).render(l, r)
    adapt = NativeUpmixRenderer(SR, cfg).render(l, r, beta_schedule=schedule)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    assert not np.allclose(static.channels[idx["SL"]][:SR],
                           adapt.channels[idx["SL"]][:SR], atol=1e-4)
