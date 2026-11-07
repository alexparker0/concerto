"""Milestone-3 rule-policy and schedule tests (gain, decorrelation, transients)."""

import numpy as np
import pytest

from concerto.adaptive import (
    BlockFeatures, FeatureConfig, PolicyConfig, block_directness,
    compute_adaptive_schedules, extract_block_features, policy_targets,
)
from concerto.audio.buffer import measure
from concerto.audio.layouts import INTERNAL_7_1
from concerto.dsp.upmix import load_config
from concerto.native import NativeUpmixRenderer, library_available

SR = 48000
N = SR * 4


def _features(l, r):
    return extract_block_features(l, r, SR, FeatureConfig())


def test_features_steady_tone_low_transients():
    t = np.arange(N) / SR
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f = _features(x, x.copy())
    assert f.transients < 0.1
    assert f.rho > 0.95          # dual mono
    assert f.ambience < 0.2      # coherent content


def test_features_clicks_high_transients():
    rng = np.random.default_rng(1)
    x = np.zeros(N, dtype=np.float32)
    for start in range(0, N, SR // 4):          # 4 clicks per second
        decay = np.exp(-0.0015 * np.arange(2000))
        x[start:start + 2000] += 0.8 * decay * rng.standard_normal(2000) * 0.3
    f = _features(x, x.copy())
    assert f.transients > 0.5


def test_features_diffuse_noise_high_ambience():
    rng = np.random.default_rng(2)
    a = (rng.standard_normal(N) * 0.2).astype(np.float32)
    b = (rng.standard_normal(N) * 0.2).astype(np.float32)
    f = _features(a, b)
    assert f.ambience > 0.5
    assert f.diffuse > 0.5
    assert abs(f.rho) < 0.2


def test_features_hard_panned_low_ambience():
    rng = np.random.default_rng(3)
    x = (0.3 * rng.standard_normal(N)).astype(np.float32)
    f = _features(x, np.zeros(N, dtype=np.float32))
    assert f.ambience < 0.3      # hard-panned content is coherent in its channel


def test_policy_transient_reduces_decorr():
    transient = BlockFeatures(rho=0.9, gamma2=0.9, width=0.2, diffuse=0.2,
                              transients=0.8, ambience=0.2)
    steady = BlockFeatures(rho=0.9, gamma2=0.9, width=0.2, diffuse=0.2,
                           transients=0.1, ambience=0.2)
    p = PolicyConfig()
    g_t, d_t = policy_targets(transient, p)
    g_s, d_s = policy_targets(steady, p)
    assert d_t < d_s             # transient protection reduces decorrelation
    assert g_t < g_s             # and slightly reduces gain


def test_policy_ambience_rules():
    amb = BlockFeatures(rho=0.2, gamma2=0.2, width=0.6, diffuse=0.8,
                        transients=0.1, ambience=0.7)
    dry = BlockFeatures(rho=0.9, gamma2=0.9, width=0.1, diffuse=0.1,
                        transients=0.1, ambience=0.1)
    p = PolicyConfig()
    _, d_amb = policy_targets(amb, p)
    g_dry, _ = policy_targets(dry, p)
    assert d_amb < 1.0           # existing ambience -> less added decorrelation
    assert g_dry > 1.0           # dry content -> envelopment lift


def test_policy_multipliers_within_bounds():
    p = PolicyConfig()
    for _ in range(20):
        f = BlockFeatures(*(np.random.default_rng().random(6).tolist()))
        g, d = policy_targets(f, p)
        assert p.gain_min_mult <= g <= p.gain_max_mult
        assert p.decorr_min <= d <= p.decorr_max


def test_schedules_shapes_and_bounds():
    rng = np.random.default_rng(4)
    l = (0.3 * rng.standard_normal(N)).astype(np.float32)
    r = l.copy()
    r[: N // 2] = (0.3 * rng.standard_normal(N // 2)).astype(np.float32)
    s = compute_adaptive_schedules(l, r, SR,
                                   __import__("concerto.adaptive", fromlist=["AdaptiveBetaConfig"]).AdaptiveBetaConfig(),
                                   PolicyConfig())
    for k in ("beta", "gain", "decorr"):
        assert len(s[k]) == (N + 16383) // 16384
    assert np.all((s["beta"] >= 0.15) & (s["beta"] <= 0.90))
    assert np.all((s["gain"] >= 0.6) & (s["gain"] <= 1.4))
    assert np.all((s["decorr"] >= 0.0) & (s["decorr"] <= 1.0))


def test_transient_protection_dips_decorrelation():
    """Decorr schedule must dip at click bursts and recover quickly."""
    rng = np.random.default_rng(5)
    x = np.zeros(N, dtype=np.float32)
    for start in range(0, N, SR * 2):            # sparse clicks every 2 s
        x[start:start + 2000] += 0.8 * rng.standard_normal(2000) * 0.3
    s = compute_adaptive_schedules(x, x.copy(), SR,
                                   __import__("concerto.adaptive", fromlist=["AdaptiveBetaConfig"]).AdaptiveBetaConfig(),
                                   PolicyConfig())
    decorr = s["decorr"]
    # Most blocks are steady (high decorrelation); click blocks dive to the
    # protection floor via the fast envelope.
    assert np.median(decorr) > 0.7
    assert np.min(decorr) < 0.5 * np.median(decorr)


@pytest.mark.skipif(not library_available(), reason="native core not built")
def test_native_full_adaptive_equivalence():
    from concerto.adaptive import AdaptiveBetaConfig
    rng = np.random.default_rng(6)
    l = (0.3 * rng.standard_normal(N)).astype(np.float32)
    r = l.copy()
    r[N // 2:] = (0.3 * rng.standard_normal(N // 2)).astype(np.float32)
    s = compute_adaptive_schedules(l, r, SR, AdaptiveBetaConfig(), PolicyConfig())
    cfg = load_config("configs/default.toml")
    a = __import__("concerto.dsp.upmix", fromlist=["UpmixRenderer"]).UpmixRenderer(SR, dict(cfg)).render(
        l, r, beta_schedule=s["beta"], gain_schedule=s["gain"], decorr_schedule=s["decorr"])
    b = NativeUpmixRenderer(SR, cfg).render(
        l, r, beta_schedule=s["beta"], gain_schedule=s["gain"], decorr_schedule=s["decorr"])
    worst = max(float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                for x, y in zip(a.channels, b.channels))
    assert worst < 1e-4, f"full adaptive deviation {worst:.3e}"


@pytest.mark.skipif(not library_available(), reason="native core not built")
def test_native_gain_schedule_changes_output():
    from concerto.adaptive import AdaptiveBetaConfig
    rng = np.random.default_rng(7)
    l = (0.3 * rng.standard_normal(N)).astype(np.float32)
    r = (0.3 * rng.standard_normal(N)).astype(np.float32)
    s = compute_adaptive_schedules(l, r, SR, AdaptiveBetaConfig(), PolicyConfig())
    cfg = load_config("configs/default.toml")
    static = NativeUpmixRenderer(SR, cfg).render(l, r)
    full = NativeUpmixRenderer(SR, cfg).render(
        l, r, beta_schedule=s["beta"], gain_schedule=s["gain"],
        decorr_schedule=s["decorr"])
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    assert not np.allclose(static.channels[idx["SL"]][:SR],
                           full.channels[idx["SL"]][:SR], atol=1e-4)
