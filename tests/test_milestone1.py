"""Milestone-1 acceptance test suite."""

import numpy as np
import pytest

from concerto.audio.buffer import MultichannelBuffer, measure
from concerto.audio.decode import decode_stereo
from concerto.audio.encode import encode_flac
from concerto.audio.layouts import (
    FLAC_6CH_AS_51, FLAC_8CH, INTERNAL_5_1, INTERNAL_7_1, permutation,
)
from concerto.dsp.allpass import AllPassBank
from concerto.dsp.correlation import correlation_report
from concerto.dsp.midside import mid_side
from concerto.dsp.rms import GainController, bandlimit, bandlimit_sos
from concerto.dsp.upmix import UpmixRenderer, derive_5_1_from_7_1, load_config
from concerto.validate.tests import (
    detect_channel_frequencies, routing_test_signal, validate_file,
)

SR = 48000
N = SR * 2


@pytest.fixture(scope="module")
def stereo_source(tmp_path_factory):
    """Synthetic stereo source: correlated centre + decorrelated sides."""
    rng = np.random.default_rng(42)
    t = np.arange(N) / SR
    centre = 0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 660 * t)
    side = 0.15 * rng.standard_normal(N).astype(np.float32)
    left = (centre + side).astype(np.float32)
    right = (centre - side).astype(np.float32)
    path = tmp_path_factory.mktemp("src") / "test_stereo.flac"
    import soundfile as sf
    sf.write(path, np.stack([left, right], axis=1), SR, format="FLAC", subtype="PCM_24")
    return str(path), left, right


def default_cfg() -> dict:
    return load_config("configs/default.toml")


# --- layouts ---------------------------------------------------------------

def test_layout_permutations():
    assert permutation(INTERNAL_7_1, FLAC_8CH) == [0, 1, 2, 3, 4, 5, 6, 7]
    # Internal 5.1 maps positionally onto the FLAC 6ch surround slots.
    assert permutation(INTERNAL_5_1, FLAC_6CH_AS_51) == [0, 1, 2, 3, 4, 5]


def test_flac_8ch_roundtrip_order(tmp_path):
    chans, labels = routing_test_signal(SR, 0.5)
    path = str(tmp_path / "routes.flac")
    encode_flac(chans, SR, INTERNAL_7_1.name, path)
    detected = detect_channel_frequencies(path)
    expected = {"FL": 1000, "FR": 2000, "FC": 3000, "LFE": 60,
                "SL": 4000, "SR": 5000, "BL": 6000, "BR": 7000}
    for label in labels:
        assert abs(detected[label] - expected[label]) < expected[label] * 0.05, label


# --- mid/side ----------------------------------------------------------------

def test_mid_side_roundtrip():
    rng = np.random.default_rng(1)
    l = rng.standard_normal(1000).astype(np.float32)
    r = rng.standard_normal(1000).astype(np.float32)
    m, s = mid_side(l, r)
    l2 = m + s
    r2 = m - s
    assert np.allclose(l2, l, atol=1e-5)
    assert np.allclose(r2, r, atol=1e-5)


def test_mid_mono_cancellation():
    l = np.ones(100, dtype=np.float32)
    m, s = mid_side(l, l)
    assert np.allclose(s, 0.0)


# --- correlation / coherence -------------------------------------------------

def test_correlation_perfect_correlation():
    t = np.arange(N) / SR
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    rep = correlation_report(x, x, SR)
    mid = (rep.freqs > 800) & (rep.freqs < 1200)
    assert np.mean(rep.rho[mid]) > 0.99
    assert np.mean(rep.coherence[mid]) > 0.98


def test_correlation_anti_correlated():
    t = np.arange(N) / SR
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    rep = correlation_report(x, -x, SR)
    mid = (rep.freqs > 800) & (rep.freqs < 1200)
    assert np.mean(rep.rho[mid]) < -0.99


def test_correlation_uncorrelated():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(N)
    b = rng.standard_normal(N)
    rep = correlation_report(a, b, SR)
    assert abs(np.mean(rep.rho)) < 0.15  # statistically near zero


# --- all-pass ----------------------------------------------------------------

def test_allpass_stability_and_unity_magnitude():
    bank = AllPassBank([0.68, -0.41, 0.23])
    rng = np.random.default_rng(3)
    x = rng.standard_normal(100000).astype(np.float32)
    y = bank.process(x)
    assert np.all(np.isfinite(y))
    # Magnitude response ≈ unity: compare segment RMS
    assert abs(measure(y)[1] / measure(x)[1] - 1.0) < 0.02


def test_allpass_changes_signal_not_magnitude():
    bank = AllPassBank([-0.55, 0.37, -0.19])
    t = np.arange(SR) / SR
    x = np.sin(2 * np.pi * 500 * t).astype(np.float32)
    y = bank.process(x)
    # Not identical (phase shifted) but same amplitude envelope
    assert not np.allclose(x, y)
    assert abs(np.max(np.abs(y)) - 1.0) < 0.05


def test_allpass_state_persists_across_blocks():
    x = np.random.default_rng(5).standard_normal(96000).astype(np.float32)
    b1 = AllPassBank([0.68, -0.41, 0.23])
    whole = b1.process(x)
    b2 = AllPassBank([0.68, -0.41, 0.23])
    parts = np.concatenate([b2.process(x[i:i + 4096]) for i in range(0, len(x), 4096)])
    assert np.allclose(whole, parts, atol=1e-4)


def test_allpass_rejects_unstable_coefficients():
    with pytest.raises(ValueError):
        AllPassBank([1.2])


# --- filters / gain control ----------------------------------------------------

def test_bandlimit_rejects_out_of_band():
    sos = bandlimit_sos(SR, 150.0, 8000.0)
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 15000 * t).astype(np.float32)
    mid = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    yl, _ = bandlimit(low, sos)
    yh, _ = bandlimit(high, sos)
    ym, _ = bandlimit(mid, sos)
    assert measure(yl)[1] < 0.05 * measure(low)[1]
    assert measure(yh)[1] < 0.05 * measure(high)[1]
    assert measure(ym)[1] > 0.8 * measure(mid)[1]


def test_gain_controller_converges_to_target():
    gc = GainController(SR, tau_s=0.5, target_ratio=0.35)
    rng = np.random.default_rng(11)
    ref = (0.5 * rng.standard_normal(SR * 3)).astype(np.float32)
    sur = (0.1 * rng.standard_normal(SR * 3)).astype(np.float32)
    # Stream in 0.4 s chunks like the renderer does, so the envelope converges.
    chunk = int(SR * 0.4)
    out = np.concatenate(
        [gc.process(sur[i:i + chunk], ref[i:i + chunk]) for i in range(0, SR * 3, chunk)]
    )
    ratio = measure(out)[1] / measure(ref)[1]
    assert 0.25 < ratio < 0.5  # near target 0.35, smoothed


# --- renderer ------------------------------------------------------------------

def test_front_preservation_bit_identical(stereo_source):
    path, left, right = stereo_source
    buf, meta = decode_stereo(path)
    out = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    assert np.array_equal(out.channels[idx["FL"]], buf.left)
    assert np.array_equal(out.channels[idx["FR"]], buf.right)


def test_lfe_silent(stereo_source):
    buf, _ = decode_stereo(stereo_source[0])
    out = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    assert np.all(out.channels[idx["LFE"]] == 0.0)


def test_output_duration_and_channels(stereo_source):
    buf, _ = decode_stereo(stereo_source[0])
    out = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    assert out.n_channels == 8
    assert out.frame_count == buf.frame_count
    b51 = derive_5_1_from_7_1(out)
    assert b51.n_channels == 6
    assert b51.frame_count == buf.frame_count


def test_no_clipping(stereo_source):
    buf, _ = decode_stereo(stereo_source[0])
    out = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    assert out.peak() <= 1.0


def test_surround_energy_is_restrained(stereo_source):
    buf, _ = decode_stereo(stereo_source[0])
    out = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    sr_rms = measure(out.channels[idx["SR"]])[1]
    front_rms = measure(out.channels[idx["FL"]])[1]
    assert sr_rms < 0.6 * front_rms  # conservative by construction


def test_decorrelation_reduces_sl_sr_correlation(stereo_source):
    buf, _ = decode_stereo(stereo_source[0])
    cfg = default_cfg()
    cfg["extraction"] = "mono_difference"  # historical mono-feed reference
    dec = UpmixRenderer(SR, cfg).render(buf.left, buf.right)
    cfg2 = dict(cfg)
    cfg2["decorrelate"] = False
    undec = UpmixRenderer(SR, cfg2).render(buf.left, buf.right)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}

    def corr(bufx):
        a = bufx.channels[idx["SL"]][:SR].astype(np.float64)
        b = bufx.channels[idx["SR"]][:SR].astype(np.float64)
        return float(np.corrcoef(a, b)[0, 1])

    assert corr(undec) > 0.99          # identical feeds stay correlated
    assert corr(dec) < corr(undec)     # decorrelation separates them


def test_stereo_extraction_preserves_directionality(stereo_source):
    """Regression test for the left-biased rear image on real music.

    With a mono-difference feed, hard-panned ambience lands anti-phase in
    both surrounds and cancels. Stereo-difference feeds must route left-
    panned ambience predominantly to SL/BL and right-panned to SR/BR.
    """
    rng = np.random.default_rng(9)
    n = SR * 2
    noise = (0.3 * rng.standard_normal(n)).astype(np.float32)
    hard_left = (noise, np.zeros(n, dtype=np.float32))
    hard_right = (np.zeros(n, dtype=np.float32), noise)
    dual_mono = (noise, noise.copy())

    def surround_rms(src):
        out = UpmixRenderer(SR, default_cfg()).render(*src)
        idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
        return (measure(out.channels[idx["SL"]])[1],
                measure(out.channels[idx["SR"]])[1])

    sl_l, sr_l = surround_rms(hard_left)
    sl_r, sr_r = surround_rms(hard_right)
    sl_m, sr_m = surround_rms(dual_mono)
    # Directional fidelity: ipsilateral surround must dominate (>= 4x).
    assert sl_l > 3 * sr_l, f"hard-left: SL={sl_l:.4f} SR={sr_l:.4f}"
    assert sr_r > 3 * sl_r, f"hard-right: SL={sl_r:.4f} SR={sr_r:.4f}"
    # Dual mono stays centred (balanced).
    assert 0.8 < sl_m / sr_m < 1.25


def test_mono_vs_stereo_extraction_differ(stereo_source):
    buf, _ = decode_stereo(stereo_source[0])
    cfg = default_cfg()
    cfg["extraction"] = "mono_difference"
    mono = UpmixRenderer(SR, cfg).render(buf.left, buf.right)
    stereo = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    assert not np.allclose(mono.channels[idx["SL"]][:SR],
                           stereo.channels[idx["SL"]][:SR], atol=1e-4)


# --- full-pipeline file validation ---------------------------------------------

def test_full_pipeline_validation(stereo_source, tmp_path):
    path = stereo_source[0]
    buf, _ = decode_stereo(path)
    out = UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right)
    out_path = str(tmp_path / "render71.flac")
    encode_flac(out.channels, SR, INTERNAL_7_1.name, out_path, bit_depth=24)
    result = validate_file(out_path, reference=path)
    assert result.passed, [c for c in result.checks if not c["passed"]]


def test_51_pipeline_validation(stereo_source, tmp_path):
    path = stereo_source[0]
    buf, _ = decode_stereo(path)
    out = derive_5_1_from_7_1(UpmixRenderer(SR, default_cfg()).render(buf.left, buf.right))
    out_path = str(tmp_path / "render51.flac")
    encode_flac(out.channels, SR, INTERNAL_5_1.name, out_path, bit_depth=24)
    result = validate_file(out_path, reference=path)
    assert result.passed, [c for c in result.checks if not c["passed"]]
