"""Milestone-4 tests: acceptance harness, V schedule, vocal centre.

No torch/demucs required — all tests use synthetic stems.
"""

import numpy as np

from concerto.ai.centre import apply_vocal_centre
from concerto.ai.harness import (
    HarnessConfig, evaluate_stems, v_schedule_from_stems,
)
from concerto.ai.separation import Stems
from concerto.audio.buffer import MultichannelBuffer
from concerto.audio.layouts import INTERNAL_7_1

SR = 48000
N = SR * 3


def _mix_and_stems():
    """Synthetic mix = vocals + drums; stems returned are the exact sources."""
    rng = np.random.default_rng(0)
    t = np.arange(N) / SR
    # vocals: 300 Hz + 900 Hz, centred
    voc = 0.25 * (np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 900 * t))
    # drums: impulsive, hard-panned content on both sides
    drums_l = np.zeros(N)
    drums_r = np.zeros(N)
    for start in range(0, N, SR // 2):
        decay = np.exp(-0.002 * np.arange(3000))
        drums_l[start:start + 3000] += 0.4 * decay * rng.standard_normal(3000)
        drums_r[start:start + 3000] += 0.4 * decay * rng.standard_normal(3000)
    voc_l = voc.astype(np.float32)
    voc_r = voc.copy().astype(np.float32)
    mix_l = (voc_l + drums_l.astype(np.float32)).astype(np.float32)
    mix_r = (voc_r + drums_r.astype(np.float32)).astype(np.float32)
    stems = Stems(vocals=(voc_l, voc_r),
                  drums=(drums_l.astype(np.float32), drums_r.astype(np.float32)),
                  bass=(np.zeros(N, dtype=np.float32),) * 2,
                  other=(np.zeros(N, dtype=np.float32),) * 2)
    return mix_l, mix_r, stems


def test_harness_passes_perfect_stems():
    mix_l, mix_r, stems = _mix_and_stems()
    rep = evaluate_stems(mix_l, mix_r, stems, SR)
    assert rep.passed, [c for c in rep.checks if not c["passed"]]
    assert rep.metrics["recombination_db"] > 60  # exact recombination


def test_harness_rejects_misaligned_stems():
    mix_l, mix_r, stems = _mix_and_stems()
    # Shift every stem by 500 samples (way beyond tolerance).
    shifted = Stems(*(tuple(ch[500:] for ch in s) for s in stems))
    rep = evaluate_stems(mix_l, mix_r, shifted, SR)
    assert not rep.passed


def test_harness_rejects_poor_recombination():
    mix_l, mix_r, stems = _mix_and_stems()
    # Corrupt the vocal stem: recombination residual becomes large.
    bad_voc = (stems.vocals[0] * 0.5, stems.vocals[1] * 0.5)
    bad = Stems(bad_voc, stems.drums, stems.bass, stems.other)
    rep = evaluate_stems(mix_l, mix_r, bad, SR)
    assert not rep.passed


def test_harness_rejects_antiphase_stem():
    mix_l, mix_r, stems = _mix_and_stems()
    anti = Stems(stems.vocals, stems.drums,
                 (stems.bass[0], -stems.bass[1]),
                 stems.other)
    # bass stem has no energy in the synthetic mix, so add some
    rng = np.random.default_rng(2)
    bass_l = (0.2 * rng.standard_normal(N)).astype(np.float32)
    bass_r = bass_l.copy()
    mix_l2 = (mix_l + bass_l).astype(np.float32)
    mix_r2 = (mix_r + bass_r).astype(np.float32)
    anti = Stems(stems.vocals, stems.drums, (bass_l, -bass_r), stems.other)
    rep = evaluate_stems(mix_l2, mix_r2, anti, SR)
    assert not rep.passed  # anti-phase bass fails stereo check


def test_v_schedule_gates_on_vocal_energy():
    mix_l, mix_r, stems = _mix_and_stems()
    v = v_schedule_from_stems(stems.vocals[0], stems.vocals[1],
                              mix_l, mix_r, SR, chunk=16384)
    # Vocals are a large share here: gate should open substantially.
    assert v.max() > 0.7
    # With silent vocals, the gate must stay closed.
    silence = np.zeros(N, dtype=np.float32)
    v0 = v_schedule_from_stems(silence, silence.copy(), mix_l, mix_r, SR,
                               chunk=16384)
    assert v0.max() < 0.05


def test_vocal_centre_preserves_front_and_gates_fc():
    mix_l, mix_r, stems = _mix_and_stems()
    v = v_schedule_from_stems(stems.vocals[0], stems.vocals[1],
                              mix_l, mix_r, SR, chunk=16384)
    chans = [np.zeros(N, dtype=np.float32) for _ in range(8)]
    buf = MultichannelBuffer(chans, SR, INTERNAL_7_1.name)
    out = apply_vocal_centre(buf, stems.vocals[0], stems.vocals[1], v,
                             gain_db=-6.0)
    idx = {lbl: i for i, lbl in enumerate(INTERNAL_7_1.order)}
    # FC now carries the gated vocal mid.
    assert np.max(np.abs(out.channels[idx["FC"]])) > 0.01
    expected = 10 ** (-6 / 20) * stems.vocals[0]
    assert np.corrcoef(out.channels[idx["FC"]], expected)[0, 1] > 0.99
    # All other channels untouched.
    for lbl in ("FL", "FR", "LFE", "SL", "SR", "BL", "BR"):
        assert np.all(out.channels[idx[lbl]] == 0.0)
    # Silent vocals -> FC stays silent.
    out2 = apply_vocal_centre(buf, np.zeros(N, np.float32),
                              np.zeros(N, np.float32),
                              np.zeros(len(v), np.float64))
    assert np.all(out2.channels[idx["FC"]] == 0.0)


def test_stem_caching_roundtrip(tmp_path):
    from concerto.ai.separation import _compute_audio_hash, _save_cached_stems, _load_cached_stems
    mix_l, mix_r, stems = _mix_and_stems()
    h = _compute_audio_hash(mix_l, mix_r, SR, "test_model")
    assert len(h) == 24
    cache_file = tmp_path / f"{h}.npz"
    _save_cached_stems(cache_file, stems)
    assert cache_file.exists()
    loaded = _load_cached_stems(cache_file)
    assert loaded is not None
    assert np.array_equal(loaded.vocals[0], stems.vocals[0])
    assert np.array_equal(loaded.drums[1], stems.drums[1])


def test_onnx_execution_provider_discovery():
    from concerto.ai.separation import get_available_execution_providers
    providers = get_available_execution_providers()
    assert isinstance(providers, list)
    assert len(providers) >= 1
    # CPU is guaranteed universal fallback
    assert "CPUExecutionProvider" in [p[0] if isinstance(p, tuple) else p for p in providers]
