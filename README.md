# Concerto — Stereo-to-Surround Upmixing Engine

Deterministic, measurable stereo → 5.1/7.1 upmixer. DSP-first: a reproducible
matrix baseline and analysis engine, with AI introduced only where it
demonstrably improves results. See `SPEC.md` for the frozen engineering
specification.

## Status — Milestone 2 (native C++20 DSP core) complete

Milestone 1 delivered the validated Python DSP prototype. Milestone 2 ports the
frozen kernels to a C++20 shared library (`native/`, CMake + MinGW-w64/MSVC),
bridged via ctypes with Python reference fallback:

- [x] Native C++20 core: radix-2 FFT (float64), all-pass banks, SOS band-limit
      (scipy layout, a0-normalised), slow RMS gain controller, chunk renderer
- [x] ctypes bridge (`concerto/native.py`), auto-used by `render`/`benchmark`
      when the core is built; graceful Python fallback otherwise
- [x] Native↔Python equivalence tests (render < 1e-4, analysis < 1e-6 rel.)
- [x] Covariance-based directness d(f) added to analysis (Python + native)
      and to CSV reports — becomes the W/D features in Milestone 3
- [x] Downmix-consistency acceptance test (MPEG-style fold-down SDR > 10 dB;
      measured ≈ 28 dB on the demo source)
- [x] White-noise stability stress test
- [x] **Milestone 4 (stage a): AI evaluation harness + vocal centre.**
  Strictly flag-gated (`--ai on`, `ai-eval` command); torch/demucs are
  optional dependencies, lazily imported — the DSP pipeline is untouched
  without them.
  - Demucs (htdemucs) stem separation, resampling-safe, timeline-preserving.
  - Acceptance harness (spec section 7): timeline alignment, recombination
    residual, segmentation-artefact envelope check, per-stem stereo sanity.
    Failed criteria **reject the stems** (render aborts) — separation success
    never assumed.
  - V feature: per-block vocal-presence confidence from the vocal stem,
    smoothed; policy-gated vocal centre (`centre.mode = "vocal_mask"`)
    applied as a post-pass — FL/FR preservation and the native fast path
    untouched, gate closed when V is low.
  - Measured on the reference track: separation 60 s CPU (2.5x rt),
    acceptance PASS (lag 0, 30.8 dB recombination), vocal gate mean 0.85,
    render validates clean.
  - Remaining M4: ONNX Runtime deployment / Arc execution provider, BS-RoFormer
    comparison, loudness-matched listening evaluation of vocal centre vs
    phantom centre.
- [x] **Milestone 3 (complete): deterministic adaptive controller.**
  - Signal-adaptive stereo_beta: side-dominance-weighted directness steers
    beta in [0.15, 0.90], tau = 1.5 s.
  - Full feature vector per block: rho, gamma2, W, D, T (envelope-peak
    transient density), A (low-coherence energy fraction). V (vocal
    confidence) is AI-gated and intentionally unavailable -> centre
    contribution stays policy-disabled.
  - Rule policy (spec section 6 table): transients -> reduced decorrelation
    + slight gain cut; existing ambience -> no added decorrelated energy;
    dry content -> envelopment lift; hard-panned/direct -> front protection.
    Multipliers clamped to [0.6, 1.4] x base gain, decorr intensity in
    [0.25, 1.0].
  - Dual-timescale smoothing: slow tau = 1.5 s for broad parameters, fast
    tau = 50 ms transient-protection envelope (dives decorr to floor at
    attacks, recovers quickly). Continuous per-block evaluation.
  - Native core: live-parameter C API (beta, gain multiplier, decorrelation
    intensity); Python/native equivalence verified with full schedules.
  - Measured on the reference track: gain x0.96-1.14, decorr dips to 0.25 at
    transients (mean 0.93), direction tracking preserved, downmix SDR
    improved 10.9 -> 13.9 dB (transient sections use the phase-coherent raw
    feed).

Note: measured on real music (left-dominant side content), the static
mono-difference feed produced a left-shifted rear image from anti-phase
cancellation. Stereo-difference extraction with adaptive beta fixes this:
surround direction-tracking correlation with the front balance rose from
~0.0 (v1) to +0.72 (v2) on the reference track.

Performance: native core renders ~40x real-time offline (0.002 s per second of
audio on this machine), ~36x faster than the numpy reference.

Build the native core:

```
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
```

### Milestone 1 checklist (all complete)
- [x] Decode stereo FLAC/WAV to float32 PCM, preserve metadata
- [x] Mid-Side decomposition
- [x] STFT analysis (4096/1024, Hann, 75% overlap)
- [x] Frequency-dependent correlation & magnitude-squared coherence
- [x] JSON + CSV analysis reports
- [x] Conservative matrix surround extraction (band-limited S, g_s=0.25)
- [x] Independent all-pass decorrelation banks (3 cascaded sections/path)
- [x] RMS-based surround energy control (slow envelope, no pumping)
- [x] 5.1 and 7.1 FLAC output with explicit channel-layout mapping
- [x] Channel-order validation (8 distinguishable routing signals)
- [x] Front-channel preservation tests (E_L = E_R = 0 in preservation mode)
- [x] Stereo fold-down validation

## Install

```
pip install numpy scipy soundfile
```

## Supported formats

- **Input:** FLAC, WAV, AIFF via libsndfile; Opus, MP3, and anything else
  FFmpeg can decode via automatic fallback (`concerto/audio/ffmpeg_io.py`).
  Lossy sources decode at their native decoder rate (e.g. Opus = 48 kHz),
  which is preserved like any other rate.
- **Output:** FLAC (default) or WAV, chosen by file extension. Channel-order
  mapping per container is handled by the layout table (WAVE 8ch orders the
  back pair before the side pair — the opposite of FLAC).

## Usage

```
# Analyse a stereo recording → JSON report + per-bin CSV (incl. directness)
python -m concerto analyse input.flac --out reports/

# Conservative DSP-only 5.1 render (centre disabled, preservation mode)
python -m concerto render input.flac --layout 5.1 --mode baseline --out output.flac

# Adaptive 7.1 render
python -m concerto render input.flac --layout 7.1 --mode adaptive --out output.flac

# Custom configuration
python -m concerto render input.flac --config configs/adaptive.toml --out output.flac

# Validate an output file
python -m concerto validate output.flac --reference input.flac

# Compare profiles with objective measurements
python -m concerto benchmark input.flac --profiles baseline,adaptive --out reports/
```

## Tests

```
python -m pytest tests/ -q
```

## Channel order (application-internal, enforced everywhere)

```
5.1:  FL FR FC LFE SL SR
7.1:  FL FR FC LFE SL SR BL BR
```

File-layout mappings are defined once in `concerto/audio/layouts.py`.
`FLAC 8ch` natively orders `FL FR FC LFE SL SR BL BR` (matches internal);
`FLAC 6ch` natively orders `FL FR FC LFE BL BR`; FFmpeg `7.1` orders
`FL FR FC LFE BL BR SL SR`. All conversions go through the mapping table —
never raw channel copy.

## Source preservation

The original file is never modified. Renders always keep input sample rate
(no silent resampling) and original `FL/FR` bit-identically in preservation
mode, before final quantization.
