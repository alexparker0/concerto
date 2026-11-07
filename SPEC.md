# Concerto Engineering Specification (v1 — frozen for Milestone 1)

## 1. Technical baseline

| Property | Specification |
| --- | --- |
| Application | Offline CLI (consumer UI later) |
| Input | Stereo FLAC, WAV |
| Output | 5.1 and 7.1 FLAC |
| Internal processing | Float32 |
| Analysis precision | Float64 where beneficial |
| Native DSP (target) | C++20 (post-validation port; Milestone 1 = Python/numpy/scipy) |
| FFT | numpy/scipy now; PFFFT or FFTW in native port |
| AI inference (later) | ONNX Runtime; Demucs / BS-RoFormer candidates |
| Audio I/O | libsndfile (soundfile); FFmpeg available for layout remux |
| Processing | Offline, non-real-time |
| Primary mode | Adaptive |
| Original preservation | Mandatory — source never modified |
| LFE | Disabled by default |
| Sample rate | Preserve input rate; never silently resample |

## 2. Internal channel order (single source of truth)

```
5.1:  FL FR FC LFE SL SR
7.1:  FL FR FC LFE SL SR BL BR
```

All file-format mappings live in `concerto/audio/layouts.py`:

| Layout | File order |
| --- | --- |
| Internal 5.1 | FL FR FC LFE SL SR |
| Internal 7.1 | FL FR FC LFE SL SR BL BR |
| FLAC 6ch native | FL FR FC LFE BL BR |
| FLAC 8ch native | FL FR FC LFE SL SR BL BR |
| FFmpeg `7.1` | FL FR FC LFE BL BR SL SR |

## 3. Signal path (Milestone 1)

1. Decode stereo → `StereoBuffer{left, right: float32; sample_rate; frame_count}` + metadata
   (bit depth subtype, peak, loudness). Original file untouched.
2. Mid/Side: `M=(L+R)/2`, `S=(L−R)/2` — analysis features only in baseline mode.
3. STFT: N=4096, hop=1024, Hann, 75% overlap, channels L, R, M, S, float64 analysis.
   Per-bin energy-weighted statistics over time:
   - Correlation `ρ(f) = Re(P_LR) / sqrt(P_LL·P_RR + ε)` (signed similarity)
   - Coherence `γ²(f) = |P_LR|² / (P_LL·P_RR + ε)` (linear relationship)
   - Mid/Side bin energies
   Reports: `analyse` emits JSON (segment features, recommended parameters) + CSV
   (freq, rho, coherence, mid_energy, side_energy).
4. Surround feed: `SL = H_L(z)·S`, `SR = H_R(z)·S`, band-limited ≈150 Hz–8 kHz
   (smooth Butterworth SOS, not brick-wall), gain `g_s = 0.25` (−12 dB, configurable).
5. Decorrelation: cascaded first-order all-pass sections
   `y[n] = a·x[n] + x[n−1] − a·y[n−1]`, `|a|<1`, 3 sections/path, independent
   coefficient sets for SL / SR (7.1 adds BL / BR with a third set). State is
   maintained across the whole signal — never reset at window boundaries.
6. Energy control: windowed RMS of surround vs stereo, configurable target ratio,
   slow one-pole gain envelope (τ ≈ 2 s). Spatial balance, not loudness — no fast
   compression.
7. Centre: `disabled` (default) or `mid_fill` (−9 dB rel. to M, configurable).
   No AI vocal extraction in Milestone 1.
8. Front preservation: `FL = L`, `FR = R` bit-identical before final quantization.
9. 7.1 is the primary representation. 5.1 derive: `S_{5.1,L} = 0.707·(SL+BL)`,
   `S_{5.1,R} = 0.707·(SR+BR)`; FL/FR/FC/LFE unchanged; measure peaks post-sum.
10. Encode FLAC at input sample rate; verify by round-trip decode.

## 4. Adaptive mode (deterministic controller, no NN)

Per analysis segment, feature vector `x = [ρ, γ², W, D, T, A, V]` (normalized):
stereo correlation, mean coherence, width, diffuse-energy proportion, transient
density, ambience estimate, vocal-presence confidence. Rule-based policy sets
surround gain, decorrelation intensity, reverb amount, centre contribution.
Rules evaluated continuously; parameters smoothed with
`p_t = α·p_target + (1−α)·p_{t−1}`, `α = 1 − e^{−Δt/τ}`, τ ≈ 1–3 s for spatial
parameters; faster timescale for transient protection.

Milestone 1 implements the smoothing infrastructure + a conservative static
adaptive policy; full feature extraction (T, A, V) arrives with the analysis
upgrade pass.

## 5. AI integration (strictly post-Milestone-1)

Priority: Demucs → BS-RoFormer → ONNX Runtime deployment → Essentia features.
Acceptance criteria for any stem: timeline alignment, acceptable transients,
recombination close to original, interpretable stereo positioning, measurable
improvement over DSP-only. Separation success ≠ reconstruction success.

## 6. Acceptance tests (automated)

| Test | Criterion |
| --- | --- |
| Front preservation | `max|L_out−L_in| = max|R_out−R_in| = 0` before quantization |
| Channel count | Exactly 6 or 8 |
| Sample alignment | No inter-channel offsets |
| Clipping | No samples outside [−1, 1] |
| LFE | Silent when disabled |
| Filter stability | No divergence (finite output, bounded state) |
| Duration | Output frames == input frames |
| FLAC integrity | Round-trip decode succeeds, frame count matches |
| Fold-down | No severe cancellation on fold-down to stereo |
| Channel routing | 8 distinguishable test signals land on labelled channels |

## 7. Roadmap

- **M1 (done):** Python DSP prototype, analysis engine, tests.
- **M2 (done):** Native C++20 DSP core (CMake, radix-2 FFT float64, scipy-layout
  SOS, all-pass banks, gain controller) behind a C ABI; ctypes bridge with
  Python fallback; equivalence tests; directness d(f); fold-down SDR test.
  FFTW/PFFFT can replace the radix-2 FFT without touching callers.
- **M3 (complete):** Deterministic adaptive controller. Signal-adaptive
  stereo_beta; full per-block feature vector (rho, gamma2, W, D, T, A;
  V deferred to M4 with AI); rule policy for surround gain and decorrelation
  intensity; dual-timescale smoothing (slow 1.5 s / transient 50 ms);
  live-parameter native C API with equivalence tests.
- **M4 (stage a complete):** AI evaluation harness + Demucs separation +
  policy-gated vocal centre, all behind `--ai on`. Stage b: ONNX Runtime
  deployment / Arc execution provider, BS-RoFormer comparison, listening
  evaluation of vocal centre vs phantom centre.
- **M5:** Consumer UI on top of the frozen engine core.

Evaluation discipline: fixed reference excerpts, loudness-matched comparisons,
one DSP parameter changed at a time.
