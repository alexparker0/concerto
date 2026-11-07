# Concerto

Deterministic, measurable stereo-to-surround upmixing engine. 

Concerto converts 2-channel stereo audio into 5.1 or 7.1 surround mixes. It is built DSP-first: a mathematically reproducible matrix core with directional steering, cascaded all-pass decorrelation banks, and dual-timescale RMS gain tracking, with neural stem separation (`htdemucs` via PyTorch) introduced only where it demonstrably improves center-channel vocal stability without degrading front image fidelity.

Complete specifications, mathematical derivations, and QA criteria are documented in [`SPEC.md`](SPEC.md).

---

## Current Status

- [x] **Milestone 1 — Validated Python DSP Pipeline**: Mid-side decomposition, STFT correlation/coherence analysis, SOS band-limiting (150 Hz–8 kHz), 3-stage cascaded first-order all-pass decorrelators, slow RMS envelope tracking, multichannel FLAC/WAV mapping, and deterministic QA validation table.
- [x] **Milestone 2 — Native C++20 Core**: High-throughput C++20 shared library (`libconcerto_core.dll` / `.so`) with radix-2 float64 FFT, vectorized chunk renderer, and ctypes bridge. Renders offline at **40×–50× realtime** (approx. 0.002 s per second of audio), with identical float32 output to the Python reference within $< 10^{-4}$ deviation.
- [x] **Milestone 3 — Adaptive Controller**: Signal-adaptive cross-feed parameter $\beta(t) \in [0.15, 0.90]$ steered by side-dominance directness ($\tau = 1.5\text{ s}$). Dual-timescale transient envelope protection ($\tau = 50\text{ ms}$) dips decorrelation intensity at attacks to eliminate transient smearing, boosting fold-down SDR from $10.9\text{ dB}$ to $13.9\text{ dB}$.
- [x] **Milestone 4 (Stage a) — Neural Vocal Centre & QA Harness**: Flag-gated (`--ai on`) vocal separation using `htdemucs` (PyTorch). Includes a strict 4-point acceptance harness (timeline alignment $\le 4$ samples, recombination residual $\ge 20\text{ dB}$, transient envelope jump $\le 3.0$, stereo correlation $\ge -0.95$). Stems meeting criteria yield a smoothed vocal presence gate ($V$) for center-channel placement (`centre_mode = "vocal_mask"`) while preserving $FL/FR$ bit-identically.
- [x] **Desktop GUI Workstation**: Full-featured Qt6 (`PySide6`) desktop application with 8 dedicated workspaces, hardware-accelerated `pyqtgraph` 2D spectral plots, Lissajous stereo vectorscope, interactive 5.1/7.1 speaker layout diagram, and low-latency audio auditioning with real-time MPEG fold-down headphone downmixing.
- [x] **Standalone Windows Distribution**: Packaged via PyInstaller into `dist/concerto-gui/concerto-gui.exe`, bundling the native C++20 DSP core, preset configurations, and GUI runtime.

---

## System Requirements

- **Operating System**: Windows 10/11 (x64), Linux (x86_64), or macOS (Apple Silicon / Intel).
- **Python**: Python 3.10+ (tested through 3.15).
- **Core DSP Dependencies**: `numpy>=1.24.0`, `scipy>=1.10.0`, `soundfile>=0.12.0`.
- **Desktop GUI Dependencies**: `PySide6>=6.6.0`, `pyqtgraph>=0.13.0`, `sounddevice>=0.4.6`.
- **Optional AI Dependencies**: `torch>=2.0.0`, `demucs>=4.0.0`.
- **C++ Compiler** (optional, for rebuilding native core): CMake 3.20+ with MinGW-w64, MSVC, GCC 11+, or Clang 13+.

---

## Installation

### 1. Python Environment Setup
```bash
git clone https://github.com/alexparker0/concerto.git
cd concerto

# Install core audio DSP dependencies
pip install -r requirements.txt

# Install desktop GUI dependencies
pip install -r requirements-gui.txt

# (Optional) Install neural vocal center dependencies
pip install -r requirements-ai.txt
```

### 2. Native C++20 Core Compilation
A pre-compiled 64-bit Windows binary is included in `native/build/bin/libconcerto_core.dll`. To rebuild on any platform:
```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
```
The Python bridge (`concerto/native.py`) automatically discovers and loads the compiled library at runtime, falling back to the pure-Python reference implementation if unavailable.

---

## Usage

### Desktop GUI Workstation

Launch the interactive desktop interface:
```bash
# Direct GUI entrypoint
python -m concerto.gui

# Via CLI subcommand
python -m concerto gui
python -m concerto gui path/to/input.flac

# Using the standalone script
python run_gui.py
```

The GUI provides 8 integrated workspaces:
1. **Render Studio & Batch Queue**: Surround rendering (5.1/7.1, Baseline/Adaptive), neural vocal centre controls, ring-buffer streaming mode, TPDF dither, realtime speed factor ($X.Xx\text{ rt}$), output peak clipping monitor, and multi-file batch rendering.
2. **Stereo Analysis Lab**: Interactive spectral correlation plot ($\rho(f)$, $\gamma^2(f)$, $d(f)$ with log/linear frequency scale and crosshair hover), Lissajous vectorscope, polar phase meter, M/S energy balance, and CSV/JSON export.
3. **Acceptance & Validation Suite**: 10-point automated QA verification (channel count, clipping, FLAC integrity, LFE isolation, bitwise front preservation $E_L=E_R=0$, MPEG fold-down SDR $> 10\text{ dB}$, routing frequencies).
4. **Multi-Profile Benchmark**: Side-by-side profile comparisons (`baseline` vs `adaptive` vs `adaptive-ai`) measuring output peak, surround RMS, front RMS, surround-to-front ratio, and spatial surround coherence.
5. **AI Stem Lab**: Standalone stem separation via `htdemucs`, 4-point acceptance dashboard, stem energy meter, and solo stem auditioning (Vocals, Drums, Bass, Other, Recombination Residual).
6. **Speaker Routing**: 8-channel calibration tone generator, interactive 5.1/7.1 speaker layout diagram with active channel halos, channel-by-channel sweep, and verification table.
7. **Preset Studio**: Live visual editor for TOML profiles (surround extraction, filter orders, all-pass coefficients, adaptive policy time constants).
8. **Diagnostics**: C++20 core detection and throughput benchmark, host audio device channel capabilities.

### Command-Line Interface (CLI)

```bash
# 1. Analyse a stereo track → summary JSON + per-bin CSV
python -m concerto analyse track.flac --out reports/

# 2. Render conservative 5.1 mix (Baseline matrix DSP)
python -m concerto render track.flac --layout 5.1 --mode baseline --out mix_51.flac

# 3. Render 7.1 mix with adaptive steering beta and transient decorrelation dipping
python -m concerto render track.flac --layout 7.1 --mode adaptive --out mix_71.flac

# 4. Render with neural vocal centre extraction (requires torch/demucs)
python -m concerto render track.flac --layout 7.1 --mode adaptive --ai on --out mix_ai.flac

# 5. Stream processing via ring buffer (bounds memory to O(chunk), ideal for long files)
python -m concerto render track.flac --layout 7.1 --mode adaptive --stream --out mix_stream.flac

# 6. Validate rendered multichannel output against original stereo reference
python -m concerto validate mix_71.flac --reference track.flac

# 7. Benchmark multiple profiles with objective spatial metrics
python -m concerto benchmark track.flac --profiles baseline,adaptive --out reports/

# 8. Generate and verify 8-channel calibration test tone file
python -m concerto routing-test --out routing_test.flac --seconds 1.0
```

---

## Channel Layout & Container Mappings

Concerto enforces an internal layout order across all internal processing blocks:

```
5.1 Internal:  FL  FR  FC  LFE  SL  SR
7.1 Internal:  FL  FR  FC  LFE  SL  SR  BL  BR
```

When writing container files, channels are remapped according to container standards defined in `concerto/audio/layouts.py`:
- **FLAC 8ch**: `FL FR FC LFE SL SR BL BR` (matches internal order).
- **FLAC 6ch**: `FL FR FC LFE BL BR` (MPEG surround mapping).
- **WAV / WAVE 8ch**: `FL FR FC LFE BL BR SL SR` (SMPTE / Microsoft standard; rear pair precedes side pair).
- **WAV / WAVE 6ch**: `FL FR FC LFE BL BR`.

---

## Source Preservation Invariants

Concerto enforces three invariants across every DSP and render stage:
1. **Sampling Rate Preservation**: Input sample rate is preserved identically without silent resampling.
2. **Bit-Identical Front Preservation**: In default preservation mode, $FL$ and $FR$ are bit-identical to the source before final quantization ($E_L = E_R = 0$).
3. **MPEG Fold-Down Consistency**: Downmixing the surround output back to stereo ($L_{\text{down}} = FL + 0.707 \cdot FC + 0.707 \cdot SL + 0.5 \cdot BL$) yields a Signal-to-Distortion Ratio (SDR) $> 10\text{ dB}$ (typically $> 20\text{ dB}$ on real-world material).

---

## Testing & Verification

Run the full automated test suite:
```bash
python -m pytest tests/ -q
```
Coverage includes:
- Milestone 1 baseline DSP math and QA acceptance table (`test_milestone1.py`).
- C++20 native core vs. Python reference equivalence tests (`test_native.py`).
- Adaptive controller beta steering and transient rules (`test_adaptive.py`, `test_policy.py`).
- AI acceptance harness and vocal gate schedules (`test_ai.py`).
- Multichannel container formats, FLAC metadata, and 24-bit PCM quantization (`test_formats.py`).
- Ring-buffer streaming renderer consistency (`test_streaming.py`).

---

## Standalone Executable Build

To compile a standalone Windows `.exe` distribution with PyInstaller:
```bash
python scripts/build_gui.py
```
The self-contained application will be created in `dist/concerto-gui/concerto-gui.exe`.

---

## License

MIT License. See [`LICENSE`](LICENSE) for details.
