// Concerto native DSP kernels (Milestone 2 C++20 core).
//
// These kernels are behavioural ports of the validated Milestone-1 Python
// implementations (concerto/dsp/*). Operation order is preserved so the
// equivalence tests in tests/test_native.py can compare against the Python
// reference within float32 tolerance.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace concerto {

constexpr double kEps = 1e-12;

// ---------------------------------------------------------------------------
// Cascaded first-order all-pass sections.
// y[n] = a*x[n] + x[n-1] - a*y[n-1], |a| < 1. State persists across calls.
class AllPassBank {
public:
    explicit AllPassBank(const std::vector<double>& coefficients);
    // Processes one channel; accumulates in double, emits float32.
    void process(const float* in, float* out, std::size_t n);
    void reset();

private:
    std::vector<double> a_;
    std::vector<std::array<double, 2>> state_;  // (x_prev, y_prev) per section
    std::vector<double> work_double_;
};

// ---------------------------------------------------------------------------
// Cascade of biquad SOS sections (direct form II transposed, scipy layout:
// b0, b1, b2, a0, a1, a2 — a0 is normalised to 1 on load).
class SosFilter {
public:
    SosFilter(const double* sos_sections, int n_sections);
    void process(const float* in, float* out, std::size_t n);
    void reset();

private:
    std::vector<std::array<double, 5>> s_;  // b0,b1,b2,a1,a2 (a0 normalised out)
    std::vector<std::array<double, 2>> z_;  // transposed DF-II state per section
    std::vector<double> work_double_;
    std::vector<double> tmp_double_;
};

// ---------------------------------------------------------------------------
// Slow one-pole RMS gain controller (spatial balance, not loudness).
class GainController {
public:
    struct Params {
        int sample_rate = 48000;
        double window_s = 0.4;
        double tau_s = 2.0;
        double target_ratio = 0.35;
        double gain_min = 0.0;
        double gain_max = 2.0;
    };
    explicit GainController(const Params& p);
    void process(const float* surround, const float* reference, float* out,
                 std::size_t n);

private:
    std::size_t window_;
    double alpha_;
    double target_ratio_;
    double gain_min_, gain_max_;
    double gain_ = 1.0;
    double env_ = 0.0;
};

// ---------------------------------------------------------------------------
// Matrix upmix renderer (8-channel primary representation; 5.1 is derived in
// the orchestration layer). Front channels are exact copies (preservation).
class UpmixRenderer {
public:
    static constexpr int kMaxSections = 4;   // hp (2) + lp (2) at order 4
    static constexpr int kAllPassSections = 3;

    struct Params {
        int sample_rate = 48000;
        double surround_gain = 0.25;
        // Band-limit SOS: n_sos_sections sections, 6 doubles each.
        int n_sos_sections = 4;
        double sos[kMaxSections * 6] = {};
        // Decorrelation coefficients, one bank per surround path.
        double a_sl[kAllPassSections] = {};
        double a_sr[kAllPassSections] = {};
        double a_bl[kAllPassSections] = {};
        double a_br[kAllPassSections] = {};
        int decorrelate = 1;
        int extraction = 1;      // 0 = mono_difference, 1 = stereo_difference
        double stereo_beta = 0.25;
        double target_ratio = 0.35;
        double rms_window_s = 0.4;
        double gain_tau_s = 2.0;
        double gain_min = 0.0;
        double gain_max = 2.0;
        int centre_mode = 0;  // 0 = disabled, 1 = mid_fill
        double mid_fill_gain = 0.3548;
        int adaptive_beta = 0;  // 1 = live beta via set_stereo_beta()
    };

    explicit UpmixRenderer(const Params& p);
    // Live parameters for signal-adaptive operation. live_gain_ is a
    // multiplier on the configured base surround gain; live_decorr_ is the
    // decorrelation intensity (0 = raw feed, 1 = fully decorrelated).
    void set_stereo_beta(double beta) { live_beta_ = beta; }
    void set_dsp_params(double gain_mult, double decorr_mix) {
        live_gain_ = p_.surround_gain * gain_mult;
        live_decorr_ = decorr_mix;
    }
    // Renders one chunk; each output is a float32 buffer of n samples.
    // Pointers may alias the inputs only for fl/fr (they are copies).
    void process(const float* L, const float* R, std::size_t n,
                 float* fl, float* fr, float* fc, float* lfe,
                 float* sl, float* sr, float* bl, float* br);

private:
    Params p_;
    SosFilter bandlimit_sl_, bandlimit_sr_, bandlimit_mono_;
    double live_beta_ = 0.25;
    double live_gain_ = 0.25;      // absolute surround gain (base * multiplier)
    double live_decorr_ = 1.0;     // decorrelation intensity 0..1
    std::vector<AllPassBank> banks_;  // SL, SR, BL, BR
    GainController gain_sl_, gain_sr_, gain_bl_, gain_br_;
    std::vector<float> scratch_sl_bp_;
    std::vector<float> scratch_sr_bp_;
    std::vector<float> scratch_sl_f_;
    std::vector<float> scratch_sr_f_;
    std::vector<float> scratch_tmp_;
    std::vector<float> scratch_mixed_;
};

// ---------------------------------------------------------------------------
// STFT stereo analysis: per-bin correlation, coherence, mid/side energies and
// covariance-based directness. Out arrays must hold fft_size/2 + 1 doubles.
struct AnalysisSettings {
    int fft_size = 4096;
    int hop_size = 1024;
    int sample_rate = 48000;
};

void analyze_stereo(const double* L, const double* R, std::size_t n,
                    const AnalysisSettings& settings,
                    double* freqs, double* rho, double* coherence,
                    double* mid_energy, double* side_energy, double* directness);

}  // namespace concerto
