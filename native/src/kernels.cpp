#include "concerto/kernels.hpp"

#include "concerto/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace concerto {

// ---------------------------------------------------------------------------
// AllPassBank

AllPassBank::AllPassBank(const std::vector<double>& coefficients) {
    if (coefficients.empty())
        throw std::invalid_argument("at least one all-pass section is required");
    for (double a : coefficients)
        if (!(a > -1.0 && a < 1.0))
            throw std::invalid_argument("unstable all-pass coefficient");
    a_ = coefficients;
    state_.assign(a_.size(), {0.0, 0.0});
}

void AllPassBank::reset() { state_.assign(a_.size(), {0.0, 0.0}); }

void AllPassBank::process(const float* in, float* out, std::size_t n) {
    // Work in double throughout the cascade; cast once at the end.
    std::vector<double>& buf = work_double_;
    buf.assign(in, in + n);
    for (std::size_t s = 0; s < a_.size(); ++s) {
        const double a = a_[s];
        double x_prev = state_[s][0], y_prev = state_[s][1];
        for (std::size_t i = 0; i < n; ++i) {
            const double xi = buf[i];
            const double yi = a * xi + x_prev - a * y_prev;
            buf[i] = yi;
            x_prev = xi;
            y_prev = yi;
        }
        state_[s] = {x_prev, y_prev};
    }
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(buf[i]);
}

// ---------------------------------------------------------------------------
// SosFilter (direct form II transposed, scipy-compatible)

SosFilter::SosFilter(const double* sos_sections, int n_sections) {
    for (int i = 0; i < n_sections; ++i) {
        const double* s = sos_sections + i * 6;
        const double a0 = s[3];
        if (a0 == 0.0) throw std::invalid_argument("SOS a0 must be non-zero");
        s_.push_back({s[0] / a0, s[1] / a0, s[2] / a0, s[4] / a0, s[5] / a0});
    }
    z_.assign(s_.size(), {0.0, 0.0});
}

void SosFilter::reset() { z_.assign(s_.size(), {0.0, 0.0}); }

void SosFilter::process(const float* in, float* out, std::size_t n) {
    work_double_.assign(in, in + n);
    if (tmp_double_.size() < n) tmp_double_.resize(n);
    for (std::size_t s = 0; s < s_.size(); ++s) {
        const auto& c = s_[s];          // b0, b1, b2, a1, a2
        double z1 = z_[s][0], z2 = z_[s][1];
        for (std::size_t i = 0; i < n; ++i) {
            const double x = work_double_[i];
            const double y = c[0] * x + z1;
            z1 = c[1] * x - c[3] * y + z2;
            z2 = c[2] * x - c[4] * y;
            tmp_double_[i] = y;
        }
        z_[s] = {z1, z2};
        work_double_.swap(tmp_double_);
    }
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<float>(work_double_[i]);
}

// ---------------------------------------------------------------------------
// GainController

GainController::GainController(const Params& p)
    : window_(std::max<std::size_t>(1, static_cast<std::size_t>(p.sample_rate * p.window_s))),
      alpha_(1.0 - std::exp(-static_cast<double>(window_) /
                            std::max(p.tau_s * p.sample_rate, 1e-9))),
      target_ratio_(p.target_ratio),
      gain_min_(p.gain_min),
      gain_max_(p.gain_max) {}

void GainController::process(const float* surround, const float* reference,
                             float* out, std::size_t n) {
    // Reference RMS envelope (one-pole smoothed, no pumping).
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) acc += static_cast<double>(reference[i]) * reference[i];
    const double ref_rms = std::sqrt(acc / static_cast<double>(n) + kEps);
    env_ = alpha_ * ref_rms + (1.0 - alpha_) * env_;

    double acc_s = 0.0;
    for (std::size_t i = 0; i < n; ++i) acc_s += static_cast<double>(surround[i]) * surround[i];
    const double sur_rms = std::sqrt(acc_s / static_cast<double>(n) + kEps);

    double target_gain = gain_;
    if (sur_rms > 0.0)
        target_gain = (target_ratio_ * env_) / (sur_rms + kEps);
    target_gain = std::clamp(target_gain, gain_min_, gain_max_);
    gain_ = alpha_ * target_gain + (1.0 - alpha_) * gain_;

    for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<float>(static_cast<double>(surround[i]) * gain_);
}

// ---------------------------------------------------------------------------
// UpmixRenderer

UpmixRenderer::UpmixRenderer(const Params& p)
    : p_(p), live_beta_(p.stereo_beta),
      live_gain_(p.surround_gain), live_decorr_(1.0),
      bandlimit_sl_(p.sos, p.n_sos_sections),
      bandlimit_sr_(p.sos, p.n_sos_sections),
      bandlimit_mono_(p.sos, p.n_sos_sections),
      gain_sl_([&] {
          GainController::Params gp;
          gp.sample_rate = p.sample_rate;
          gp.window_s = p.rms_window_s;
          gp.tau_s = p.gain_tau_s;
          gp.target_ratio = p.target_ratio;
          gp.gain_min = p.gain_min;
          gp.gain_max = p.gain_max;
          return gp;
      }()),
      gain_sr_(gain_sl_),
      gain_bl_(gain_sl_),
      gain_br_(gain_sl_) {
    if (p.decorrelate) {
        banks_.emplace_back(std::vector<double>(p.a_sl, p.a_sl + kAllPassSections));
        banks_.emplace_back(std::vector<double>(p.a_sr, p.a_sr + kAllPassSections));
        banks_.emplace_back(std::vector<double>(p.a_bl, p.a_bl + kAllPassSections));
        banks_.emplace_back(std::vector<double>(p.a_br, p.a_br + kAllPassSections));
    } else {
        // Undecorrelated reference mode: unity all-pass sections.
        banks_.emplace_back(std::vector<double>{0.0});
        banks_.emplace_back(std::vector<double>{0.0});
        banks_.emplace_back(std::vector<double>{0.0});
        banks_.emplace_back(std::vector<double>{0.0});
    }
}

void UpmixRenderer::process(const float* L, const float* R, std::size_t n,
                            float* fl, float* fr, float* fc, float* lfe,
                            float* sl, float* sr, float* bl, float* br) {
    // Front preservation: exact copies, no arithmetic.
    std::copy_n(L, n, fl);
    std::copy_n(R, n, fr);
    std::fill_n(fc, n, 0.0f);
    std::fill_n(lfe, n, 0.0f);

    // Surround feeds: stereo-difference keeps per-side directional feeds
    // (L-beta*R / R-beta*L); mono-difference shares one band-limited S.
    const double beta = p_.adaptive_beta ? live_beta_ : p_.stereo_beta;
    if (scratch_sl_bp_.size() < n) {
        scratch_sl_bp_.resize(n);
        scratch_sr_bp_.resize(n);
        scratch_sl_f_.resize(n);
        scratch_sr_f_.resize(n);
        scratch_tmp_.resize(n);
        scratch_mixed_.resize(n);
    }
    float* sl_bp = scratch_sl_bp_.data();
    float* sr_bp = scratch_sr_bp_.data();
    if (p_.extraction == 1) {
        float* sl_f = scratch_sl_f_.data();
        float* sr_f = scratch_sr_f_.data();
        for (std::size_t i = 0; i < n; ++i) {
            sl_f[i] = static_cast<float>(static_cast<double>(L[i]) - beta * R[i]);
            sr_f[i] = static_cast<float>(static_cast<double>(R[i]) - beta * L[i]);
        }
        bandlimit_sl_.process(sl_f, sl_bp, n);
        bandlimit_sr_.process(sr_f, sr_bp, n);
    } else {
        float* s_f = scratch_sl_f_.data();
        for (std::size_t i = 0; i < n; ++i)
            s_f[i] = static_cast<float>(0.5 * (static_cast<double>(L[i]) - R[i]));
        bandlimit_mono_.process(s_f, sl_bp, n);
        std::copy_n(sl_bp, n, sr_bp);
    }
    for (std::size_t i = 0; i < n; ++i) {
        sl_bp[i] = static_cast<float>(sl_bp[i] * live_gain_);
        sr_bp[i] = static_cast<float>(sr_bp[i] * live_gain_);
    }

    // Decorrelation intensity: (1-k)*feed + k*allpassed(feed).
    const double k = std::clamp(live_decorr_, 0.0, 1.0);
    float* tmp = scratch_tmp_.data();
    float* mixed = scratch_mixed_.data();
    auto mix = [&](const float* feed) -> const float* {
        if (k <= 0.001) return feed;  // intensity 0: raw band-limited feed
        for (std::size_t i = 0; i < n; ++i)
            mixed[i] = static_cast<float>((1.0 - k) * feed[i] + k * tmp[i]);
        return mixed;
    };

    banks_[0].process(sl_bp, tmp, n);
    gain_sl_.process(mix(sl_bp), L, sl, n);
    banks_[1].process(sr_bp, tmp, n);
    gain_sr_.process(mix(sr_bp), R, sr, n);
    banks_[2].process(sl_bp, tmp, n);
    gain_bl_.process(mix(sl_bp), L, bl, n);
    banks_[3].process(sr_bp, tmp, n);
    gain_br_.process(mix(sr_bp), R, br, n);

    if (p_.centre_mode == 1) {
        for (std::size_t i = 0; i < n; ++i)
            fc[i] = static_cast<float>(p_.mid_fill_gain * 0.5 * (static_cast<double>(L[i]) + R[i]));
    }
}

// ---------------------------------------------------------------------------
// Analysis

void analyze_stereo(const double* L, const double* R, std::size_t n,
                    const AnalysisSettings& st,
                    double* freqs, double* rho, double* coherence,
                    double* mid_energy, double* side_energy, double* directness) {
    const std::size_t N = static_cast<std::size_t>(st.fft_size);
    const std::size_t hop = static_cast<std::size_t>(st.hop_size);
    const auto w = hann(N);

    std::vector<double> l(L, L + n), r(R, R + n);
    if (l.size() < N) { l.resize(N, 0.0); r.resize(N, 0.0); }
    const std::size_t n_frames = 1 + (l.size() - N) / hop;
    const std::size_t bins = N / 2 + 1;

    // Per-frame POWER accumulation. Complex spectra must never be averaged
    // across frames: stationary tones with rational hop/frame ratios would
    // cancel phase and corrupt the estimates. Matches the Python reference:
    //   P_LR = mean_f(XL_f * conj(XR_f)),  P_LL = mean_f(|XL_f|^2), etc.
    std::vector<double> p_ll(bins, 0.0), p_rr(bins, 0.0);
    std::vector<double> p_mm(bins, 0.0), p_ss(bins, 0.0);
    std::vector<std::complex<double>> p_lr(bins, {0.0, 0.0});

    std::vector<std::complex<double>> xl(bins), xr(bins);
    std::vector<double> windowed(N);
    for (std::size_t f = 0; f < n_frames; ++f) {
        const std::size_t off = f * hop;

        for (std::size_t i = 0; i < N; ++i) windowed[i] = l[off + i] * w[i];
        rfft_exec(windowed.data(), xl.data(), N);

        for (std::size_t i = 0; i < N; ++i) windowed[i] = r[off + i] * w[i];
        rfft_exec(windowed.data(), xr.data(), N);

        for (std::size_t k = 0; k < bins; ++k) {
            p_ll[k] += std::norm(xl[k]);
            p_rr[k] += std::norm(xr[k]);
            p_lr[k] += xl[k] * std::conj(xr[k]);
        }

        for (std::size_t k = 0; k < bins; ++k) {
            const std::complex<double> xm = 0.5 * (xl[k] + xr[k]);
            const std::complex<double> xs = 0.5 * (xl[k] - xr[k]);
            p_mm[k] += std::norm(xm);
            p_ss[k] += std::norm(xs);
        }
    }

    const double nf = static_cast<double>(n_frames);
    for (std::size_t k = 0; k < bins; ++k) {
        const double pll = p_ll[k] / nf;
        const double prr = p_rr[k] / nf;
        const auto plr = p_lr[k] / nf;
        rho[k] = plr.real() / (std::sqrt(pll * prr) + kEps);
        coherence[k] = std::norm(plr) / (pll * prr + kEps);
        mid_energy[k] = p_mm[k] / nf;
        side_energy[k] = p_ss[k] / nf;
        // Covariance-based directness (frontier spec): normalized dominant
        // eigenvalue separation of the 2x2 stereo covariance matrix.
        directness[k] = std::sqrt((pll - prr) * (pll - prr) + 4.0 * std::norm(plr)) /
                        (pll + prr + kEps);
        if (freqs)
            freqs[k] = static_cast<double>(k) * st.sample_rate / st.fft_size;
    }
}

}  // namespace concerto
