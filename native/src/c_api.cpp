// Minimal extern "C" surface for the Python ctypes bridge (concerto/native.py).
// Plain C ABI: no exceptions may cross this boundary.
#include "concerto/kernels.hpp"

#include <cstring>
#include <new>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
namespace {
struct ScopedFtzDaz {
    unsigned int old_csr;
    ScopedFtzDaz() : old_csr(_mm_getcsr()) {
        _mm_setcsr(old_csr | 0x8040);  // FTZ (bit 15) and DAZ (bit 6)
    }
    ~ScopedFtzDaz() {
        _mm_setcsr(old_csr);
    }
};
}
#else
namespace {
struct ScopedFtzDaz {};
}
#endif
using concerto::AllPassBank;
using concerto::AnalysisSettings;
using concerto::GainController;
using concerto::UpmixRenderer;

extern "C" {

// --- Renderer ---------------------------------------------------------------
// C-visible mirror of UpmixRenderer::Params (POD only).

struct CcRendererParams {
    int32_t sample_rate;
    double surround_gain;
    int32_t n_sos_sections;
    double sos[UpmixRenderer::kMaxSections * 6];
    double a_sl[UpmixRenderer::kAllPassSections];
    double a_sr[UpmixRenderer::kAllPassSections];
    double a_bl[UpmixRenderer::kAllPassSections];
    double a_br[UpmixRenderer::kAllPassSections];
    int32_t decorrelate;
    int32_t extraction;
    double stereo_beta;
    double target_ratio;
    double rms_window_s;
    double gain_tau_s;
    double gain_min;
    double gain_max;
    int32_t centre_mode;
    double mid_fill_gain;
    int32_t adaptive_beta;
};

static_assert(std::is_trivially_copyable_v<CcRendererParams>);

UpmixRenderer* cc_renderer_new(const CcRendererParams* p) {
    if (!p) return nullptr;
    try {
        UpmixRenderer::Params pp{};
        pp.sample_rate = p->sample_rate;
        pp.surround_gain = p->surround_gain;
        pp.n_sos_sections = p->n_sos_sections;
        std::memcpy(pp.sos, p->sos, sizeof(pp.sos));
        std::memcpy(pp.a_sl, p->a_sl, sizeof(pp.a_sl));
        std::memcpy(pp.a_sr, p->a_sr, sizeof(pp.a_sr));
        std::memcpy(pp.a_bl, p->a_bl, sizeof(pp.a_bl));
        std::memcpy(pp.a_br, p->a_br, sizeof(pp.a_br));
        pp.decorrelate = p->decorrelate;
        pp.extraction = p->extraction;
        pp.stereo_beta = p->stereo_beta;
        pp.target_ratio = p->target_ratio;
        pp.rms_window_s = p->rms_window_s;
        pp.gain_tau_s = p->gain_tau_s;
        pp.gain_min = p->gain_min;
        pp.gain_max = p->gain_max;
        pp.centre_mode = p->centre_mode;
        pp.mid_fill_gain = p->mid_fill_gain;
        pp.adaptive_beta = p->adaptive_beta;
        return new UpmixRenderer(pp);
    } catch (...) {
        return nullptr;
    }
}

void cc_renderer_free(UpmixRenderer* r) { delete r; }

void cc_renderer_set_stereo_beta(UpmixRenderer* r, double beta) {
    if (r) r->set_stereo_beta(beta);
}

void cc_renderer_set_dsp_params(UpmixRenderer* r, double gain_mult,
                                double decorr_mix) {
    if (r) r->set_dsp_params(gain_mult, decorr_mix);
}

int cc_renderer_process(UpmixRenderer* r, const float* L, const float* R,
                        float* fl, float* fr, float* fc, float* lfe,
                        float* sl, float* sr, float* bl, float* br, int32_t n) {
    if (!r || !L || !R || n < 0) return -1;
    ScopedFtzDaz ftz;
    try {
        r->process(L, R, static_cast<std::size_t>(n), fl, fr, fc, lfe, sl, sr, bl, br);
        return 0;
    } catch (...) {
        return -2;
    }
}

// --- Analysis ---------------------------------------------------------------

int cc_analyze_stereo(const double* L, const double* R, int64_t n,
                      int32_t fft_size, int32_t hop_size, int32_t sample_rate,
                      double* freqs, double* rho, double* coherence,
                      double* mid_energy, double* side_energy, double* directness) {
    if (!L || !R || n < 0) return -1;
    ScopedFtzDaz ftz;
    try {
        AnalysisSettings st{};
        st.fft_size = fft_size;
        st.hop_size = hop_size;
        st.sample_rate = sample_rate;
        concerto::analyze_stereo(L, R, static_cast<std::size_t>(n), st, freqs, rho,
                                 coherence, mid_energy, side_energy, directness);
        return 0;
    } catch (...) {
        return -2;
    }
}

}  // extern "C"
