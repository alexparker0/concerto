// Temporary debug harness for the renderer NaN issue (delete after fix).
#include "concerto/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace concerto;

int main() {
    UpmixRenderer::Params p{};
    p.sample_rate = 48000;
    p.surround_gain = 0.25;
    // Identity SOS section (b=[1,0,0], a=[1,0,0] in scipy layout).
    p.n_sos_sections = 1;
    p.sos[0] = 1; p.sos[1] = 0; p.sos[2] = 0; p.sos[3] = 1; p.sos[4] = 0; p.sos[5] = 0;
    const double a[3] = {0.68, -0.41, 0.23};
    for (int i = 0; i < 3; ++i) {
        p.a_sl[i] = a[i];
        p.a_sr[i] = -a[i];
        p.a_bl[i] = a[i];
        p.a_br[i] = -a[i];
    }
    p.decorrelate = 1;
    p.target_ratio = 0.35;
    p.rms_window_s = 0.4;
    p.gain_tau_s = 2.0;
    p.gain_min = 0;
    p.gain_max = 2;
    p.centre_mode = 0;
    p.mid_fill_gain = 0.3548;

    UpmixRenderer r(p);
    const std::size_t n = 16384;
    std::vector<float> L(n), R(n);
    for (std::size_t i = 0; i < n; ++i) {
        L[i] = 0.3f * std::sin(2.f * 3.14159265f * 440.f * static_cast<float>(i) / 48000.f);
        R[i] = 0.2f * L[i];
    }
    std::vector<float> fl(n), fr(n), fc(n), lfe(n), sl(n), sr(n), bl(n), br(n);
    r.process(L.data(), R.data(), n, fl.data(), fr.data(), fc.data(), lfe.data(),
              sl.data(), sr.data(), bl.data(), br.data());
    auto peak = [](const std::vector<float>& v) {
        float m = 0.f;
        for (float x : v) m = std::max(m, std::fabs(x));
        return m;
    };
    std::printf("SL[0..3]=%g %g %g %g\n", sl[0], sl[1], sl[2], sl[3]);
    std::printf("peaks: SL=%g SR=%g BL=%g BR=%g FL=%g\n",
                peak(sl), peak(sr), peak(bl), peak(br), peak(fl));
    // Stage isolation: S -> bandlimit -> allpass -> gain
    std::vector<float> s(n), s_bp(n), tmp(n), gout(n);
    for (std::size_t i = 0; i < n; ++i) s[i] = 0.5f * (L[i] - R[i]);
    std::printf("S[0]=%g finite=%d\n", s[0], std::isfinite(s[0]));

    SosFilter bp(p.sos, 1);
    bp.process(s.data(), s_bp.data(), n);
    std::printf("after bandlimit: %g finite=%d\n", s_bp[0], std::isfinite(s_bp[0]));

    AllPassBank bank({0.68, -0.41, 0.23});
    bank.process(s_bp.data(), tmp.data(), n);
    std::printf("after allpass: %g finite=%d\n", tmp[0], std::isfinite(tmp[0]));

    GainController::Params gp{};
    gp.sample_rate = 48000;
    GainController gc(gp);
    gc.process(tmp.data(), L.data(), gout.data(), n);
    std::printf("after gain: %g finite=%d\n", gout[0], std::isfinite(gout[0]));

    int bad = 0;
    for (float x : sl) if (!std::isfinite(x)) ++bad;
    std::printf("SL nonfinite: %d / %zu\n", bad, n);
    return 0;
}
