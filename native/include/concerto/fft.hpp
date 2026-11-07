// PocketFFT backend (double precision).
// Replaces the educational radix-2 implementation with Martin Reinecke's
// high-performance pocketfft (used in NumPy/SciPy).
#pragma once

#include "concerto/pocketfft_hdronly.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace concerto {

// In-place complex-to-complex FFT.
inline void fft_inplace(std::complex<double>* x, std::size_t n) {
    pocketfft::shape_t shape{n};
    pocketfft::stride_t stride{sizeof(std::complex<double>)};
    pocketfft::shape_t axes{0};
    pocketfft::c2c(shape, stride, stride, axes, true, x, x, 1.0);
}

// Direct real-to-complex transform into caller-provided buffer of size n/2 + 1.
// Avoids converting real data to complex and computes only the positive bins directly.
inline void rfft_exec(const double* in, std::complex<double>* out, std::size_t n) {
    pocketfft::shape_t shape_in{n};
    pocketfft::stride_t stride_in{sizeof(double)};
    pocketfft::stride_t stride_out{sizeof(std::complex<double>)};
    pocketfft::r2c(shape_in, stride_in, stride_out, std::size_t(0), true, in, out, 1.0);
}

// Full real FFT of a real signal; returns bins 0..n/2 (equivalent to numpy.fft.rfft).
inline std::vector<std::complex<double>> rfft(const double* x, std::size_t n) {
    std::vector<std::complex<double>> out(n / 2 + 1);
    rfft_exec(x, out.data(), n);
    return out;
}

// numpy.hanning-compatible symmetric Hann window.
inline std::vector<double> hann(std::size_t n) {
    std::vector<double> w(n);
    const double denom = static_cast<double>(n - 1);
    for (std::size_t k = 0; k < n; ++k)
        w[k] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 *
                                    static_cast<double>(k) / denom);
    return w;
}

}  // namespace concerto
