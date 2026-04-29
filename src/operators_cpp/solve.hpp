#ifndef MAGSPIN_OPERATORS_CPP_SOLVE_HPP
#define MAGSPIN_OPERATORS_CPP_SOLVE_HPP

#include <complex>

namespace magspin {

void solve_cuda_complex(
    const std::complex<double>* A_host,
    const std::complex<double>* b_host,
    std::complex<double>* x_host,
    int n);

bool check_cuda_available();

} // namespace magspin

#endif
