#ifndef CUDA_SOLVER_H
#define CUDA_SOLVER_H

#include <complex>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Solve complex double precision linear system Ax = b on GPU using cuSOLVER
 *
 * @param A_host: Input matrix A (n x n) in column-major format
 * @param b_host: Input vector b (n x 1)
 * @param x_host: Output vector x (n x 1), pre-allocated
 * @param n: Dimension of the system
 *
 * Note: All arrays use std::complex<double>, which is binary-compatible with cuDoubleComplex
 */
void solve_cuda_complex(
    const std::complex<double>* A_host,
    const std::complex<double>* b_host,
    std::complex<double>* x_host,
    int n);

/**
 * Check if CUDA is available and print GPU information
 * @return true if CUDA GPU is available, false otherwise
 */
bool check_cuda_available();

#ifdef __cplusplus
}
#endif

#endif // CUDA_SOLVER_H
