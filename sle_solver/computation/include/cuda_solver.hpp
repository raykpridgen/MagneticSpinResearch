#ifndef SLE_CUDA_SOLVER_HPP
#define SLE_CUDA_SOLVER_HPP

#include <complex>

/**
 * Check if CUDA GPU is available
 * @return true if at least one CUDA device is available
 */
bool check_cuda_available();

/**
 * Print CUDA device information to stdout
 */
void print_cuda_info();

/**
 * Solve complex double precision linear system Ax = b on GPU using cuSOLVER
 * Uses LU factorization with partial pivoting
 *
 * @param A_host    Input matrix A (n x n) in column-major format
 * @param b_host    Input vector b (n x 1)
 * @param x_host    Output vector x (n x 1), must be pre-allocated
 * @param n         Dimension of the system
 * 
 * @throws runtime_error if CUDA operations fail
 */
extern "C" void solve_cuda_complex(
    const std::complex<double>* A_host,
    const std::complex<double>* b_host,
    std::complex<double>* x_host,
    int n
);

#endif // SLE_CUDA_SOLVER_HPP
