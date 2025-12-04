#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <complex>
#include <stdexcept>
#include <iostream>
#include "cuda_solver.h"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuSOLVER error code: ") + std::to_string(status)); \
        } \
    } while(0)

/**
 * Solve complex double precision linear system Ax = b on GPU using cuSOLVER
 *
 * @param A_host: Input matrix A (n x n) in column-major format, std::complex<double>*
 * @param b_host: Input vector b (n x 1), std::complex<double>*
 * @param x_host: Output vector x (n x 1), std::complex<double>* (pre-allocated)
 * @param n: Dimension of the system
 *
 * Note: cuDoubleComplex and std::complex<double> are binary compatible
 */
extern "C" void solve_cuda_complex(
    const std::complex<double>* A_host,
    const std::complex<double>* b_host,
    std::complex<double>* x_host,
    int n)
{
    // Reinterpret as cuDoubleComplex for cuSOLVER
    const cuDoubleComplex* A_cplx = reinterpret_cast<const cuDoubleComplex*>(A_host);
    const cuDoubleComplex* b_cplx = reinterpret_cast<const cuDoubleComplex*>(b_host);
    cuDoubleComplex* x_cplx = reinterpret_cast<cuDoubleComplex*>(x_host);

    // Calculate sizes
    size_t size_A = n * n * sizeof(cuDoubleComplex);
    size_t size_b = n * sizeof(cuDoubleComplex);

    // Allocate device memory
    cuDoubleComplex *d_A = nullptr;
    cuDoubleComplex *d_b = nullptr;
    int *d_Ipiv = nullptr;
    int *d_info = nullptr;
    cuDoubleComplex *d_Workspace = nullptr;

    try {
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_b, size_b));
        CUDA_CHECK(cudaMalloc(&d_Ipiv, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, A_cplx, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b_cplx, size_b, cudaMemcpyHostToDevice));

        // Create cuSOLVER handle
        cusolverDnHandle_t handle;
        CUSOLVER_CHECK(cusolverDnCreate(&handle));

        // Query workspace size for LU factorization
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, n, n, d_A, n, &lwork));

        // Allocate workspace
        CUDA_CHECK(cudaMalloc(&d_Workspace, lwork * sizeof(cuDoubleComplex)));

        // Perform LU factorization: A = P*L*U
        CUSOLVER_CHECK(cusolverDnZgetrf(handle, n, n, d_A, n, d_Workspace, d_Ipiv, d_info));

        // Check if factorization succeeded
        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("LU factorization failed: info = " + std::to_string(h_info));
        }

        // Solve the system: A*x = b using the LU factorization
        CUSOLVER_CHECK(cusolverDnZgetrs(
            handle,
            CUBLAS_OP_N,  // No transpose
            n,            // Number of rows
            1,            // Number of right-hand sides (NRHS)
            d_A,          // Factorized matrix
            n,            // Leading dimension of A
            d_Ipiv,       // Pivot indices
            d_b,          // Right-hand side (solution will be stored here)
            n,            // Leading dimension of b
            d_info        // Info
        ));

        // Check if solve succeeded
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("Linear solve failed: info = " + std::to_string(h_info));
        }

        // Copy solution back to host (solution is in d_b)
        CUDA_CHECK(cudaMemcpy(x_cplx, d_b, size_b, cudaMemcpyDeviceToHost));

        // Cleanup
        CUSOLVER_CHECK(cusolverDnDestroy(handle));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_Workspace));
        CUDA_CHECK(cudaFree(d_Ipiv));
        CUDA_CHECK(cudaFree(d_info));

    } catch (...) {
        // Cleanup on error
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_Workspace) cudaFree(d_Workspace);
        if (d_Ipiv) cudaFree(d_Ipiv);
        if (d_info) cudaFree(d_info);
        throw;
    }
}

/**
 * Check if CUDA is available and print GPU info
 */
extern "C" bool check_cuda_available()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CUDA GPU detected: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;

    return true;
}
