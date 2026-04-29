#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <complex>
#include <string>
#include <stdexcept>

#include "solve.hpp"

namespace {

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err = call;                                                     \
        if (err != cudaSuccess) {                                                   \
            throw std::runtime_error(std::string("CUDA error: ") +                  \
                                     cudaGetErrorString(err));                      \
        }                                                                           \
    } while (0)

#define CUSOLVER_CHECK(call)                                                        \
    do {                                                                            \
        cusolverStatus_t status = call;                                             \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                    \
            throw std::runtime_error("cuSOLVER error code: " +                      \
                                     std::to_string(static_cast<int>(status)));     \
        }                                                                           \
    } while (0)

} // namespace

namespace magspin {

void solve_cuda_complex(
    const std::complex<double>* A_host,
    const std::complex<double>* b_host,
    std::complex<double>* x_host,
    int n) {
    if (n <= 0) {
        throw std::invalid_argument("solve_cuda_complex requires n > 0");
    }

    const cuDoubleComplex* A_cplx = reinterpret_cast<const cuDoubleComplex*>(A_host);
    const cuDoubleComplex* b_cplx = reinterpret_cast<const cuDoubleComplex*>(b_host);
    cuDoubleComplex* x_cplx = reinterpret_cast<cuDoubleComplex*>(x_host);

    const size_t size_A = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(cuDoubleComplex);
    const size_t size_b = static_cast<size_t>(n) * sizeof(cuDoubleComplex);

    cuDoubleComplex *d_A = nullptr, *d_b = nullptr, *d_workspace = nullptr;
    int *d_ipiv = nullptr, *d_info = nullptr;
    cusolverDnHandle_t handle = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_b, size_b));
        CUDA_CHECK(cudaMalloc(&d_ipiv, static_cast<size_t>(n) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_A, A_cplx, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b_cplx, size_b, cudaMemcpyHostToDevice));

        CUSOLVER_CHECK(cusolverDnCreate(&handle));

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, n, n, d_A, n, &lwork));
        CUDA_CHECK(cudaMalloc(&d_workspace, static_cast<size_t>(lwork) * sizeof(cuDoubleComplex)));

        CUSOLVER_CHECK(cusolverDnZgetrf(handle, n, n, d_A, n, d_workspace, d_ipiv, d_info));

        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("LU factorization failed: info = " + std::to_string(h_info));
        }

        CUSOLVER_CHECK(cusolverDnZgetrs(handle, CUBLAS_OP_N, n, 1, d_A, n, d_ipiv, d_b, n, d_info));
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("Linear solve failed: info = " + std::to_string(h_info));
        }

        CUDA_CHECK(cudaMemcpy(x_cplx, d_b, size_b, cudaMemcpyDeviceToHost));

        CUSOLVER_CHECK(cusolverDnDestroy(handle));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_workspace));
        CUDA_CHECK(cudaFree(d_ipiv));
        CUDA_CHECK(cudaFree(d_info));
    } catch (...) {
        if (handle != nullptr) {
            cusolverDnDestroy(handle);
        }
        if (d_A != nullptr) cudaFree(d_A);
        if (d_b != nullptr) cudaFree(d_b);
        if (d_workspace != nullptr) cudaFree(d_workspace);
        if (d_ipiv != nullptr) cudaFree(d_ipiv);
        if (d_info != nullptr) cudaFree(d_info);
        throw;
    }
}

bool check_cuda_available() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

} // namespace magspin