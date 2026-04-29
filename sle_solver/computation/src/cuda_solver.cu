#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <complex>
#include <stdexcept>
#include <iostream>
#include "cuda_solver.hpp"

// error checking macro
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

bool check_cuda_available()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

void print_cuda_info()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0)
    {
        std::cout << "No CUDA devices found" << std::endl;
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "CUDA GPU: " << prop.name << std::endl;
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
}

extern "C" void solve_cuda_complex(
    const std::complex<double>* A_host,
    const std::complex<double>* b_host,
    std::complex<double>* x_host,
    int n)
{
    // Reinterpret as cuDoubleComplex for cuSOLVER
    // std::complex<double> and cuDoubleComplex are binary compatible
    const cuDoubleComplex* A_cplx = reinterpret_cast<const cuDoubleComplex*>(A_host);
    const cuDoubleComplex* b_cplx = reinterpret_cast<const cuDoubleComplex*>(b_host);
    cuDoubleComplex* x_cplx = reinterpret_cast<cuDoubleComplex*>(x_host);

    size_t size_A = n * n * sizeof(cuDoubleComplex);
    size_t size_b = n * sizeof(cuDoubleComplex);

    cuDoubleComplex *d_A = nullptr;
    cuDoubleComplex *d_b = nullptr;
    int *d_Ipiv = nullptr;
    int *d_info = nullptr;
    cuDoubleComplex *d_Workspace = nullptr;
    cusolverDnHandle_t handle = nullptr;

    try
    {
        // Allocate GPU memory
        CUDA_CHECK(cudaMalloc(&d_A, size_A));
        CUDA_CHECK(cudaMalloc(&d_b, size_b));
        CUDA_CHECK(cudaMalloc(&d_Ipiv, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, A_cplx, size_A, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b_cplx, size_b, cudaMemcpyHostToDevice));

        // Create cuSOLVER handle
        CUSOLVER_CHECK(cusolverDnCreate(&handle));

        // Query workspace size for LU factorization
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnZgetrf_bufferSize(handle, n, n, d_A, n, &lwork));

        // Allocate workspace
        CUDA_CHECK(cudaMalloc(&d_Workspace, lwork * sizeof(cuDoubleComplex)));

        // Perform LU factorization: A = P*L*U
        CUSOLVER_CHECK(cusolverDnZgetrf(handle, n, n, d_A, n, d_Workspace, d_Ipiv, d_info));

        // Check factorization result
        int h_info = 0;
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0)
        {
            throw std::runtime_error("LU factorization failed: info = " + std::to_string(h_info));
        }

        // Solve the system using the LU factorization
        CUSOLVER_CHECK(cusolverDnZgetrs(
            handle,
            CUBLAS_OP_N,
            n,
            1,
            d_A,
            n,
            d_Ipiv,
            d_b,
            n,
            d_info
        ));

        // Check solve result
        CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0)
        {
            throw std::runtime_error("Linear solve failed: info = " + std::to_string(h_info));
        }

        // Copy solution back to host
        CUDA_CHECK(cudaMemcpy(x_cplx, d_b, size_b, cudaMemcpyDeviceToHost));

        // Cleanup
        cusolverDnDestroy(handle);
        cudaFree(d_A);
        cudaFree(d_b);
        cudaFree(d_Workspace);
        cudaFree(d_Ipiv);
        cudaFree(d_info);
    }
    catch (...)
    {
        // Cleanup on error
        if (handle) cusolverDnDestroy(handle);
        if (d_A) cudaFree(d_A);
        if (d_b) cudaFree(d_b);
        if (d_Workspace) cudaFree(d_Workspace);
        if (d_Ipiv) cudaFree(d_Ipiv);
        if (d_info) cudaFree(d_info);
        throw;
    }
}
