#include <iostream>
#include <cuda_runtime.h>

// Determines size of one dimension of the matrix
#define N 4


// Kernel for kronecker product
__global__ void kronecker(const float *A, const float *B, float *C, int n)
{
    // Global row for a thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Global column for a thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Dimension for an output
    int outN = n * n;

    if (row < outN && col < outN)
    {
        // Break down row and col into components
        int i = row / n;   // row index of A
        int k = row % n;   // row index of B
        int j = col / n;   // col index of A
        int l = col % n;   // col index of B

        // Formula: C[(i*n+k),(j*n+l)] = A[i,j] * B[k,l]
        C[row * outN + col] = A[i * n + j] * B[k * n + l];
    }
}

int main()
{
    // Calculate size (n x n) matrix * size of float
    int size = N * N * sizeof(float);
    int outSize = N * N * N * N * sizeof(float);

    // Allocate host (CPU) memory, where the data starts at
    float h_A[N * N], h_B[N * N], h_C[N * N * N * N];

    // Initialize matrices A and B - fill the data within the host memory
    for (int i = 0; i < N * N; i++)
    {
        // Matrix A gets 0 - N*N for each entry
        h_A[i] = static_cast<float>(i);
        // Matrix B gets 0 - N repeating for each entry
        h_B[i] = static_cast<float>(i % N);
    }

    std::cout << "Matrix A:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_A[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Matrix B:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_B[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Allocate device memory - Now set up the data bounds for the GPU itself
    float *d_A, *d_B, *d_C;
    // Notice pointers and references, d_X points to portion of GPU ?
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, outSize);

    // Copy host data to device - CPU -> GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel: organize threads in 2D grid
    // 16 x 16 is a design decision
    // Min 32 threads executed at once, so block sizes mult of 32. 16 x 16 = 256 = 8 warps
    // dim3 is a struct that organizes up to 3D blocks
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    // <<<>>> is kernel launch syntax, () is kernel parameters
    // <<<>>> thread specs, () regular parameters
    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host - GPU -> CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result matrix C
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < N * N; i++)
    {
        for (int j = 0; j < N * not_eq; j++)
        {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}