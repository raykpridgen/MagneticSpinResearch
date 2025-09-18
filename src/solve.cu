#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusolverSp.h>

/**
 * Dense Systems
 * cuSOLVERDn
 * 
 * Sparse Systems
 * cuSOLVERSp
 * 
 */

 /**
  * Dense:  Matrix where more than 50% of the entries are nonzero
  *   BLAS routines fast and simple, wastes time and memory on zeros
  * Sparse: Matrix where less than 20% of the entries are nonzero
  *   Save memory and compute, memory overhead to index
  * 
  * Gray zone 20-50% depends on size, pattern, solver
  * 
  * cuSolverRF: solve multiple sparse systems with the same sparsity pattern
  *   Avoids re-indexing an additional matrix by using same indexing
  */

// Determines size of one dimension of the matrix

struct MatrixInputs
{
    std::vector<float> A;
    std::vector<float> B;
    int rowsA, colsA, rowsB, colsB;
};

struct MatrixOutput
{
    std::vector<float> x;
    int rowsX, colsX;
};

// Read data from CSV
MatrixInputs readData(const std::string& matrixA_path, const std::string& matrixb_path)
{
    MatrixInputs data;

    // Read in Matrix A from CSV
    std::ifstream finA(matrixA_path);

    if (!finA)
    {
        throw std::runtime_error("Could not open " + matrixA_path);
    }
    
    finA >> data.rowsA >> data.colsA;

    data.A.resize(data.rowsA * data.colsA);
    for (int i = 0; i < (data.rowsA * data.colsA); i++)
    {
        finA >> data.A[i];
    }
    finA.close();

    // Repeat for b
    std::ifstream finB(matrixb_path);

    if (!finB)
    {
        throw std::runtime_error("Could not open " + matrixb_path);
    }

    int rowsB, colsB;
    finB >> rowsB >> colsB;

    if (rowsB != data.rowsA || colsB != 1)
    {
        throw std::runtime_error("Vector b dimensions do not match matrix A");
    }
    data.rowsB = rowsB;
    data.colsB = colsB;
    data.B.resize(rowsB);
    for (int i = 0; i < (data.rowsB * data.colsB); i++)
    {
        finB >> data.B[i];
    }
    finB.close();

    return data;
}

// Write output to a file
int writeData(MatrixOutput output, const std::string& matrixX_path)
{
    std::ofstream fout(matrixX_path);
    if (!fout)
    {
        throw std::runtime_error("Could not open " + matrixX_path + " for writing.");
    }

    for (int i = 0; i < output.rowsX; i++)
    {
        for (int j = 0; j < output.colsX; j++)
        {
            fout << output.x[i * output.colsX + j];
            if (j < output.colsX - 1)
            {
                fout << ",";
            }
        }
        fout << "\n";
    }
    fout.close();
    return 0;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_A_file> <matrix_b_file> <matrix_x_file>";
        return 1;
    }

    // Take files from arguments
    std:: string matrix_A_filename = argv[1];
    std:: string matrix_b_filename = argv[2];
    std:: string matrix_x_filename = argv[3];

    // Get data from files and place into structure
    MatrixInputs inputMatrices = readData(matrix_A_filename, matrix_b_filename);

    // Calculate sizes of matrices
    int size_A = inputMatrices.rowsA * inputMatrices.colsA * sizeof(float);
    int size_b = inputMatrices.rowsB * inputMatrices.colsB * sizeof(float);
    int size_x = inputMatrices.colsA * sizeof(float);


    // Allocate / Set data for hosts
    std::vector<float> h_A(inputMatrices.A);
    std::vector<float> h_b(inputMatrices.B);
    std::vector<float> h_x(inputMatrices.colsA);

    // Convert A to col major
    std::vector<float> h_A_col(h_A.size());
    for (int i = 0; i < inputMatrices.rowsA; ++i)
    {
        for (int j = 0; j < inputMatrices.colsA; ++j)
        {
            h_A_col[j * inputMatrices.rowsA + i] = h_A[i * inputMatrices.colsA + j];
        }
    }
    h_A = h_A_col;

    // Convert b to column-major (if multiple columns, otherwise optional)
    std::vector<float> h_b_col(h_b.size());
    for (int i = 0; i < inputMatrices.rowsB; ++i)
    {
        for (int j = 0; j < inputMatrices.colsB; ++j)
        {
            h_b_col[j * inputMatrices.rowsB + i] = h_b[i * inputMatrices.colsB + j];
        }
    }
    h_b = h_b_col;
        
    

    std::cout << "Matrix A:\n";
    for (int i = 0; i < inputMatrices.rowsA; i++)
    {
        for (int j = 0; j < inputMatrices.colsA; j++)
        {
            std::cout << h_A[i * inputMatrices.colsA + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Matrix B:\n";
    for (int i = 0; i < inputMatrices.rowsB; i++)
    {
        std::cout << h_b[i] << "\n";
    }
    std::cout << "\n";

    // Allocate device memory
    float *d_A, *d_b;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_b, size_b);

    // Copy A and b into device memory
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size_b, cudaMemcpyHostToDevice);
    
    // Create handle and arguments for solver
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int m = inputMatrices.rowsA;
    int n = inputMatrices.colsA;
    // Lead dimension of A (usually m)
    int lda = m;
    // Lead dimension of b (usually n)
    int ldb = m;
    
    // Allocate for pivot indices, out info
    int *d_Ipiv, *d_info;
    cudaMalloc(&d_Ipiv, n * sizeof(int));
    cudaMalloc(&d_info, sizeof(int));

    // Workspace size
    int lwork;
    cusolverDnSgetrf_bufferSize(handle, m, n, d_A, lda, &lwork);
    float *d_Workspace;
    cudaMalloc(&d_Workspace, lwork * sizeof(float));   

    // factor A - LU
    cusolverDnSgetrf(handle, m, n, d_A, lda, d_Workspace, d_Ipiv, d_info);

    // Solve
    cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A, lda, d_Ipiv, d_b, ldb, d_info);

    // Copy data back
    cudaMemcpy(h_x.data(), d_b, size_b, cudaMemcpyDeviceToHost);
    int h_info;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_info != 0)
    {
        std::cerr << "LU factorization / solve failed: info = " << h_info << "\n";
    }

    // Print result matrix C
    MatrixOutput outputData;
    outputData.x = h_x;
    outputData.rowsX = inputMatrices.rowsB;
    outputData.colsX = 1;
    writeData(outputData, matrix_x_filename);


    // Free device memory
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_Workspace);
    cudaFree(d_Ipiv);
    cudaFree(d_info);
    cusolverDnDestroy(handle);

    return 0;
}