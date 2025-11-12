#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <string>
#include <chrono>
#include <thread>

void readData(const std::string &matA_path, const std::string &matB_path, Eigen::MatrixXd* matA, Eigen::MatrixXd* matB)
{
    // Read inputs from file
    std::ifstream finA(matA_path);
    std::ifstream finB(matB_path);

    // Error if no data
    if (!finA)
    {
        throw std::runtime_error("Could not open " + matA_path);
    }
    if (!finB)
    {
        throw std::runtime_error("Could not open " + matB_path);
    }

    // Rows and Columns of A and b
    int rowsA, colsA, rowsB, colsB;
    
    // Read in Rows and Columns from first line
    finA >> rowsA >> colsA;
    finB >> rowsB >> colsB;

    // Size the matrix based on rows and columns
    matA->resize(rowsA, colsA);
    matB->resize(rowsB, colsB);

    // Fill data into matrix
    for (size_t i = 0; i < rowsA; ++i)
    {
        for (size_t j = 0; j < colsA; ++j)
        {
            finA >> (*matA)(i, j);
        }
    }   
    for (size_t i = 0; i < rowsB; ++i)
    {
        for (size_t j = 0; j < colsB; ++j)
        {
            finB >> (*matB)(i, j);
        }
    }   
}


int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "\nUsage: " << argv[0] << " <matrix_A_file> <matrix_b_file> <matrix_x_file> <timing_file>\n\n";
        return 1;
    }

    // Take files from arguments
    std:: string matrix_A_filename = argv[1];
    std:: string matrix_b_filename = argv[2];
    std:: string matrix_x_filename = argv[3];
    std:: string timing_file = argv[4];
    
    // Define a 3x3 matrix A
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;

    // Get data from file into struct
    readData(matrix_A_filename, matrix_b_filename, &A, &B);

    // Solve the system and time
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(B);
    auto end = std::chrono::high_resolution_clock::now();

    //std::cout << "A =\n" << A << std::endl;
    //std::cout << "b =\n" << B << std::endl;
    //std::cout << "x =\n" << x << std::endl;

    std::chrono::duration<double> elapsed = end - start;

    std::ofstream outFile(timing_file, std::ios::app);
    if (!outFile)
    {
        std::cerr << "Error opening file for writing\n";
        return 1;
    }
    outFile << "C\t\t" << A.rows() << "\t\t" << elapsed.count() << "\n";

    return 0;
}
