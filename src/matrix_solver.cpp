/**
 * Matrix Solver for Mathematica-exported Complex Matrices
 *
 * Parses complex matrices from Mathematica format and solves linear systems
 * using either CUDA (GPU) or Eigen (CPU).
 *
 * Usage: ./matrix_solver <input_file> [--cuda]
 *
 * Input format: Mathematica array with Complex(real, imag) notation
 * Output: Solution vector written to <input_file>_solution.txt
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <string>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>

#ifdef USE_CUDA
#include "cuda_solver.h"
#endif

namespace fs = std::filesystem;

using namespace std;
using namespace Eigen;

// Parse a complex number from Mathematica format: Complex(real, imag) or just a real number
complex<double> parseComplexNumber(const string& str) {
    string trimmed = str;
    // Remove whitespace
    trimmed.erase(remove(trimmed.begin(), trimmed.end(), ' '), trimmed.end());

    if (trimmed.find("Complex(") == 0) {
        // Extract real and imaginary parts
        size_t start = trimmed.find('(') + 1;
        size_t comma = trimmed.find(',', start);
        size_t end = trimmed.find(')', comma);

        double real = stod(trimmed.substr(start, comma - start));
        double imag = stod(trimmed.substr(comma + 1, end - comma - 1));

        return complex<double>(real, imag);
    } else {
        // Just a real number
        return complex<double>(stod(trimmed), 0.0);
    }
}

// Parse matrix from Mathematica format
MatrixXcd parseMatrixFile(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    // Read entire file
    stringstream buffer;
    buffer << file.rdbuf();
    string content = buffer.str();
    file.close();

    // Remove outer brackets and whitespace
    content.erase(remove(content.begin(), content.end(), '\n'), content.end());
    content.erase(remove(content.begin(), content.end(), '\r'), content.end());

    // Find the outermost [[ and ]]
    size_t start = content.find("[[");
    size_t end = content.rfind("]]");

    if (start == string::npos || end == string::npos) {
        throw runtime_error("Invalid matrix format: missing [[ or ]]");
    }

    // Extract matrix content
    string matrixContent = content.substr(start + 2, end - start - 2);

    // Parse rows - after stripping [[...]], format is: row1_data], [row2_data], [...], [rowN_data
    // Split by "], [" to get rows
    vector<vector<complex<double>>> rows;

    vector<string> rowStrings;
    size_t pos = 0;

    while (pos < matrixContent.length()) {
        // Find next row delimiter "], ["
        size_t delimPos = matrixContent.find("], [", pos);

        if (delimPos == string::npos) {
            // Last row (or only row)
            rowStrings.push_back(matrixContent.substr(pos));
            break;
        } else {
            // Extract this row
            rowStrings.push_back(matrixContent.substr(pos, delimPos - pos));
            pos = delimPos + 4;  // Skip past "], ["
        }
    }

    // Parse each row string
    for (const auto& rowStr : rowStrings) {
        vector<complex<double>> row;

        // Parse elements in row
        size_t elemStart = 0;
        int parenDepth = 0;

        for (size_t j = 0; j <= rowStr.length(); ++j) {
            if (j < rowStr.length()) {
                if (rowStr[j] == '(') parenDepth++;
                else if (rowStr[j] == ')') parenDepth--;
            }

            if ((j == rowStr.length() || (rowStr[j] == ',' && parenDepth == 0))) {
                string elemStr = rowStr.substr(elemStart, j - elemStart);
                // Trim whitespace
                size_t first = elemStr.find_first_not_of(" \t\n\r");
                size_t last = elemStr.find_last_not_of(" \t\n\r");

                if (first != string::npos && last != string::npos) {
                    elemStr = elemStr.substr(first, last - first + 1);
                    if (!elemStr.empty()) {
                        row.push_back(parseComplexNumber(elemStr));
                    }
                }
                elemStart = j + 1;
            }
        }

        if (!row.empty()) {
            rows.push_back(row);
        }
    }

    // Convert to Eigen matrix
    if (rows.empty()) {
        throw runtime_error("No rows parsed from matrix");
    }

    int n = rows.size();
    int m = rows[0].size();

    MatrixXcd matrix(n, m);
    for (int i = 0; i < n; ++i) {
        if (rows[i].size() != m) {
            throw runtime_error("Inconsistent row sizes in matrix");
        }
        for (int j = 0; j < m; ++j) {
            matrix(i, j) = rows[i][j];
        }
    }

    return matrix;
}

// Write solution vector to file in Mathematica format
void writeSolution(const string& filename, const VectorXcd& solution) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open output file: " + filename);
    }

    file << "{";
    for (int i = 0; i < solution.size(); ++i) {
        complex<double> val = solution(i);

        if (abs(val.imag()) < 1e-15) {
            // Real number
            file << val.real();
        } else {
            // Complex number - use Mathematica's square bracket notation
            file << "Complex[" << val.real() << ", " << val.imag() << "]";
        }

        if (i < solution.size() - 1) {
            file << ", ";
        }
    }
    file << "}" << endl;

    file.close();
}

// Solve using Eigen (CPU)
VectorXcd solveCPU(const MatrixXcd& A, const VectorXcd& b) {
    cout << "Solving with Eigen (CPU)..." << endl;

    auto start = chrono::high_resolution_clock::now();

    // Use LU decomposition for general square matrices
    VectorXcd x = A.fullPivLu().solve(b);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "CPU solve time: " << elapsed.count() << " seconds" << endl;

    // Check solution accuracy
    double relativeError = (A * x - b).norm() / b.norm();
    cout << "Relative error: " << relativeError << endl;

    return x;
}

#ifdef USE_CUDA
// Solve using CUDA (GPU)
VectorXcd solveCUDA(const MatrixXcd& A, const VectorXcd& b) {
    cout << "Solving with CUDA (GPU)..." << endl;

    if (!check_cuda_available()) {
        throw runtime_error("CUDA not available");
    }

    int n = A.rows();

    // Prepare arrays for CUDA (column-major format required by cuSOLVER)
    vector<complex<double>> A_data(n * n);
    vector<complex<double>> b_data(n);
    vector<complex<double>> x_data(n);

    // Convert to column-major
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_data[j * n + i] = A(i, j);
        }
        b_data[i] = b(i);
    }

    auto start = chrono::high_resolution_clock::now();

    // Call CUDA solver
    solve_cuda_complex(A_data.data(), b_data.data(), x_data.data(), n);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "CUDA solve time: " << elapsed.count() << " seconds" << endl;

    // Convert back to Eigen vector
    VectorXcd x(n);
    for (int i = 0; i < n; ++i) {
        x(i) = x_data[i];
    }

    // Check solution accuracy
    double relativeError = (A * x - b).norm() / b.norm();
    cout << "Relative error: " << relativeError << endl;

    return x;
}
#endif

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file> [--cuda]" << endl;
        cerr << endl;
        cerr << "Solves linear system Ax = b where A is read from input_file" << endl;
        cerr << "and b is automatically generated (ones vector normalized)." << endl;
        cerr << "Output is written to <input_file>_solution.txt" << endl;
        cerr << endl;
        cerr << "Options:" << endl;
        cerr << "  --cuda    Use CUDA GPU acceleration (requires CUDA build)" << endl;
        return 1;
    }

    string inputFile = argv[1];
    bool useCuda = false;

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--cuda") {
            #ifdef USE_CUDA
            useCuda = true;
            #else
            cerr << "Warning: CUDA support not compiled. Using CPU instead." << endl;
            #endif
        }
    }

    try {
        cout << "=======================================" << endl;
        cout << "Matrix Solver for Mathematica Matrices" << endl;
        cout << "=======================================" << endl;
        cout << endl;

        // Parse input matrix
        cout << "Reading matrix from: " << inputFile << endl;
        MatrixXcd A = parseMatrixFile(inputFile);

        int n = A.rows();
        int m = A.cols();

        cout << "Matrix dimensions: " << n << " x " << m << endl;

        if (n != m) {
            cerr << "Error: Matrix must be square" << endl;
            return 1;
        }

        // Create RHS vector b
        // For SLE problems, often we want to solve with a normalized constraint
        VectorXcd b = VectorXcd::Ones(n);
        b = b / sqrt(n);  // Normalize

        cout << "RHS vector: ones normalized to unit length" << endl;
        cout << endl;

        // Solve the system
        VectorXcd x;

        #ifdef USE_CUDA
        if (useCuda) {
            x = solveCUDA(A, b);
        } else {
            x = solveCPU(A, b);
        }
        #else
        x = solveCPU(A, b);
        #endif

        // Write solution to output file
        fs::path inputPath(inputFile);
        string outputFile = inputPath.parent_path().string() + "/" +
                          inputPath.stem().string() + "_solution.txt";

        cout << endl;
        cout << "Writing solution to: " << outputFile << endl;
        writeSolution(outputFile, x);

        cout << endl;
        cout << "Solution summary:" << endl;
        cout << "  Dimension: " << x.size() << endl;
        cout << "  Norm: " << x.norm() << endl;
        cout << "  Max element magnitude: " << x.cwiseAbs().maxCoeff() << endl;
        cout << endl;
        cout << "Done!" << endl;

        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
