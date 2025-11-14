#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <string>
#include <chrono>
#include <thread>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <cmath>
#include <complex>

using cplx = std::complex<double>;
using Mat4c = Eigen::Matrix<cplx, 4, 4>;

// Raw spin operators struct
struct SpinOperators
{
    Eigen::Matrix4d sx1, sx2, sz1, sz2;
    Mat4c sy1, sy2;
};

// Kroneckered spin operator struct
struct KronOperators
{
    Eigen::Matrix<double, 8, 8> Sx1, Sx2, Sz1, Sz2;
    Eigen::Matrix<cplx, 8, 8> Sy1, Sy2, Ix, Iy, Iz;

};

SpinOperators spin_operators()
{
    // 1 / sqrt(2)
    double sq2_inv = 1/(std::sqrt(2));

    // Struct to return
    SpinOperators operators;

    // sx1
    operators.sx1 << 
        0, 0, -sq2_inv, sq2_inv,
        0, 0, sq2_inv, sq2_inv,
        -sq2_inv, sq2_inv, 0, 0,
        sq2_inv, sq2_inv, 0, 0;

    // sx2
    operators.sx2 <<
        0, 0, sq2_inv, -sq2_inv,
        0, 0, sq2_inv, sq2_inv,
        sq2_inv, sq2_inv, 0, 0,
        -sq2_inv, sq2_inv, 0, 0;

    // sy1
    operators.sy1 <<
        cplx(0, 0), cplx(0, 0), cplx(sq2_inv, 0), cplx(sq2_inv, 0),
        cplx(0, 0), cplx(0, 0), cplx(-sq2_inv, 0), cplx(sq2_inv, 0),
        cplx(-sq2_inv, 0), cplx(sq2_inv, 0), cplx(0, 0), cplx(0, 0),
        cplx(-sq2_inv, 0), cplx(-sq2_inv, 0), cplx(0, 0), cplx(0, 0);

    // sy2
    operators.sy2 <<
        cplx(0, 0), cplx(0, 0), cplx(-sq2_inv, 0), cplx(-sq2_inv, 0),
        cplx(0, 0), cplx(0, 0), cplx(-sq2_inv, 0), cplx(sq2_inv, 0),
        cplx(sq2_inv, 0), cplx(sq2_inv, 0), cplx(0, 0), cplx(0, 0),
        cplx(sq2_inv, 0), cplx(-sq2_inv, 0), cplx(0, 0), cplx(0, 0);

    // sz1
    operators.sz1 <<
        0, 0.5, 0, 0,
        0.5, 0, 0, 0,
        0, 0, 0.5, 0,
        0, 0, 0, -0.5;

    // sz2
    operators.sz2 <<
        0, -0.5, 0, 0,
        -0.5, 0, 0, 0,
        0, 0, 0.5, 0,
        0, 0, 0, -0.5;

    // Apply scaling
    operators.sx1 = 0.5 * operators.sx1;
    operators.sx2 = 0.5 * operators.sx2;
    operators.sy1 = cplx(0, -0.5) * operators.sy1;
    operators.sy2 = cplx(0, -0.5) * operators.sy2;

    return operators;
}

KronOperators lifted_operators(const SpinOperators& orig_operators)
{
    // Define struct
    KronOperators lifted;

    // Define identity matrix 4x4
    Eigen::Matrix<double, 2, 2> id_matrix_2 = Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 4, 4> id_matrix_4 = Eigen::Matrix4d::Identity();

    // Define pauli operators
    Eigen::Matrix<double, 2, 2> pauli_x;
    Eigen::Matrix<cplx, 2, 2> pauli_y;
    Eigen::Matrix<double, 2, 2> pauli_z;

    pauli_x <<
        0, 1,
        1, 0;

    pauli_y << 
        0, cplx(0, -1),
        cplx(0, 1), 0;

    pauli_z <<
        1, 0,
        0, -1;

    pauli_x = 0.5 * pauli_x;
    pauli_y = 0.5 * pauli_y;
    pauli_z = 0.5 * pauli_z;

    // Apply Kronecker(operator, identity4)
    lifted.Sx1 = Eigen::kroneckerProduct(orig_operators.sx1, id_matrix_2).eval();
    lifted.Sx2 = Eigen::kroneckerProduct(orig_operators.sx2, id_matrix_2).eval();
    lifted.Sy1 = Eigen::kroneckerProduct(orig_operators.sy1, id_matrix_2).eval();
    lifted.Sy2 = Eigen::kroneckerProduct(orig_operators.sy2, id_matrix_2).eval();
    lifted.Sz1 = Eigen::kroneckerProduct(orig_operators.sz1, id_matrix_2).eval();
    lifted.Sz2 = Eigen::kroneckerProduct(orig_operators.sz2, id_matrix_2).eval();

    // Create Ix, Iy, Iz
    lifted.Ix = Eigen::kroneckerProduct(id_matrix_4, pauli_x).eval();
    lifted.Iy = Eigen::kroneckerProduct(id_matrix_4, pauli_y).eval();
    lifted.Iz = Eigen::kroneckerProduct(id_matrix_4, pauli_z).eval();
    
    return lifted;
}

Eigen::Matrix<double, 8, 8> add_matrices(const Eigen::)


void test_operations()
{
    SpinOperators raw_operators = spin_operators();

    std::cout << "Raw Operators: " << "\n" << std::endl;
    std::cout << "sx1 =\n" << raw_operators.sx1 << "\n" << std::endl;
    std::cout << "sx2 =\n" << raw_operators.sx2 << "\n" << std::endl;
    std::cout << "sy1 =\n" << raw_operators.sy1 << "\n" << std::endl;
    std::cout << "sy2 =\n" << raw_operators.sy2 << "\n" << std::endl;
    std::cout << "sz1 =\n" << raw_operators.sz1 << "\n" << std::endl;
    std::cout << "sz2 =\n" << raw_operators.sz2 << "\n" << std::endl;

    KronOperators krons = lifted_operators(raw_operators);

    std::cout << "Kronecker Oeprators: " << "\n" << std::endl;
    std::cout << "Sx1 =\n" << krons.Sx1 << "\n" << std::endl;
    std::cout << "Sx2 =\n" << krons.Sx2 << "\n" << std::endl;
    std::cout << "Sy1 =\n" << krons.Sy1 << "\n" << std::endl;
    std::cout << "Sy2 =\n" << krons.Sy2 << "\n" << std::endl;
    std::cout << "Sz1 =\n" << krons.Sz1 << "\n" << std::endl;
    std::cout << "Sz2 =\n" << krons.Sz2 << "\n" << std::endl;
    std::cout << "Ix =\n" << krons.Ix << "\n" << std::endl;
    std::cout << "Iy =\n" << krons.Iy << "\n" << std::endl;
    std::cout << "Iz =\n" << krons.Iz << "\n" << std::endl;

    return;
}

int main()
{
    test_operations();
    return 0;
}