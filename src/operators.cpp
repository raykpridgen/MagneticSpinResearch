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
#include <filesystem>
#include <unistd.h>
#include <limits.h>
#include <fstream>
#include <stdexcept>

using cplx = std::complex<double>;
using Mat2c = Eigen::Matrix<cplx, 2, 2>;
using Mat4c = Eigen::Matrix<cplx, 4, 4>;
using Mat8c = Eigen::Matrix<cplx, 8, 8>;

// Raw spin operators struct
struct SpinOperators
{
    Mat4c sx1, sx2, sy1, sy2, sz1, sz2;
};

// Kroneckered spin operator struct
struct KronOperators
{
    Mat8c Sx1, Sx2, Sy1, Sy2, Sz1, Sz2, Ix, Iy, Iz;
};

// Projection singleton and triplet
struct ProjOperators
{
    Mat8c Ps, Pt;
};

// Get directory of executable so writing always works correctly
std::filesystem::path getExecutableDir()
{
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);

    if (len <= 0)
    {
        throw std::runtime_error("Failed to read /proc/self/exe");
    }

    buf[len] = '\0';
    std::filesystem::path exe = buf;
    return exe.parent_path();
}

// Get base spin operators for computation, 4x4 matrices
SpinOperators get_spin_operators()
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
        0, 0, sq2_inv, sq2_inv,
        0, 0, -sq2_inv, sq2_inv,
        -sq2_inv, sq2_inv, 0, 0,
        -sq2_inv, -sq2_inv, 0, 0;

    // sy2
    operators.sy2 <<
        0, 0, -sq2_inv, -sq2_inv,
        0, 0, -sq2_inv, sq2_inv,
        sq2_inv, sq2_inv, 0, 0,
        sq2_inv, -sq2_inv, 0, 0;

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

// Compute lifted operators (kronecker product), 8x8 matrices
KronOperators get_lifted_operators(const SpinOperators& orig_operators)
{
    // Define struct
    KronOperators lifted;

    // Define identity matrix 4x4
    Mat2c id_matrix_2 = Mat2c::Identity();
    Mat4c id_matrix_4 = Mat4c::Identity();

    // Define pauli operators
    Mat2c pauli_x;
    Mat2c pauli_y;
    Mat2c pauli_z;

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
    lifted.Sx1 = Eigen::kroneckerProduct(orig_operators.sx1, id_matrix_2);
    lifted.Sx2 = Eigen::kroneckerProduct(orig_operators.sx2, id_matrix_2);
    lifted.Sy1 = Eigen::kroneckerProduct(orig_operators.sy1, id_matrix_2);
    lifted.Sy2 = Eigen::kroneckerProduct(orig_operators.sy2, id_matrix_2);
    lifted.Sz1 = Eigen::kroneckerProduct(orig_operators.sz1, id_matrix_2);
    lifted.Sz2 = Eigen::kroneckerProduct(orig_operators.sz2, id_matrix_2);

    // Create Ix, Iy, Iz
    lifted.Ix = Eigen::kroneckerProduct(id_matrix_4, pauli_x);
    lifted.Iy = Eigen::kroneckerProduct(id_matrix_4, pauli_y);
    lifted.Iz = Eigen::kroneckerProduct(id_matrix_4, pauli_z);
    
    return lifted;
}

// Get projection singleton and triplet operators
ProjOperators get_proj_operators()
{
    ProjOperators proj_ops;

    proj_ops.Ps = Eigen::MatrixXcd::Zero(8, 8);
    proj_ops.Ps(0, 0) = 1;
    proj_ops.Ps(1, 1) = 1;
    proj_ops.Pt = Mat8c::Identity() - proj_ops.Ps;
    return proj_ops;
}

// Construct H, the hamiltonian
Mat8c construct_hamiltonian(KronOperators krons, double g, double mu, double Bz, double a)
{
    // Construct terms
    Mat8c term_1 = krons.Sz1 + krons.Sz2;
    Mat8c term_2 = 
        (krons.Ix * krons.Sx1) +
        (krons.Iy * krons.Sy1) + 
        (krons.Iz * krons.Sz1);

    // Scale by constants
    term_1 = (g * mu * Bz) * term_1;
    term_2 = (g * mu * a) * term_2;

    return term_1 + term_2;
}

// Compute the Stochastic Liouville Equation
Mat8c construct_sle(Mat8c H, Mat8c Ps, Mat8c Pt, double rho, double Ks, double Kd)
{
    Mat8c G = Mat8c::Identity();
    return ((H * rho) - (rho * H)) + (0.5 * (Ks + Kd) * ((Pt * rho) + (rho * Ps))) + (0.5 * Kd * ((rho * Pt) * (Pt * rho)) + G);
}

// Test above functions
void test_operations(double g, double mu, double Bz, double a, double rho, double Ks, double Kd)
{
    auto base = getExecutableDir();
    std::filesystem::path outpath = base / ".." / "data" / "ops_check.txt";
    outpath = std::filesystem::canonical(outpath);
    
    std::ofstream out(outpath);

    if (!out)
    {
        throw std::runtime_error("Cannot open output file");
    }
    // Test spin ops
    SpinOperators raw_operators = get_spin_operators();
    out << "Raw Operators: " << "\n" << std::endl;
    out << "sx1 =\n" << raw_operators.sx1 << "\n" << std::endl;
    out << "sx2 =\n" << raw_operators.sx2 << "\n" << std::endl;
    out << "sy1 =\n" << raw_operators.sy1 << "\n" << std::endl;
    out << "sy2 =\n" << raw_operators.sy2 << "\n" << std::endl;
    out << "sz1 =\n" << raw_operators.sz1 << "\n" << std::endl;
    out << "sz2 =\n" << raw_operators.sz2 << "\n" << std::endl;

    // Test lifted ops
    KronOperators krons = get_lifted_operators(raw_operators);
    out << "Kronecker Oeprators: " << "\n" << std::endl;
    out << "Sx1 =\n" << krons.Sx1 << "\n" << std::endl;
    out << "Sx2 =\n" << krons.Sx2 << "\n" << std::endl;
    out << "Sy1 =\n" << krons.Sy1 << "\n" << std::endl;
    out << "Sy2 =\n" << krons.Sy2 << "\n" << std::endl;
    out << "Sz1 =\n" << krons.Sz1 << "\n" << std::endl;
    out << "Sz2 =\n" << krons.Sz2 << "\n" << std::endl;
    out << "Ix =\n" << krons.Ix << "\n" << std::endl;
    out << "Iy =\n" << krons.Iy << "\n" << std::endl;
    out << "Iz =\n" << krons.Iz << "\n" << std::endl;

    // Test proj ops
    ProjOperators proj_ops = get_proj_operators();
    out << "Ps =\n" << proj_ops.Ps << "\n" << std::endl;
    out << "Pt =\n" << proj_ops.Pt << "\n" << std::endl;

    // Test hamiltonian
    out << "g = " << g << ", mu = " << mu << ", a = " << a << ", Bz = " << Bz << "\n" << std::endl;
    Mat8c hamiltonian = construct_hamiltonian(krons, g, mu, Bz, a);
    out << "Hamiltonian =\n" << hamiltonian << "\n" << std::endl;

    // Test SLE function
    out << "rho = " << rho << ", Ks = " << Ks << ", Kd = " << Kd << "\n" << std::endl;
    Mat8c sle_8 = construct_sle(hamiltonian, proj_ops.Ps, proj_ops.Pt, rho, Ks, Kd);
    out << "Stochastic Liouville Equation =\n" << sle_8  << "\n" << std::endl;

    return;
}

int main(int argc, char* argv[])
{
    if (argc != 8)
    {
        std::cout << "Usage: ./program <g> <mu> <Bz> <a> <rho> <Ks> <Kd>" << std::endl;
        return 0;
    }
    double g = std::stod(argv[1]);
    double mu = std::stod(argv[2]);
    double Bz = std::stod(argv[3]);
    double a = std::stod(argv[4]);
    double rho = std::stod(argv[5]);
    double Ks = std::stod(argv[6]);
    double Kd = std::stod(argv[7]);
    test_operations(g, mu, Bz, a, rho, Ks, Kd);
    return 0;
}