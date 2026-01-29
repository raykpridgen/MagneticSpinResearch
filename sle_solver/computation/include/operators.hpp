#ifndef SLE_OPERATORS_HPP
#define SLE_OPERATORS_HPP

#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include "config.hpp"

using cplx = std::complex<double>;
using Mat2c = Eigen::Matrix<cplx, 2, 2>;
using Mat4c = Eigen::Matrix<cplx, 4, 4>;
using MatXc = Eigen::MatrixXcd;
using VecXc = Eigen::VectorXcd;

/**
 * Raw spin operators for N electrons
 * Each operator is (2^N x 2^N) dimensional
 */
struct SpinOperators
{
    std::vector<MatXc> sx;  // sx[i] for electron i
    std::vector<MatXc> sy;  // sy[i] for electron i
    std::vector<MatXc> sz;  // sz[i] for electron i
    int dim;
};

/**
 * Kronecker-lifted operators for full electron+nuclear system
 * Each operator is (2^N * 2^M) x (2^N * 2^M) dimensional
 */
struct KronOperators
{
    std::vector<MatXc> Sx;  // Sx[i] for electron i
    std::vector<MatXc> Sy;  // Sy[i] for electron i
    std::vector<MatXc> Sz;  // Sz[i] for electron i
    MatXc Ix, Iy, Iz;       // Nuclear spin operators
    int dim;
};

/**
 * Singlet and triplet projection operators
 * From Mathematica:
 *   Ps = DiagonalMatrix[Join[Table[1, {2^1}], Table[0, {4*2^1 - 2^1}]]]
 *   Pt = I - Ps
 */
struct ProjOperators
{
    MatXc Ps;  // Singlet projector
    MatXc Pt;  // Triplet projector
};

/**
 * Get Pauli matrices scaled by 1/2 for spin-1/2
 * sigma_x, sigma_y, sigma_z
 */
void get_pauli_matrices(Mat2c& sigma_x, Mat2c& sigma_y, Mat2c& sigma_z);

/**
 * Extend single-particle operator to N-particle space at position k
 * Returns I ⊗ ... ⊗ I ⊗ op ⊗ I ⊗ ... ⊗ I (op at position k, 0-indexed)
 */
MatXc extend_operator(const Mat2c& single_op, int n_particles, int position);

/**
 * Get base spin operators for N electrons
 * Returns (2^N x 2^N) matrices for each electron
 */
SpinOperators get_spin_operators(const SystemConfig& config);

/**
 * Compute lifted operators for full system (electron + nuclear)
 * Returns (2^N_e * 2^N_n) x (2^N_e * 2^N_n) matrices
 * 
 * From Mathematica:
 *   Sx1 = KroneckerProduct[sx1, IdentityMatrix[2]]
 *   Ix = KroneckerProduct[IdentityMatrix[4], 1/2 PauliMatrix[1]]
 */
KronOperators get_lifted_operators(const SpinOperators& spin_ops, const SystemConfig& config);

/**
 * Get singlet and triplet projection operators
 * 
 * From Mathematica:
 *   Ps = DiagonalMatrix[Join[Table[1, {2^1}], Table[0, {4*2^1 - 2^1}]]]
 *   Pt = IdentityMatrix[dim] - Ps
 */
ProjOperators get_proj_operators(const SystemConfig& config);

#endif // SLE_OPERATORS_HPP
