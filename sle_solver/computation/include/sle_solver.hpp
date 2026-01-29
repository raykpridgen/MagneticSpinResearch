#ifndef SLE_SOLVER_HPP
#define SLE_SOLVER_HPP

#include <utility>
#include <vector>
#include "operators.hpp"
#include "config.hpp"

/**
 * Convert SLE operator equation into linear system M·vec(ρ) = b
 * 
 * From Mathematica:
 *   -(i/ℏ)[H,ρ] - (1/2)(Ks+Kd){Ps,ρ} - (1/2)Kd{Pt,ρ} + (1/dim)I = 0
 * 
 * Using vectorization identities:
 *   vec([H,ρ]) = (I⊗H - H^T⊗I)·vec(ρ)
 *   vec({P,ρ}) = (I⊗P + P^T⊗I)·vec(ρ)
 * 
 * @param H         Hamiltonian matrix
 * @param Ps        Singlet projection operator
 * @param Pt        Triplet projection operator
 * @param params    Physical parameters (Ks, Kd, hbar)
 * @param dim       Total Hilbert space dimension
 * @return          Pair of (M matrix, b vector)
 */
std::pair<MatXc, VecXc> construct_sle_linear_system(
    const MatXc& H,
    const MatXc& Ps,
    const MatXc& Pt,
    const PhysicalParams& params,
    int dim
);

/**
 * Solve the linear system using CPU (Eigen)
 * 
 * @param M             Coefficient matrix (dim^2 x dim^2)
 * @param b             Right-hand side vector (dim^2)
 * @param dim           Hilbert space dimension
 * @param solve_time    Output: time taken to solve (seconds)
 * @return              Steady-state density matrix (dim x dim)
 */
MatXc solve_steady_state_cpu(
    const MatXc& M,
    const VecXc& b,
    int dim,
    double* solve_time = nullptr
);

/**
 * Solve the linear system using GPU (CUDA cuSOLVER)
 * Throws runtime_error if CUDA is not available
 * 
 * @param M             Coefficient matrix (dim^2 x dim^2)
 * @param b             Right-hand side vector (dim^2)
 * @param dim           Hilbert space dimension
 * @param solve_time    Output: time taken to solve (seconds)
 * @return              Steady-state density matrix (dim x dim)
 */
MatXc solve_steady_state_gpu(
    const MatXc& M,
    const VecXc& b,
    int dim,
    double* solve_time = nullptr
);

/**
 * Compute singlet population: Tr(Ps·ρ)
 * 
 * @param Ps    Singlet projection operator
 * @param rho   Density matrix
 * @return      Real part of trace (singlet population)
 */
double compute_singlet_population(const MatXc& Ps, const MatXc& rho);

/**
 * Validation result for density matrix physical constraints
 */
struct DensityMatrixValidation
{
    bool is_hermitian;
    bool is_normalized;
    bool is_positive_semidefinite;
    double trace_value;
    double hermiticity_error;
    std::vector<double> eigenvalues;
};

/**
 * Validate that density matrix satisfies physical requirements
 * - Hermitian: ρ = ρ†
 * - Normalized: Tr(ρ) = 1
 * - Positive semi-definite: all eigenvalues ≥ 0
 */
DensityMatrixValidation validate_density_matrix(const MatXc& rho, double tol = 1e-6);

/**
 * Result from a single Bz point simulation
 */
struct SimulationPoint
{
    double Bz;
    double singlet_population;
};

/**
 * Run magnetic field sweep simulation
 * 
 * @param config        System configuration
 * @param params        Physical parameters
 * @param sweep         Sweep parameters (Bz range)
 * @param use_gpu       Use GPU solver (true) or CPU solver (false)
 * @return              Vector of (Bz, singlet_population) results
 */
std::vector<SimulationPoint> run_sweep(
    const SystemConfig& config,
    const PhysicalParams& params,
    const SweepParams& sweep,
    bool use_gpu
);

#endif // SLE_SOLVER_HPP
