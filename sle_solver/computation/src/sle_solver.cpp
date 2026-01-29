#include "sle_solver.hpp"
#include "hamiltonian.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

#ifdef USE_CUDA
#include "cuda_solver.hpp"
#endif

std::pair<MatXc, VecXc> construct_sle_linear_system(
    const MatXc& H,
    const MatXc& Ps,
    const MatXc& Pt,
    const PhysicalParams& params,
    int dim)
{
    // From Mathematica:
    // -(i/ℏ)[H,ρ] - (1/2)(Ks+Kd){Ps,ρ} - (1/2)Kd{Pt,ρ} + (1/dim)I = 0
    //
    // Using vectorization identities:
    //   vec([A,B]) = (I⊗A - A^T⊗I)·vec(B)
    //   vec({A,B}) = (I⊗A + A^T⊗I)·vec(B)

    const int matrix_size = dim * dim;
    MatXc M = MatXc::Zero(matrix_size, matrix_size);
    VecXc b = VecXc::Zero(matrix_size);

    MatXc I = MatXc::Identity(dim, dim);

    // Commutator operator: vec([H,ρ]) = (I⊗H - H^T⊗I)·vec(ρ)
    MatXc commutator_op =
        Eigen::kroneckerProduct(I, H).eval() -
        Eigen::kroneckerProduct(H.transpose(), I).eval();

    // Anticommutator operators: vec({P,ρ}) = (I⊗P + P^T⊗I)·vec(ρ)
    MatXc anticomm_s_op =
        Eigen::kroneckerProduct(I, Ps).eval() +
        Eigen::kroneckerProduct(Ps.transpose(), I).eval();

    MatXc anticomm_t_op =
        Eigen::kroneckerProduct(I, Pt).eval() +
        Eigen::kroneckerProduct(Pt.transpose(), I).eval();

    // Build M matrix
    // -(i/ℏ)[H,ρ] - (1/2)(Ks+Kd){Ps,ρ} - (1/2)Kd{Pt,ρ} = -(1/dim)I
    M = -cplx(0, 1.0 / params.hbar) * commutator_op
        - 0.5 * (params.Ks + params.Kd) * anticomm_s_op
        - 0.5 * params.Kd * anticomm_t_op;

    // Build RHS vector: -(1/dim)I vectorized
    MatXc rhs_matrix = (1.0 / dim) * I;
    Eigen::Map<const VecXc> b_vec(rhs_matrix.data(), matrix_size);
    b = -b_vec;

    return {M, b};
}

MatXc solve_steady_state_cpu(
    const MatXc& M,
    const VecXc& b,
    int dim,
    double* solve_time)
{
    auto start = std::chrono::high_resolution_clock::now();

    // Solve using Eigen's complete orthogonal decomposition
    VecXc x = M.completeOrthogonalDecomposition().solve(b);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (solve_time)
    {
        *solve_time = elapsed.count();
    }

    // Reshape vector to matrix (column-major)
    MatXc rho = Eigen::Map<const MatXc>(x.data(), dim, dim);

    // Normalize to ensure Tr(ρ) = 1
    cplx trace = rho.trace();
    if (std::abs(trace) > 1e-10)
    {
        rho = rho / trace;
    }

    // Enforce Hermiticity: ρ = (ρ + ρ†)/2
    rho = 0.5 * (rho + rho.adjoint());

    return rho;
}

MatXc solve_steady_state_gpu(
    const MatXc& M,
    const VecXc& b,
    int dim,
    double* solve_time)
{
#ifdef USE_CUDA
    VecXc x(b.size());

    auto start = std::chrono::high_resolution_clock::now();

    solve_cuda_complex(
        M.data(),
        b.data(),
        x.data(),
        M.rows()
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (solve_time)
    {
        *solve_time = elapsed.count();
    }

    // Reshape vector to matrix (column-major)
    MatXc rho = Eigen::Map<const MatXc>(x.data(), dim, dim);

    // Normalize to ensure Tr(ρ) = 1
    cplx trace = rho.trace();
    if (std::abs(trace) > 1e-10)
    {
        rho = rho / trace;
    }

    // Enforce Hermiticity: ρ = (ρ + ρ†)/2
    rho = 0.5 * (rho + rho.adjoint());

    return rho;
#else
    throw std::runtime_error("GPU solver not available: compiled without CUDA support");
#endif
}

double compute_singlet_population(const MatXc& Ps, const MatXc& rho)
{
    // Tr(Ps·ρ)
    MatXc product = Ps * rho;
    return product.trace().real();
}

DensityMatrixValidation validate_density_matrix(const MatXc& rho, double tol)
{
    DensityMatrixValidation result;

    // Check Hermiticity: ρ = ρ†
    MatXc rho_adjoint = rho.adjoint();
    double hermitian_error = (rho - rho_adjoint).norm();
    result.hermiticity_error = hermitian_error;
    result.is_hermitian = (hermitian_error < tol);

    // Check trace = 1
    cplx trace = rho.trace();
    result.trace_value = trace.real();
    result.is_normalized = (std::abs(result.trace_value - 1.0) < tol);

    // Check positive semi-definite (all eigenvalues ≥ 0)
    Eigen::SelfAdjointEigenSolver<MatXc> solver(rho);
    Eigen::VectorXd eigs = solver.eigenvalues();
    result.is_positive_semidefinite = true;
    for (int i = 0; i < eigs.size(); i++)
    {
        result.eigenvalues.push_back(eigs(i));
        if (eigs(i) < -tol)
        {
            result.is_positive_semidefinite = false;
        }
    }

    return result;
}

std::vector<SimulationPoint> run_sweep(
    const SystemConfig& config,
    const PhysicalParams& params,
    const SweepParams& sweep,
    bool use_gpu)
{
    std::vector<SimulationPoint> results;

    // Pre-compute operators (don't depend on Bz)
    SpinOperators spin_ops = get_spin_operators(config);
    KronOperators kron_ops = get_lifted_operators(spin_ops, config);
    ProjOperators proj_ops = get_proj_operators(config);

    // Sweep over magnetic field values
    for (double Bz = sweep.Bz_min; Bz <= sweep.Bz_max; Bz += sweep.Bz_step)
    {
        // Construct Hamiltonian for this Bz
        MatXc H = construct_hamiltonian(kron_ops, config, params, Bz);

        // Build linear system
        auto [M, b] = construct_sle_linear_system(H, proj_ops.Ps, proj_ops.Pt, params, config.total_dim);

        // Solve for steady-state density matrix
        MatXc rho;
        if (use_gpu)
        {
            rho = solve_steady_state_gpu(M, b, config.total_dim);
        }
        else
        {
            rho = solve_steady_state_cpu(M, b, config.total_dim);
        }

        // Compute singlet population
        double singlet_pop = compute_singlet_population(proj_ops.Ps, rho);

        // Apply fudge factor and store
        results.push_back({Bz, params.fudge * singlet_pop});
    }

    return results;
}
