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

// Include CUDA solver if available
#ifdef USE_CUDA
#include "cuda_solver.h"
#endif

using cplx = std::complex<double>;
using Mat2c = Eigen::Matrix<cplx, 2, 2>;
using Mat4c = Eigen::Matrix<cplx, 4, 4>;
using Mat8c = Eigen::Matrix<cplx, 8, 8>;
using Mat16c = Eigen::Matrix<cplx, 16, 16>;
using Mat32c = Eigen::Matrix<cplx, 32, 32>;
using MatXc = Eigen::MatrixXcd;  // Dynamic size matrix
using VecXc = Eigen::VectorXcd;  // Dynamic size vector

// Configuration for N-electron, M-nuclear spin system
struct SystemConfig
{
    int n_electrons;  // Number of electrons
    int n_nuclei;     // Number of nuclear spins
    int electron_dim; // 2^n_electrons
    int nuclear_dim;  // 2^n_nuclei
    int total_dim;    // electron_dim * nuclear_dim
    int matrix_size;  // total_dim^2 (vectorized density matrix size)

    SystemConfig(int n_e, int n_n)
        : n_electrons(n_e), n_nuclei(n_n),
          electron_dim(1 << n_e),  // 2^n_e
          nuclear_dim(1 << n_n),   // 2^n_n
          total_dim((1 << n_e) * (1 << n_n)),
          matrix_size(((1 << n_e) * (1 << n_n)) * ((1 << n_e) * (1 << n_n)))
    {}
};

// Raw spin operators struct (dynamic)
struct SpinOperators
{
    std::vector<MatXc> sx;  // sx[i] for electron i
    std::vector<MatXc> sy;  // sy[i] for electron i
    std::vector<MatXc> sz;  // sz[i] for electron i
    int dim;  // Dimension of each operator matrix
};

// Kroneckered spin operator struct (dynamic)
struct KronOperators
{
    std::vector<MatXc> Sx;  // Sx[i] for electron i
    std::vector<MatXc> Sy;  // Sy[i] for electron i
    std::vector<MatXc> Sz;  // Sz[i] for electron i
    MatXc Ix, Iy, Iz;       // Nuclear spin operators
    int dim;  // Dimension of each operator matrix
};

// Projection singleton and triplet
struct ProjOperators
{
    MatXc Ps, Pt;
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

// Helper: Construct single-particle spin-1/2 operators (2x2 Pauli matrices)
void get_pauli_matrices(Mat2c& sigma_x, Mat2c& sigma_y, Mat2c& sigma_z)
{
    sigma_x << 0, 1,
               1, 0;

    sigma_y << 0, cplx(0, -1),
               cplx(0, 1), 0;

    sigma_z << 1, 0,
               0, -1;
}

// Helper: Extend single-particle operator to N-particle space at position k
// Returns I ⊗ ... ⊗ I ⊗ op ⊗ I ⊗ ... ⊗ I (op at position k, 0-indexed)
MatXc extend_operator(const Mat2c& single_op, int n_particles, int position)
{
    // Start with the appropriate operator for the first position
    MatXc result;
    if (position == 0)
    {
        result = single_op;
    }
    else
    {
        result = Mat2c::Identity();
    }

    // Kronecker product with remaining operators
    for (int i = 1; i < n_particles; i++)
    {
        if (i == position)
        {
            result = Eigen::kroneckerProduct(result, single_op).eval();
        }
        else
        {
            result = Eigen::kroneckerProduct(result, Mat2c::Identity()).eval();
        }
    }

    return result;
}

// Get base spin operators for N electrons (returns 2^N × 2^N matrices)
SpinOperators get_spin_operators(const SystemConfig& config)
{
    SpinOperators operators;
    operators.dim = config.electron_dim;

    // Get Pauli matrices
    Mat2c pauli_x, pauli_y, pauli_z;
    get_pauli_matrices(pauli_x, pauli_y, pauli_z);

    // Scale by 1/2 for spin-1/2
    pauli_x = 0.5 * pauli_x;
    pauli_y = 0.5 * pauli_y;
    pauli_z = 0.5 * pauli_z;

    // Build operators for each electron
    for (int i = 0; i < config.n_electrons; i++)
    {
        // sx[i]: x-component of spin for electron i
        operators.sx.push_back(extend_operator(pauli_x, config.n_electrons, i));

        // sy[i]: y-component of spin for electron i
        operators.sy.push_back(extend_operator(pauli_y, config.n_electrons, i));

        // sz[i]: z-component of spin for electron i
        operators.sz.push_back(extend_operator(pauli_z, config.n_electrons, i));
    }

    return operators;
}

// Compute lifted operators for full system (electron + nuclear)
// Returns (2^N_e * 2^N_n) × (2^N_e * 2^N_n) matrices
KronOperators get_lifted_operators(const SpinOperators& orig_operators, const SystemConfig& config)
{
    KronOperators lifted;
    lifted.dim = config.total_dim;

    // Identity matrix for nuclear space
    MatXc id_nuclear = MatXc::Identity(config.nuclear_dim, config.nuclear_dim);

    // Lift electron operators: S_i = s_i ⊗ I_nuclear
    for (int i = 0; i < config.n_electrons; i++)
    {
        lifted.Sx.push_back(Eigen::kroneckerProduct(orig_operators.sx[i], id_nuclear).eval());
        lifted.Sy.push_back(Eigen::kroneckerProduct(orig_operators.sy[i], id_nuclear).eval());
        lifted.Sz.push_back(Eigen::kroneckerProduct(orig_operators.sz[i], id_nuclear).eval());
    }

    // Identity matrix for electron space
    MatXc id_electron = MatXc::Identity(config.electron_dim, config.electron_dim);

    // Nuclear spin operators: I_α = I_electron ⊗ σ_α/2
    Mat2c pauli_x, pauli_y, pauli_z;
    get_pauli_matrices(pauli_x, pauli_y, pauli_z);

    pauli_x = 0.5 * pauli_x;
    pauli_y = 0.5 * pauli_y;
    pauli_z = 0.5 * pauli_z;

    lifted.Ix = Eigen::kroneckerProduct(id_electron, pauli_x).eval();
    lifted.Iy = Eigen::kroneckerProduct(id_electron, pauli_y).eval();
    lifted.Iz = Eigen::kroneckerProduct(id_electron, pauli_z).eval();

    return lifted;
}

// Get projection singleton and triplet operators
// Ps projects onto first 2^M states (where M = number of nuclei)
ProjOperators get_proj_operators(const SystemConfig& config)
{
    ProjOperators proj_ops;

    int dim = config.total_dim;
    int singlet_states = config.nuclear_dim;  // 2^M

    proj_ops.Ps = MatXc::Zero(dim, dim);
    for (int i = 0; i < singlet_states; i++)
    {
        proj_ops.Ps(i, i) = 1;
    }

    proj_ops.Pt = MatXc::Identity(dim, dim) - proj_ops.Ps;
    return proj_ops;
}

// Construct H, the Hamiltonian for N electrons + M nuclei
// H = g·μB·Bz·(Σ Sz_i) + g·μB·a·(Ix·Sx1 + Iy·Sy1 + Iz·Sz1)
MatXc construct_hamiltonian(const KronOperators& krons, const SystemConfig& config,
                             double g, double mu, double Bz, double a)
{
    int dim = config.total_dim;

    // Term 1: Zeeman splitting for all electrons: g·μB·Bz·(Σ Sz_i)
    MatXc zeeman = MatXc::Zero(dim, dim);
    for (int i = 0; i < config.n_electrons; i++)
    {
        zeeman += krons.Sz[i];
    }
    zeeman = (g * mu * Bz) * zeeman;

    // Term 2: Hyperfine interaction with first electron only
    // g·μB·a·(Ix·Sx1 + Iy·Sy1 + Iz·Sz1)
    MatXc hyperfine = (krons.Ix * krons.Sx[0]) +
                      (krons.Iy * krons.Sy[0]) +
                      (krons.Iz * krons.Sz[0]);
    hyperfine = (g * mu * a) * hyperfine;

    // Combine both terms
    MatXc H = zeeman + hyperfine;

    return H;
}

// Convert SLE into linear system: M·vec(ρ) = b
std::pair<MatXc, VecXc> construct_sle_linear_system(
    const MatXc& H, const MatXc& Ps, const MatXc& Pt,
    double Ks, double Kd, double hbar, int dim)
{
    // need to convert the operator equation into M·x = b
    // where x = vec(rho) is the dim^2-element vector form of rho

    const int matrix_size = dim * dim;
    MatXc M = MatXc::Zero(matrix_size, matrix_size);
    VecXc b = VecXc::Zero(matrix_size);

    // Identity matrix
    MatXc I = MatXc::Identity(dim, dim);

    // Build the linear operator M using Kronecker products
    // vec([H,ρ]) = (I⊗H - H^T⊗I)·vec(ρ)
    MatXc commutator_op =
        Eigen::kroneckerProduct(I, H).eval() -
        Eigen::kroneckerProduct(H.transpose(), I).eval();

    // vec({Ps,ρ}) = (I⊗Ps + Ps^T⊗I)·vec(ρ)
    MatXc anticomm_s_op =
        Eigen::kroneckerProduct(I, Ps).eval() +
        Eigen::kroneckerProduct(Ps.transpose(), I).eval();

    // vec({Pt,ρ}) = (I⊗Pt + Pt^T⊗I)·vec(ρ)
    MatXc anticomm_t_op =
        Eigen::kroneckerProduct(I, Pt).eval() +
        Eigen::kroneckerProduct(Pt.transpose(), I).eval();

    // Combine into M: SLE equation
    // -(i/ℏ)[H,ρ] - (1/2)(Ks+Kd){Ps,ρ} - (1/2)Kd{Pt,ρ} + (1/dim)I = 0
    M = -cplx(0, 1.0/hbar) * commutator_op
        - 0.5 * (Ks + Kd) * anticomm_s_op
        - 0.5 * Kd * anticomm_t_op;

    // The right-hand side: -(1/dim)I, vectorized
    // Note: The Mathematica code adds (1/dim)I on the LHS, so RHS is zero minus this
    MatXc norm_matrix = (1.0/dim) * I;
    Eigen::Map<const VecXc> b_vec(norm_matrix.data(), matrix_size);
    b = -b_vec;

    return {M, b};
}

// Solve the system and reshape to dim×dim matrix
MatXc solve_steady_state(
    const MatXc& M,
    const VecXc& b,
    int dim,
    bool use_cuda = false,
    double* solve_time_ptr = nullptr
)
{
    VecXc x(b.size());

    auto start = std::chrono::high_resolution_clock::now();

#ifdef USE_CUDA
    if (use_cuda)
    {
        try {
            // Use CUDA solver
            // Note: Eigen stores data in column-major format by default, which matches cuSOLVER
            solve_cuda_complex(
                M.data(),  // Matrix A (column-major)
                b.data(),  // Vector b
                x.data(),  // Output x (pre-allocated)
                M.rows()   // Dimension
            );
        }
        catch (const std::exception& e)
        {
            std::cerr << "CUDA solver failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU solver..." << std::endl;
            x = M.completeOrthogonalDecomposition().solve(b);
        }
    }
    else
#endif
    {
        // Use Eigen's CPU solver
        x = M.completeOrthogonalDecomposition().solve(b);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (solve_time_ptr)
    {
        *solve_time_ptr = elapsed.count();
    }

    // Reshape the dim²-element vector back to dim×dim matrix
    // vec(ρ) stores matrix in column-major order
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

// Validation functions for density matrix
struct DensityMatrixValidation
{
    bool is_hermitian;
    bool is_normalized;
    bool is_positive_semidefinite;
    double trace_value;
    double hermiticity_error;
    std::vector<double> eigenvalues;
};

// Validate that density matrix satisfies physical requirements
DensityMatrixValidation validate_density_matrix(const MatXc& rho, double tol=1e-6)
{
    DensityMatrixValidation result;

    // Check Hermiticity: ρ = ρ†
    MatXc rho_conjugate_transpose = rho.adjoint();
    double hermitian_error = (rho - rho_conjugate_transpose).norm();
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

// Compute singlet population: Tr(Ps·rho)
double compute_singlet_population(const MatXc& Ps, const MatXc& rho)
{
    MatXc product = Ps * rho;
    return product.trace().real();  // Take real part of the trace
}

// Main simulation loop
std::vector<std::pair<double, double>> sweep_magnetic_field(
    const SystemConfig& config,
    double g, double mu, double a,
    double Ks, double Kd, double hbar,
    double Bz_min, double Bz_max, double Bz_step,
    double fudge = 1.0)
{
    std::vector<std::pair<double, double>> results;

    // Get operators (these don't change with Bz)
    SpinOperators raw_ops = get_spin_operators(config);
    KronOperators kron_ops = get_lifted_operators(raw_ops, config);
    ProjOperators proj_ops = get_proj_operators(config);

    // Loop over magnetic field values
    for (double Bz = Bz_min; Bz <= Bz_max; Bz += Bz_step)
    {
        // Construct Hamiltonian for this Bz
        MatXc H = construct_hamiltonian(kron_ops, config, g, mu, Bz, a);

        // Solve for steady-state density matrix
        auto system = construct_sle_linear_system(
            H, proj_ops.Ps, proj_ops.Pt, Ks, Kd, hbar, config.total_dim);

        MatXc rho = solve_steady_state(system.first, system.second, config.total_dim);

        // Compute singlet population
        double singlet_pop = compute_singlet_population(proj_ops.Ps, rho);

        // Apply fudge factor and store result
        results.push_back({Bz, fudge * singlet_pop});
    }

    return results;
}

// Test above functions
void test_operations(const SystemConfig& config, double g, double mu, double Bz, double a, double Ks, double Kd, double hbar, bool use_cuda = false)
{
    auto base = getExecutableDir();
    std::filesystem::path outpath = base / ".." / "data" / "ops_check.txt";
    outpath = std::filesystem::canonical(outpath);

    std::ofstream out(outpath);

    if (!out)
    {
        throw std::runtime_error("Cannot open output file");
    }

    // Print system configuration
    out << "System Configuration:" << "\n";
    out << "  Number of electrons: " << config.n_electrons << "\n";
    out << "  Number of nuclei: " << config.n_nuclei << "\n";
    out << "  Electron dimension: " << config.electron_dim << "\n";
    out << "  Nuclear dimension: " << config.nuclear_dim << "\n";
    out << "  Total dimension: " << config.total_dim << "\n";
    out << "  Linear system size: " << config.matrix_size << "\n" << std::endl;

    // Test spin ops
    SpinOperators raw_operators = get_spin_operators(config);
    out << "Raw Operators (dimension " << raw_operators.dim << "x" << raw_operators.dim << "): " << "\n" << std::endl;
    for (int i = 0; i < config.n_electrons; i++)
    {
        out << "sx[" << i << "] =\n" << raw_operators.sx[i] << "\n" << std::endl;
        out << "sy[" << i << "] =\n" << raw_operators.sy[i] << "\n" << std::endl;
        out << "sz[" << i << "] =\n" << raw_operators.sz[i] << "\n" << std::endl;
    }

    // Test lifted ops
    KronOperators krons = get_lifted_operators(raw_operators, config);
    out << "Kronecker Operators (dimension " << krons.dim << "x" << krons.dim << "): " << "\n" << std::endl;
    for (int i = 0; i < config.n_electrons; i++)
    {
        out << "Sx[" << i << "] =\n" << krons.Sx[i] << "\n" << std::endl;
        out << "Sy[" << i << "] =\n" << krons.Sy[i] << "\n" << std::endl;
        out << "Sz[" << i << "] =\n" << krons.Sz[i] << "\n" << std::endl;
    }
    out << "Ix =\n" << krons.Ix << "\n" << std::endl;
    out << "Iy =\n" << krons.Iy << "\n" << std::endl;
    out << "Iz =\n" << krons.Iz << "\n" << std::endl;

    // Test proj ops
    ProjOperators proj_ops = get_proj_operators(config);
    out << "Ps =\n" << proj_ops.Ps << "\n" << std::endl;
    out << "Pt =\n" << proj_ops.Pt << "\n" << std::endl;

    // Test hamiltonian
    out << "g = " << g << ", mu = " << mu << ", a = " << a << ", Bz = " << Bz << "\n" << std::endl;
    MatXc hamiltonian = construct_hamiltonian(krons, config, g, mu, Bz, a);
    out << "Hamiltonian =\n" << hamiltonian << "\n" << std::endl;

    // Test SLE function
    out <<"Ks = " << Ks << ", Kd = " << Kd << ", hbar = " << hbar <<  "\n" << std::endl;
    auto system = construct_sle_linear_system(hamiltonian, proj_ops.Ps, proj_ops.Pt, Ks, Kd, hbar, config.total_dim);
    out << "M =\n" << system.first << "\n\nb =\n" << system.second  << "\n" << std::endl;

    // Solve for steady-state density matrix
    out << "Solving for steady-state density matrix";
    if (use_cuda)
    {
        out << " (using CUDA GPU)";
    }
    else
    {
        out << " (using CPU)";
    }
    out << "..." << "\n" << std::endl;

    double solve_time = 0.0;
    MatXc rho = solve_steady_state(system.first, system.second, config.total_dim, use_cuda, &solve_time);

    out << "Solve time: " << solve_time << " seconds\n" << std::endl;
    out << "Density matrix ρ =\n" << rho << "\n" << std::endl;

    // Validate the density matrix
    out << "Validating density matrix properties..." << "\n" << std::endl;
    DensityMatrixValidation validation = validate_density_matrix(rho);

    out << "  Is Hermitian: " << (validation.is_hermitian ? "YES" : "NO") << std::endl;
    out << "    Hermiticity error: " << validation.hermiticity_error << "\n" << std::endl;

    out << "  Is Normalized (Tr(ρ) = 1): " << (validation.is_normalized ? "YES" : "NO") << std::endl;
    out << "    Trace value: " << validation.trace_value << "\n" << std::endl;

    out << "  Is Positive Semi-Definite: " << (validation.is_positive_semidefinite ? "YES" : "NO") << std::endl;
    out << "    Eigenvalues: ";
    for (size_t i = 0; i < validation.eigenvalues.size(); i++)
    {
        out << validation.eigenvalues[i];
        if (i < validation.eigenvalues.size() - 1) out << ", ";
    }
    out << "\n" << std::endl;

    // Compute singlet population
    double singlet_pop = compute_singlet_population(proj_ops.Ps, rho);
    out << "Singlet Population Tr(Ps·ρ) = " << singlet_pop << "\n" << std::endl;

    return;
}

#ifndef NO_MAIN
int main(int argc, char* argv[])
{
    if (argc < 9 || argc > 10)
    {
        std::cout << "Usage: ./program <n_electrons> <g> <mu> <Bz> <a> <Ks> <Kd> <hbar> [--cuda]" << std::endl;
        std::cout << "  n_electrons: Number of coupled electrons (1, 2, 3, or 4)" << std::endl;
        std::cout << "  g: g-factor" << std::endl;
        std::cout << "  mu: Bohr magneton (eV/mT)" << std::endl;
        std::cout << "  Bz: Magnetic field (mT)" << std::endl;
        std::cout << "  a: Hyperfine coupling constant" << std::endl;
        std::cout << "  Ks: Singlet recombination rate" << std::endl;
        std::cout << "  Kd: Dephasing rate" << std::endl;
        std::cout << "  hbar: Reduced Planck constant (eV·s)" << std::endl;
        std::cout << "  --cuda: (optional) Use CUDA GPU solver" << std::endl;
        return 0;
    }

    int n_electrons = std::stoi(argv[1]);
    double g = std::stod(argv[2]);
    double mu = std::stod(argv[3]);
    double Bz = std::stod(argv[4]);
    double a = std::stod(argv[5]);
    double Ks = std::stod(argv[6]);
    double Kd = std::stod(argv[7]);
    double hbar = std::stod(argv[8]);

    // Check for --cuda flag
    bool use_cuda = false;
    if (argc == 10 && std::string(argv[9]) == "--cuda")
    {
        use_cuda = true;
#ifdef USE_CUDA
        if (check_cuda_available())
        {
            std::cout << "CUDA GPU acceleration enabled\n" << std::endl;
        }
        else
        {
            std::cerr << "CUDA requested but no GPU available. Using CPU." << std::endl;
            use_cuda = false;
        }
#else
        std::cerr << "Warning: CUDA support not compiled. Using CPU solver." << std::endl;
        std::cerr << "To enable CUDA: compile with -DUSE_CUDA and link with -lcusolver -lcudart\n" << std::endl;
        use_cuda = false;
#endif
    }

    // Validate n_electrons
    if (n_electrons < 1 || n_electrons > 4)
    {
        std::cerr << "Error: n_electrons must be between 1 and 4" << std::endl;
        return 1;
    }

    // Create system configuration (n_electrons coupled electrons, 1 nuclear spin)
    SystemConfig config(n_electrons, 1);

    std::cout << "Simulating " << n_electrons << "-electron system:" << std::endl;
    std::cout << "  Total Hilbert space dimension: " << config.total_dim << std::endl;
    std::cout << "  Density matrix size: " << config.total_dim << "x" << config.total_dim << std::endl;
    std::cout << "  Linear system size: " << config.matrix_size << "x" << config.matrix_size << std::endl;

    test_operations(config, g, mu, Bz, a, Ks, Kd, hbar, use_cuda);

    std::cout << "Results written to data/ops_check.txt" << std::endl;

    return 0;
}
#endif // NO_MAIN