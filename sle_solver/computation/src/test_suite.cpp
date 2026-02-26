/**
 * Test suite for SLE Solver computation functions
 *
 * Each test exercises a function at the fundamental level using contrived
 * inputs with known expected outputs, rather than building the full physics
 * pipeline.  This ensures that each function's core logic is verified
 * independently.
 *
 * Tests cover:
 *   - operators.cpp   (Pauli matrices, extend_operator, spin/lifted/proj operators)
 *   - hamiltonian.cpp  (construct_hamiltonian)
 *   - sle_solver.cpp   (construct_sle_linear_system, solve_steady_state_cpu,
 *                        compute_singlet_population, validate_density_matrix, run_sweep)
 *
 * Usage:
 *   ./sle_test_suite              Run tests, console summary only
 *   ./sle_test_suite --report     Run tests and write detailed report to output/
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <iomanip>
#include <chrono>
#include <functional>
#include <stdexcept>

#include "config.hpp"
#include "operators.hpp"
#include "hamiltonian.hpp"
#include "sle_solver.hpp"

// ---------------------------------------------------------------------------
// Lightweight test framework
// ---------------------------------------------------------------------------

struct TestResult
{
    std::string name;
    bool passed;
    std::string detail;   // verbose input/output info
    std::string failure;  // only set on failure
};

static std::vector<TestResult> g_results;
static std::ostringstream g_detail;  // accumulator for current test detail

static const double TOL = 1e-10;

#define ASSERT_TRUE(cond, msg)                                        \
    do {                                                              \
        if (!(cond)) {                                                \
            throw std::runtime_error(std::string("ASSERT_TRUE failed: ") + (msg)); \
        }                                                             \
    } while(0)

#define ASSERT_NEAR(a, b, tol, msg)                                   \
    do {                                                              \
        if (std::abs((a) - (b)) > (tol)) {                            \
            std::ostringstream _os;                                   \
            _os << "ASSERT_NEAR failed: " << (msg)                    \
                << " | got " << (a) << ", expected " << (b)           \
                << ", diff " << std::abs((a)-(b));                    \
            throw std::runtime_error(_os.str());                      \
        }                                                             \
    } while(0)

#define ASSERT_CPLX_NEAR(a, b, tol, msg)                              \
    do {                                                              \
        if (std::abs((a) - (b)) > (tol)) {                            \
            std::ostringstream _os;                                   \
            _os << "ASSERT_CPLX_NEAR failed: " << (msg)              \
                << " | got " << (a) << ", expected " << (b)           \
                << ", diff " << std::abs((a)-(b));                    \
            throw std::runtime_error(_os.str());                      \
        }                                                             \
    } while(0)

static void run_test(const std::string& name, std::function<void()> fn)
{
    g_detail.str("");
    g_detail.clear();
    TestResult r;
    r.name = name;
    try
    {
        fn();
        r.passed = true;
    }
    catch (const std::exception& e)
    {
        r.passed = false;
        r.failure = e.what();
    }
    r.detail = g_detail.str();
    g_results.push_back(r);
}

// Helper: format a small matrix for the report
static std::string mat_to_string(const MatXc& m, const std::string& label = "")
{
    std::ostringstream os;
    if (!label.empty()) os << label << ":\n";
    Eigen::IOFormat fmt(6, 0, ", ", "\n", "  [", "]");
    os << m.format(fmt) << "\n";
    return os.str();
}

[[maybe_unused]]
static std::string vec_to_string(const VecXc& v, const std::string& label = "")
{
    std::ostringstream os;
    if (!label.empty()) os << label << ":\n";
    Eigen::IOFormat fmt(6, 0, ", ", "\n", "  [", "]");
    os << v.transpose().format(fmt) << "\n";
    return os.str();
}

// ===================================================================
//  OPERATOR TESTS  (already fundamental — each uses minimal inputs)
// ===================================================================

static void test_pauli_matrices()
{
    Mat2c sx, sy, sz;
    get_pauli_matrices(sx, sy, sz);

    g_detail << "=== get_pauli_matrices ===\n";
    g_detail << mat_to_string(sx, "sigma_x / 2");
    g_detail << mat_to_string(sy, "sigma_y / 2");
    g_detail << mat_to_string(sz, "sigma_z / 2");

    // Check specific entries of sigma_x/2
    ASSERT_CPLX_NEAR(sx(0,0), cplx(0,0), TOL, "sx(0,0)");
    ASSERT_CPLX_NEAR(sx(0,1), cplx(0.5,0), TOL, "sx(0,1)");
    ASSERT_CPLX_NEAR(sx(1,0), cplx(0.5,0), TOL, "sx(1,0)");
    ASSERT_CPLX_NEAR(sx(1,1), cplx(0,0), TOL, "sx(1,1)");

    // Check specific entries of sigma_y/2
    ASSERT_CPLX_NEAR(sy(0,0), cplx(0,0), TOL, "sy(0,0)");
    ASSERT_CPLX_NEAR(sy(0,1), cplx(0,-0.5), TOL, "sy(0,1)");
    ASSERT_CPLX_NEAR(sy(1,0), cplx(0,0.5), TOL, "sy(1,0)");
    ASSERT_CPLX_NEAR(sy(1,1), cplx(0,0), TOL, "sy(1,1)");

    // Check specific entries of sigma_z/2
    ASSERT_CPLX_NEAR(sz(0,0), cplx(0.5,0), TOL, "sz(0,0)");
    ASSERT_CPLX_NEAR(sz(0,1), cplx(0,0), TOL, "sz(0,1)");
    ASSERT_CPLX_NEAR(sz(1,0), cplx(0,0), TOL, "sz(1,0)");
    ASSERT_CPLX_NEAR(sz(1,1), cplx(-0.5,0), TOL, "sz(1,1)");

    // Hermiticity: sigma_i = sigma_i^dagger
    ASSERT_NEAR((sx - sx.adjoint()).norm(), 0.0, TOL, "sx Hermitian");
    ASSERT_NEAR((sy - sy.adjoint()).norm(), 0.0, TOL, "sy Hermitian");
    ASSERT_NEAR((sz - sz.adjoint()).norm(), 0.0, TOL, "sz Hermitian");

    // Trace should be zero for all Pauli matrices
    ASSERT_NEAR(std::abs(sx.trace()), 0.0, TOL, "Tr(sx)=0");
    ASSERT_NEAR(std::abs(sy.trace()), 0.0, TOL, "Tr(sy)=0");
    ASSERT_NEAR(std::abs(sz.trace()), 0.0, TOL, "Tr(sz)=0");

    // Commutation: [sx, sy] = i*sz
    MatXc comm_xy = sx * sy - sy * sx;
    MatXc expected_comm = cplx(0, 1) * sz;
    ASSERT_NEAR((comm_xy - expected_comm).norm(), 0.0, TOL, "[sx,sy] = i*sz");
    g_detail << "Commutation relation [sx,sy] = i*sz verified.\n\n";
}

static void test_extend_operator_1particle()
{
    // For 1 particle, extend_operator should return the operator itself
    Mat2c sx, sy, sz;
    get_pauli_matrices(sx, sy, sz);

    MatXc ext = extend_operator(sx, 1, 0);

    g_detail << "=== extend_operator (1 particle) ===\n";
    g_detail << "Input: sigma_x/2, n_particles=1, position=0\n";
    g_detail << mat_to_string(ext, "Output");

    ASSERT_TRUE(ext.rows() == 2 && ext.cols() == 2, "dim 2x2");
    ASSERT_NEAR((ext - sx.cast<cplx>()).norm(), 0.0, TOL, "extend 1-particle = identity op");
}

static void test_extend_operator_2particle()
{
    // For 2 particles, sx at position 0 should be sx ⊗ I
    // and sx at position 1 should be I ⊗ sx
    Mat2c sx, sy, sz;
    get_pauli_matrices(sx, sy, sz);

    MatXc ext0 = extend_operator(sx, 2, 0);
    MatXc ext1 = extend_operator(sx, 2, 1);

    g_detail << "=== extend_operator (2 particles) ===\n";
    g_detail << mat_to_string(ext0, "sx at position 0 (sx ⊗ I)");
    g_detail << mat_to_string(ext1, "sx at position 1 (I ⊗ sx)");

    ASSERT_TRUE(ext0.rows() == 4 && ext0.cols() == 4, "dim 4x4 pos0");
    ASSERT_TRUE(ext1.rows() == 4 && ext1.cols() == 4, "dim 4x4 pos1");

    // Build expected: sx ⊗ I
    MatXc expected0 = Eigen::kroneckerProduct(MatXc(sx), MatXc(Mat2c::Identity())).eval();
    ASSERT_NEAR((ext0 - expected0).norm(), 0.0, TOL, "sx ⊗ I");

    // Build expected: I ⊗ sx
    MatXc expected1 = Eigen::kroneckerProduct(MatXc(Mat2c::Identity()), MatXc(sx)).eval();
    ASSERT_NEAR((ext1 - expected1).norm(), 0.0, TOL, "I ⊗ sx");
}

static void test_extend_operator_3particle()
{
    // For 3 particles, test operator at middle position (position 1)
    // Expected: I ⊗ sz ⊗ I
    Mat2c sx, sy, sz;
    get_pauli_matrices(sx, sy, sz);

    MatXc ext = extend_operator(sz, 3, 1);

    g_detail << "=== extend_operator (3 particles, position 1) ===\n";
    g_detail << "Input: sigma_z/2, n_particles=3, position=1\n";
    g_detail << "Output dimension: " << ext.rows() << "x" << ext.cols() << "\n";

    ASSERT_TRUE(ext.rows() == 8 && ext.cols() == 8, "dim 8x8");

    MatXc id2 = MatXc::Identity(2, 2);
    MatXc step1 = Eigen::kroneckerProduct(id2, MatXc(sz)).eval();
    MatXc expected = Eigen::kroneckerProduct(step1, id2).eval();
    ASSERT_NEAR((ext - expected).norm(), 0.0, TOL, "I ⊗ sz ⊗ I");
    g_detail << "Verified: result matches I ⊗ sz ⊗ I\n\n";
}

static void test_get_spin_operators()
{
    SystemConfig config(2, 1);  // 2 electrons, 1 nucleus
    SpinOperators ops = get_spin_operators(config);

    g_detail << "=== get_spin_operators (2 electrons) ===\n";
    g_detail << "electron_dim = " << ops.dim << "\n";
    g_detail << "Number of operator sets: " << ops.sx.size() << "\n";
    for (int i = 0; i < (int)ops.sx.size(); i++)
    {
        g_detail << mat_to_string(ops.sx[i], "sx[" + std::to_string(i) + "]");
        g_detail << mat_to_string(ops.sy[i], "sy[" + std::to_string(i) + "]");
        g_detail << mat_to_string(ops.sz[i], "sz[" + std::to_string(i) + "]");
    }

    // Should produce 2 sets of operators, each 4x4
    ASSERT_TRUE(ops.dim == 4, "electron_dim = 4");
    ASSERT_TRUE(ops.sx.size() == 2, "2 sx operators");
    ASSERT_TRUE(ops.sy.size() == 2, "2 sy operators");
    ASSERT_TRUE(ops.sz.size() == 2, "2 sz operators");

    for (int i = 0; i < 2; i++)
    {
        ASSERT_TRUE(ops.sx[i].rows() == 4 && ops.sx[i].cols() == 4,
                     "sx[" + std::to_string(i) + "] is 4x4");
        // Hermiticity
        ASSERT_NEAR((ops.sx[i] - ops.sx[i].adjoint()).norm(), 0.0, TOL,
                     "sx[" + std::to_string(i) + "] Hermitian");
        ASSERT_NEAR((ops.sy[i] - ops.sy[i].adjoint()).norm(), 0.0, TOL,
                     "sy[" + std::to_string(i) + "] Hermitian");
        ASSERT_NEAR((ops.sz[i] - ops.sz[i].adjoint()).norm(), 0.0, TOL,
                     "sz[" + std::to_string(i) + "] Hermitian");
    }

    // Operators for different electrons should commute: [S1_x, S2_x] = 0
    MatXc comm = ops.sx[0] * ops.sx[1] - ops.sx[1] * ops.sx[0];
    ASSERT_NEAR(comm.norm(), 0.0, TOL, "[sx1, sx2] = 0");
    g_detail << "Cross-electron commutation [sx1, sx2] = 0 verified.\n\n";
}

static void test_get_lifted_operators()
{
    SystemConfig config(2, 1);  // 2 electrons, 1 nucleus
    SpinOperators spin_ops = get_spin_operators(config);
    KronOperators krons = get_lifted_operators(spin_ops, config);

    g_detail << "=== get_lifted_operators (2e, 1n) ===\n";
    g_detail << "total_dim = " << krons.dim << "\n";
    g_detail << "Sx.size() = " << krons.Sx.size() << "\n";
    g_detail << mat_to_string(krons.Sx[0], "Sx[0] (first 8 rows/cols shown)");
    g_detail << mat_to_string(krons.Ix, "Ix");
    g_detail << mat_to_string(krons.Iy, "Iy");
    g_detail << mat_to_string(krons.Iz, "Iz");

    int dim = config.total_dim;  // 4*2 = 8
    ASSERT_TRUE(krons.dim == dim, "lifted dim = total_dim");
    ASSERT_TRUE((int)krons.Sx.size() == config.n_electrons, "Sx count");

    // All lifted operators should be (total_dim x total_dim)
    for (int i = 0; i < config.n_electrons; i++)
    {
        ASSERT_TRUE(krons.Sx[i].rows() == dim && krons.Sx[i].cols() == dim,
                     "Sx[" + std::to_string(i) + "] dim");
        ASSERT_NEAR((krons.Sx[i] - krons.Sx[i].adjoint()).norm(), 0.0, TOL,
                     "Sx[" + std::to_string(i) + "] Hermitian");
    }
    ASSERT_TRUE(krons.Ix.rows() == dim && krons.Ix.cols() == dim, "Ix dim");
    ASSERT_NEAR((krons.Ix - krons.Ix.adjoint()).norm(), 0.0, TOL, "Ix Hermitian");
    ASSERT_NEAR((krons.Iy - krons.Iy.adjoint()).norm(), 0.0, TOL, "Iy Hermitian");
    ASSERT_NEAR((krons.Iz - krons.Iz.adjoint()).norm(), 0.0, TOL, "Iz Hermitian");

    // Verify Sx[0] = sx[0] ⊗ I_nuclear
    MatXc id_nuc = MatXc::Identity(config.nuclear_dim, config.nuclear_dim);
    MatXc expected_Sx0 = Eigen::kroneckerProduct(spin_ops.sx[0], id_nuc).eval();
    ASSERT_NEAR((krons.Sx[0] - expected_Sx0).norm(), 0.0, TOL, "Sx[0] = sx[0] ⊗ I_n");

    // Verify Ix = I_electron ⊗ pauli_x/2
    Mat2c pauli_x, pauli_y, pauli_z;
    get_pauli_matrices(pauli_x, pauli_y, pauli_z);
    MatXc id_elec = MatXc::Identity(config.electron_dim, config.electron_dim);
    MatXc expected_Ix = Eigen::kroneckerProduct(id_elec, MatXc(pauli_x)).eval();
    ASSERT_NEAR((krons.Ix - expected_Ix).norm(), 0.0, TOL, "Ix = I_e ⊗ sx");
    g_detail << "Kronecker structure verified.\n\n";
}

static void test_get_proj_operators()
{
    SystemConfig config(2, 1);
    ProjOperators proj = get_proj_operators(config);
    int dim = config.total_dim;

    g_detail << "=== get_proj_operators (2e, 1n) ===\n";
    g_detail << "total_dim = " << dim << "\n";
    g_detail << mat_to_string(proj.Ps, "Ps (singlet projector)");
    g_detail << mat_to_string(proj.Pt, "Pt (triplet projector)");

    ASSERT_TRUE(proj.Ps.rows() == dim && proj.Ps.cols() == dim, "Ps dim");
    ASSERT_TRUE(proj.Pt.rows() == dim && proj.Pt.cols() == dim, "Pt dim");

    // Ps + Pt = I
    MatXc sum = proj.Ps + proj.Pt;
    MatXc I = MatXc::Identity(dim, dim);
    ASSERT_NEAR((sum - I).norm(), 0.0, TOL, "Ps + Pt = I");

    // Ps^2 = Ps (projector idempotent)
    ASSERT_NEAR((proj.Ps * proj.Ps - proj.Ps).norm(), 0.0, TOL, "Ps^2 = Ps");

    // Pt^2 = Pt
    ASSERT_NEAR((proj.Pt * proj.Pt - proj.Pt).norm(), 0.0, TOL, "Pt^2 = Pt");

    // Ps * Pt = 0 (orthogonal)
    ASSERT_NEAR((proj.Ps * proj.Pt).norm(), 0.0, TOL, "Ps * Pt = 0");

    // Hermiticity
    ASSERT_NEAR((proj.Ps - proj.Ps.adjoint()).norm(), 0.0, TOL, "Ps Hermitian");
    ASSERT_NEAR((proj.Pt - proj.Pt.adjoint()).norm(), 0.0, TOL, "Pt Hermitian");

    // Trace checks: Tr(Ps) = nuclear_dim, Tr(Pt) = total_dim - nuclear_dim
    ASSERT_NEAR(proj.Ps.trace().real(), (double)config.nuclear_dim, TOL, "Tr(Ps)");
    ASSERT_NEAR(proj.Pt.trace().real(), (double)(dim - config.nuclear_dim), TOL, "Tr(Pt)");
    g_detail << "Projector properties verified (idempotent, orthogonal, complete).\n\n";
}

// ===================================================================
//  HAMILTONIAN TESTS  (contrived operators — no physics pipeline)
// ===================================================================

static void test_hamiltonian_formula()
{
    // Verify construct_hamiltonian implements the correct formula:
    //   H = g*mu*Bz * sum(Sz[i]) + g*mu*a * (Ix*Sx[0] + Iy*Sy[0] + Iz*Sz[0])
    // by providing hand-crafted operators and independently computing expected H.

    SystemConfig config(1, 1);  // 1 electron, 1 nucleus → total_dim = 4
    int dim = config.total_dim;

    PhysicalParams params;
    params.g = 2.5;
    params.mu = 0.1;
    params.a = 3.0;
    double Bz = 4.0;

    // Build non-trivial 4x4 operators with known values
    KronOperators krons;
    krons.dim = dim;

    MatXc Sz0 = MatXc::Zero(dim, dim);
    Sz0(0,0) = 1.0;  Sz0(1,1) = -1.0;  Sz0(2,2) = 0.5;  Sz0(3,3) = -0.5;
    Sz0(0,1) = cplx(0.1, 0.2);  Sz0(1,0) = cplx(0.1, -0.2);

    MatXc Sx0 = MatXc::Zero(dim, dim);
    Sx0(0,0) = 0.3;  Sx0(1,1) = -0.3;  Sx0(2,2) = 0.7;  Sx0(3,3) = -0.7;
    Sx0(0,2) = cplx(0.5, 0);  Sx0(2,0) = cplx(0.5, 0);

    MatXc Sy0 = MatXc::Zero(dim, dim);
    Sy0(0,0) = 0.2;  Sy0(1,1) = -0.2;
    Sy0(0,3) = cplx(0, -0.4);  Sy0(3,0) = cplx(0, 0.4);

    MatXc Ix = MatXc::Zero(dim, dim);
    Ix(0,0) = 0.5;  Ix(1,1) = 0.5;  Ix(2,2) = -0.5;  Ix(3,3) = -0.5;
    Ix(1,2) = cplx(0.3, 0.1);  Ix(2,1) = cplx(0.3, -0.1);

    MatXc Iy = MatXc::Zero(dim, dim);
    Iy(0,1) = cplx(0, -0.5);  Iy(1,0) = cplx(0, 0.5);
    Iy(2,3) = cplx(0, -0.5);  Iy(3,2) = cplx(0, 0.5);

    MatXc Iz = MatXc::Zero(dim, dim);
    Iz(0,0) = 0.25;  Iz(1,1) = -0.25;  Iz(2,2) = 0.25;  Iz(3,3) = -0.25;

    krons.Sx = {Sx0};
    krons.Sy = {Sy0};
    krons.Sz = {Sz0};
    krons.Ix = Ix;
    krons.Iy = Iy;
    krons.Iz = Iz;

    MatXc H = construct_hamiltonian(krons, config, params, Bz);

    // Independently compute expected H using the formula
    MatXc expected = (params.g * params.mu * Bz) * Sz0
                   + (params.g * params.mu * params.a) * (Ix * Sx0 + Iy * Sy0 + Iz * Sz0);

    g_detail << "=== construct_hamiltonian: formula verification ===\n";
    g_detail << "Params: g=" << params.g << ", mu=" << params.mu
             << ", a=" << params.a << ", Bz=" << Bz << "\n";
    g_detail << mat_to_string(H, "H (computed)");
    g_detail << mat_to_string(expected, "H (expected)");
    g_detail << "Difference norm: " << (H - expected).norm() << "\n\n";

    ASSERT_NEAR((H - expected).norm(), 0.0, TOL, "H matches formula with contrived operators");
}

static void test_hamiltonian_multi_electron_zeeman()
{
    // For n_electrons=2, the Zeeman term should sum Sz[0]+Sz[1].
    // The hyperfine term should use only electron 0's operators.
    // We verify by providing two distinct Sz operators and checking both appear
    // in the Zeeman sum, while Sx[1], Sy[1] do NOT appear in the hyperfine term.

    SystemConfig config(2, 1);  // total_dim = 8
    int dim = config.total_dim;

    PhysicalParams params;
    params.g = 1.0;  params.mu = 1.0;  params.a = 1.0;
    double Bz = 2.0;

    KronOperators krons;
    krons.dim = dim;

    // Create distinct diagonal operators for each electron
    MatXc Sz0 = MatXc::Zero(dim, dim);
    MatXc Sz1 = MatXc::Zero(dim, dim);
    MatXc Sx0 = MatXc::Zero(dim, dim);
    MatXc Sy0 = MatXc::Zero(dim, dim);
    MatXc Sx1 = MatXc::Zero(dim, dim);  // should NOT appear in hyperfine
    MatXc Sy1 = MatXc::Zero(dim, dim);  // should NOT appear in hyperfine

    for (int i = 0; i < dim; i++)
    {
        Sz0(i,i) = cplx(0.1 * (i + 1), 0);
        Sz1(i,i) = cplx(0.2 * (dim - i), 0);
        Sx0(i,i) = cplx(0.05 * (i + 2), 0);
        Sy0(i,i) = cplx(0.03 * (i + 1), 0);
        // Give Sx1, Sy1 large values — they must NOT affect H
        Sx1(i,i) = cplx(999.0, 0);
        Sy1(i,i) = cplx(888.0, 0);
    }

    MatXc Ix = MatXc::Identity(dim, dim) * 0.4;
    MatXc Iy = MatXc::Identity(dim, dim) * 0.6;
    MatXc Iz = MatXc::Identity(dim, dim) * 0.8;

    krons.Sx = {Sx0, Sx1};
    krons.Sy = {Sy0, Sy1};
    krons.Sz = {Sz0, Sz1};
    krons.Ix = Ix;  krons.Iy = Iy;  krons.Iz = Iz;

    MatXc H = construct_hamiltonian(krons, config, params, Bz);

    // Expected: Zeeman sums BOTH Sz, hyperfine uses ONLY electron 0
    MatXc expected = params.g * params.mu * Bz * (Sz0 + Sz1)
                   + params.g * params.mu * params.a * (Ix * Sx0 + Iy * Sy0 + Iz * Sz0);

    g_detail << "=== construct_hamiltonian: 2-electron Zeeman sum ===\n";
    g_detail << "Verified Zeeman sums Sz[0]+Sz[1], hyperfine uses only electron 0\n";
    g_detail << "Sx[1], Sy[1] filled with large sentinel values (999, 888) — must not affect H\n";
    g_detail << "Difference norm: " << (H - expected).norm() << "\n\n";

    ASSERT_NEAR((H - expected).norm(), 0.0, TOL, "2-electron H: Zeeman sums correctly, hyperfine uses electron 0 only");
}

static void test_hamiltonian_zero_field()
{
    // At Bz=0, the Zeeman term should vanish; only hyperfine survives.
    // Use deterministic non-trivial operators to verify.

    SystemConfig config(1, 1);
    int dim = config.total_dim;

    PhysicalParams params;
    params.g = 2.0;  params.mu = 0.5;  params.a = 3.0;

    KronOperators krons;
    krons.dim = dim;

    // Deterministic non-trivial operators
    MatXc Sz0(dim, dim), Sx0(dim, dim), Sy0(dim, dim);
    MatXc Ix(dim, dim), Iy(dim, dim), Iz(dim, dim);
    for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
    {
        Sz0(i,j) = cplx(0.1 * (i+1) * (j+1), 0.05 * (i-j));
        Sx0(i,j) = cplx(0.2 * (i+1) - 0.1*j, 0.03 * (i+j));
        Sy0(i,j) = cplx(-0.1 * (i+2), 0.04 * j);
        Ix(i,j)  = cplx(0.15 * (j+1), 0.02 * (i-j));
        Iy(i,j)  = cplx(-0.05 * (i+j+1), 0.01 * (i+1));
        Iz(i,j)  = cplx(0.08 * (i+1) * (j+2), -0.03 * j);
    }

    krons.Sx = {Sx0};  krons.Sy = {Sy0};  krons.Sz = {Sz0};
    krons.Ix = Ix;  krons.Iy = Iy;  krons.Iz = Iz;

    MatXc H = construct_hamiltonian(krons, config, params, 0.0);

    // At Bz=0, only hyperfine term should survive
    MatXc expected = (params.g * params.mu * params.a) * (Ix * Sx0 + Iy * Sy0 + Iz * Sz0);

    g_detail << "=== construct_hamiltonian: Bz=0 (Zeeman vanishes) ===\n";
    g_detail << "Difference norm: " << (H - expected).norm() << "\n\n";

    ASSERT_NEAR((H - expected).norm(), 0.0, TOL, "H(Bz=0) = hyperfine only");
}

static void test_hamiltonian_linearity_in_Bz()
{
    // H(alpha*Bz) - H(0) should scale linearly in Bz because only the
    // Zeeman term depends on Bz.  Test with contrived operators.

    SystemConfig config(1, 1);
    int dim = config.total_dim;

    PhysicalParams params;
    params.g = 2.0;  params.mu = 0.5;  params.a = 3.0;

    KronOperators krons;
    krons.dim = dim;

    // Deterministic non-trivial operators
    MatXc Sz0(dim, dim), Sx0(dim, dim), Sy0(dim, dim);
    MatXc Ix(dim, dim), Iy(dim, dim), Iz(dim, dim);
    for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
    {
        double x = 0.1 * (i+1) * (j+2);
        Sz0(i,j) = cplx(x, 0.05 * i);
        Sx0(i,j) = cplx(0.3 * x, -0.02 * j);
        Sy0(i,j) = cplx(-0.1 * x, 0.04 * (i-j));
        Ix(i,j)  = cplx(0.2 * x, 0.01);
        Iy(i,j)  = cplx(-0.15 * x, 0.03);
        Iz(i,j)  = cplx(0.25 * x, -0.02);
    }

    krons.Sx = {Sx0};  krons.Sy = {Sy0};  krons.Sz = {Sz0};
    krons.Ix = Ix;  krons.Iy = Iy;  krons.Iz = Iz;

    MatXc H0 = construct_hamiltonian(krons, config, params, 0.0);
    MatXc H1 = construct_hamiltonian(krons, config, params, 1.0);
    MatXc H2 = construct_hamiltonian(krons, config, params, 2.0);

    // H(2) - H(0) should be 2 * (H(1) - H(0))
    MatXc diff1 = H1 - H0;
    MatXc diff2 = H2 - H0;

    g_detail << "=== construct_hamiltonian: linearity in Bz ===\n";
    g_detail << "||H(2)-H(0) - 2*(H(1)-H(0))|| = " << (diff2 - 2.0 * diff1).norm() << "\n\n";

    ASSERT_NEAR((diff2 - 2.0 * diff1).norm(), 0.0, TOL, "Zeeman linear in Bz");
}

// ===================================================================
//  SLE SOLVER TESTS  (contrived matrices — no physics pipeline)
// ===================================================================

static void test_construct_sle_linear_system()
{
    // Verify the vectorization formula with a contrived 2x2 system.
    // construct_sle_linear_system should produce:
    //   M = -(i/hbar)(I⊗H - H^T⊗I) - 0.5*(Ks+Kd)*(I⊗Ps+Ps^T⊗I) - 0.5*Kd*(I⊗Pt+Pt^T⊗I)
    //   b = -vec(I/dim)

    int dim = 2;
    int N = dim * dim;  // 4

    // Contrived Hermitian Hamiltonian
    MatXc H(dim, dim);
    H(0,0) = cplx(1.0, 0);      H(0,1) = cplx(0.5, 0.3);
    H(1,0) = cplx(0.5, -0.3);   H(1,1) = cplx(-1.0, 0);

    // Contrived projectors with Ps + Pt = I
    MatXc Ps = MatXc::Zero(dim, dim);
    Ps(0,0) = 1.0;
    MatXc Pt = MatXc::Identity(dim, dim) - Ps;

    PhysicalParams params;
    params.hbar = 2.0;  // non-default to verify coefficient
    params.Ks = 4.0;
    params.Kd = 1.5;

    auto [M, b] = construct_sle_linear_system(H, Ps, Pt, params, dim);

    // Independently compute expected M and b using the formula
    MatXc I = MatXc::Identity(dim, dim);

    MatXc comm = Eigen::kroneckerProduct(I, H).eval()
               - Eigen::kroneckerProduct(H.transpose(), I).eval();

    MatXc anticomm_s = Eigen::kroneckerProduct(I, Ps).eval()
                     + Eigen::kroneckerProduct(Ps.transpose(), I).eval();

    MatXc anticomm_t = Eigen::kroneckerProduct(I, Pt).eval()
                     + Eigen::kroneckerProduct(Pt.transpose(), I).eval();

    MatXc expected_M = -cplx(0, 1.0 / params.hbar) * comm
                     - 0.5 * (params.Ks + params.Kd) * anticomm_s
                     - 0.5 * params.Kd * anticomm_t;

    MatXc rhs = (1.0 / dim) * I;
    Eigen::Map<const VecXc> rhs_vec(rhs.data(), N);
    VecXc expected_b = -VecXc(rhs_vec);

    g_detail << "=== construct_sle_linear_system (contrived 2x2) ===\n";
    g_detail << "Params: hbar=" << params.hbar << ", Ks=" << params.Ks << ", Kd=" << params.Kd << "\n";
    g_detail << mat_to_string(H, "H (input)");
    g_detail << "||M - expected_M|| = " << (M - expected_M).norm() << "\n";
    g_detail << "||b - expected_b|| = " << (b - expected_b).norm() << "\n";
    g_detail << "M dimensions: " << M.rows() << " x " << M.cols() << "\n";
    g_detail << "b dimensions: " << b.size() << "\n\n";

    ASSERT_TRUE(M.rows() == N && M.cols() == N, "M is dim^2 x dim^2");
    ASSERT_TRUE(b.size() == N, "b has dim^2 entries");
    ASSERT_NEAR((M - expected_M).norm(), 0.0, TOL, "M matches vectorization formula");
    ASSERT_NEAR((b - expected_b).norm(), 0.0, TOL, "b matches -vec(I/dim)");
}

static void test_solve_steady_state_cpu()
{
    // Test the linear solver with a contrived system that has a known solution.
    // Construct a known Hermitian, unit-trace density matrix rho_true,
    // an invertible M, and b = M * vec(rho_true).
    // Since rho_true is already normalized and Hermitian, the post-processing
    // steps in the solver should be near-identity, and we recover rho_true.

    int dim = 3;
    int N = dim * dim;  // 9

    // Known density matrix: Hermitian, trace = 1, positive semi-definite
    MatXc rho_true = MatXc::Zero(dim, dim);
    rho_true(0,0) = cplx(0.5, 0);
    rho_true(0,1) = cplx(0.1, 0.05);
    rho_true(0,2) = cplx(0.02, 0);
    rho_true(1,0) = cplx(0.1, -0.05);
    rho_true(1,1) = cplx(0.3, 0);
    rho_true(1,2) = cplx(0.02, 0.03);
    rho_true(2,0) = cplx(0.02, 0);
    rho_true(2,1) = cplx(0.02, -0.03);
    rho_true(2,2) = cplx(0.2, 0);

    // Vectorize rho_true (column-major, Eigen default)
    VecXc x_true = Eigen::Map<VecXc>(rho_true.data(), N);

    // Diagonally-dominant invertible M (well-conditioned)
    MatXc M = MatXc::Zero(N, N);
    for (int i = 0; i < N; i++)
    {
        M(i,i) = cplx(10.0 + i, 0.5 * ((i % 3) - 1));
        for (int j = 0; j < N; j++)
        {
            if (i != j)
                M(i,j) = cplx(0.1 * (((i+j) % 5) - 2), 0.05 * (((i*j) % 3) - 1));
        }
    }

    VecXc b = M * x_true;

    double solve_time = 0.0;
    MatXc rho = solve_steady_state_cpu(M, b, dim, &solve_time);

    g_detail << "=== solve_steady_state_cpu (contrived 3x3 system) ===\n";
    g_detail << "dim=" << dim << ", M is " << N << "x" << N << "\n";
    g_detail << "Solve time: " << solve_time << " s\n";
    g_detail << mat_to_string(rho_true, "rho_true");
    g_detail << mat_to_string(rho, "rho_solved");
    g_detail << "||rho - rho_true|| = " << (rho - rho_true).norm() << "\n\n";

    ASSERT_NEAR((rho - rho_true).norm(), 0.0, 1e-8, "solver recovers known density matrix");
    ASSERT_TRUE(rho.rows() == dim && rho.cols() == dim, "rho has correct dimensions");
    ASSERT_TRUE(solve_time >= 0.0, "solve time is non-negative");
}

static void test_solve_normalization_hermiticity()
{
    // Test that the solver correctly normalizes trace to 1 and enforces
    // Hermiticity when the raw solution is neither normalized nor Hermitian.

    int dim = 2;
    int N = dim * dim;

    // Raw matrix: non-Hermitian (rho(0,1) != conj(rho(1,0))) and trace > 1
    MatXc rho_raw = MatXc::Zero(dim, dim);
    rho_raw(0,0) = cplx(3.0, 0.01);
    rho_raw(0,1) = cplx(0.5, 0.3);
    rho_raw(1,0) = cplx(0.4, -0.1);   // NOT conj of (0,1)
    rho_raw(1,1) = cplx(2.0, -0.01);

    VecXc x_raw = Eigen::Map<VecXc>(rho_raw.data(), N);

    // Diagonally-dominant invertible M
    MatXc M = MatXc::Zero(N, N);
    M(0,0) = cplx(10, 1);    M(0,1) = cplx(0.2, 0);
    M(0,2) = cplx(0.05, 0.02);  M(0,3) = cplx(0.1, 0);
    M(1,0) = cplx(0.1, 0);   M(1,1) = cplx(8, -1);
    M(1,2) = cplx(0.2, 0);   M(1,3) = cplx(0.03, -0.01);
    M(2,0) = cplx(0.04, 0.01);  M(2,1) = cplx(0.15, 0);
    M(2,2) = cplx(9, 0.5);   M(2,3) = cplx(0.3, 0);
    M(3,0) = cplx(0.1, 0);   M(3,1) = cplx(0.02, 0.03);
    M(3,2) = cplx(0.1, 0);   M(3,3) = cplx(7, -0.5);

    VecXc b = M * x_raw;

    MatXc rho = solve_steady_state_cpu(M, b, dim);

    // The function first normalizes: rho_norm = rho_raw / trace(rho_raw)
    // then enforces Hermiticity: rho_out = (rho_norm + rho_norm†) / 2
    cplx tr = rho_raw.trace();
    MatXc normalized = rho_raw / tr;
    MatXc expected = 0.5 * (normalized + normalized.adjoint());

    g_detail << "=== solve: normalization & Hermiticity enforcement ===\n";
    g_detail << "Raw trace: " << tr << "\n";
    g_detail << mat_to_string(rho_raw, "rho_raw (non-Hermitian, trace>1)");
    g_detail << mat_to_string(expected, "expected (after normalization + Hermiticity)");
    g_detail << mat_to_string(rho, "rho_solved");
    g_detail << "||rho - expected|| = " << (rho - expected).norm() << "\n\n";

    ASSERT_NEAR((rho - expected).norm(), 0.0, 1e-8, "post-processing matches expected");
    ASSERT_NEAR(rho.trace().real(), 1.0, 1e-8, "trace = 1 after normalization");
    ASSERT_NEAR((rho - rho.adjoint()).norm(), 0.0, 1e-10, "Hermitian after enforcement");
}

static void test_compute_singlet_population()
{
    // Test Tr(Ps*rho) with fully contrived matrices — no physics pipeline.

    g_detail << "=== compute_singlet_population (contrived) ===\n";

    // Test 1: Arbitrary 3x3 matrices
    {
        int dim = 3;
        MatXc Ps(dim, dim);
        Ps << cplx(1,0),    cplx(0.2,0.1), cplx(0,0),
              cplx(0.2,-0.1), cplx(0.5,0),  cplx(0,0),
              cplx(0,0),      cplx(0,0),     cplx(0,0);

        MatXc rho(dim, dim);
        rho << cplx(0.5,0),     cplx(0.1,0.2),    cplx(0,0),
               cplx(0.1,-0.2),  cplx(0.3,0),       cplx(0.05,0.1),
               cplx(0,0),       cplx(0.05,-0.1),    cplx(0.2,0);

        double result = compute_singlet_population(Ps, rho);
        double expected = (Ps * rho).trace().real();

        g_detail << "Test 1: Arbitrary 3x3 → result=" << result
                 << ", expected=" << expected << "\n";
        ASSERT_NEAR(result, expected, TOL, "Tr(Ps*rho) with arbitrary 3x3");
    }

    // Test 2: Ps = Identity → result = Tr(rho) = 1 for normalized rho
    {
        int dim = 4;
        MatXc Ps = MatXc::Identity(dim, dim);
        MatXc rho = MatXc::Identity(dim, dim) / (double)dim;  // trace = 1

        double result = compute_singlet_population(Ps, rho);
        g_detail << "Test 2: Ps=I, rho=I/4 → result=" << result << ", expected=1\n";
        ASSERT_NEAR(result, 1.0, TOL, "Tr(I * rho) = Tr(rho) = 1");
    }

    // Test 3: Ps = zero matrix → result = 0
    {
        int dim = 3;
        MatXc Ps = MatXc::Zero(dim, dim);
        MatXc rho = MatXc::Identity(dim, dim) / (double)dim;

        double result = compute_singlet_population(Ps, rho);
        g_detail << "Test 3: Ps=0 → result=" << result << ", expected=0\n";
        ASSERT_NEAR(result, 0.0, TOL, "Tr(0 * rho) = 0");
    }

    // Test 4: Pure state projector |0><0| with matching state → 1
    {
        int dim = 3;
        MatXc Ps = MatXc::Zero(dim, dim);
        Ps(0,0) = 1.0;
        MatXc rho = MatXc::Zero(dim, dim);
        rho(0,0) = 1.0;

        double result = compute_singlet_population(Ps, rho);
        g_detail << "Test 4: Ps=|0><0|, rho=|0><0| → result=" << result << "\n";
        ASSERT_NEAR(result, 1.0, TOL, "matching projector and state → 1");
    }

    // Test 5: Orthogonal projector and state → 0
    {
        int dim = 3;
        MatXc Ps = MatXc::Zero(dim, dim);
        Ps(0,0) = 1.0;
        MatXc rho = MatXc::Zero(dim, dim);
        rho(1,1) = 1.0;

        double result = compute_singlet_population(Ps, rho);
        g_detail << "Test 5: Ps=|0><0|, rho=|1><1| → result=" << result << "\n";
        ASSERT_NEAR(result, 0.0, TOL, "orthogonal projector and state → 0");
    }

    g_detail << "\n";
}

static void test_validate_density_matrix_valid()
{
    // Construct a known valid density matrix (contrived, no physics)
    int dim = 4;
    MatXc rho = MatXc::Identity(dim, dim) / (double)dim;

    g_detail << "=== validate_density_matrix (valid input) ===\n";
    g_detail << mat_to_string(rho, "rho = I/4");

    DensityMatrixValidation val = validate_density_matrix(rho, 1e-6);

    g_detail << "is_hermitian: " << val.is_hermitian << "\n";
    g_detail << "is_normalized: " << val.is_normalized << "\n";
    g_detail << "is_positive_semidefinite: " << val.is_positive_semidefinite << "\n";
    g_detail << "trace: " << val.trace_value << "\n";
    g_detail << "hermiticity_error: " << val.hermiticity_error << "\n";

    ASSERT_TRUE(val.is_hermitian, "valid rho Hermitian");
    ASSERT_TRUE(val.is_normalized, "valid rho normalized");
    ASSERT_TRUE(val.is_positive_semidefinite, "valid rho pos-semidef");
    ASSERT_NEAR(val.trace_value, 1.0, 1e-6, "valid rho trace");
    g_detail << "\n";
}

static void test_validate_density_matrix_invalid()
{
    g_detail << "=== validate_density_matrix (invalid inputs) ===\n";

    int dim = 4;

    // Not normalized: Tr != 1
    {
        MatXc rho = MatXc::Identity(dim, dim);  // Tr = 4
        DensityMatrixValidation val = validate_density_matrix(rho, 1e-6);
        g_detail << "Test: rho = I (Tr=4): is_normalized=" << val.is_normalized
                 << ", trace=" << val.trace_value << "\n";
        ASSERT_TRUE(!val.is_normalized, "unnormalized detected");
    }

    // Not Hermitian
    {
        MatXc rho = MatXc::Zero(dim, dim);
        rho(0, 0) = 1.0;
        rho(0, 1) = cplx(0, 1.0);
        rho(1, 0) = cplx(0, 1.0);  // same, not conjugate => not Hermitian
        DensityMatrixValidation val = validate_density_matrix(rho, 1e-6);
        g_detail << "Test: non-Hermitian rho: is_hermitian=" << val.is_hermitian
                 << ", hermiticity_error=" << val.hermiticity_error << "\n";
        ASSERT_TRUE(!val.is_hermitian, "non-Hermitian detected");
    }

    // Not positive semi-definite
    {
        MatXc rho = MatXc::Zero(dim, dim);
        rho(0, 0) = 2.0;
        rho(1, 1) = -1.0;  // negative eigenvalue
        // Tr = 1 but has negative eigenvalue
        DensityMatrixValidation val = validate_density_matrix(rho, 1e-6);
        g_detail << "Test: negative eigenvalue: is_positive_semidefinite="
                 << val.is_positive_semidefinite << "\n";
        ASSERT_TRUE(!val.is_positive_semidefinite, "negative eigenvalue detected");
    }
    g_detail << "\n";
}

static void test_run_sweep()
{
    // Test the sweep logic: correct number of points, correct Bz values,
    // and finite outputs.  Uses a minimal config to keep it fast.
    SystemConfig config(2, 1);
    PhysicalParams params;

    g_detail << "=== run_sweep (sweep logic) ===\n";

    // Test 1: 5-point sweep
    {
        SweepParams sweep;
        sweep.Bz_min = -2.0;
        sweep.Bz_max = 2.0;
        sweep.Bz_step = 1.0;

        std::vector<SimulationPoint> results = run_sweep(config, params, sweep, false);

        g_detail << "5-point sweep: got " << results.size() << " points\n";
        for (const auto& pt : results)
        {
            g_detail << "  Bz=" << std::setw(6) << pt.Bz
                     << ", pop=" << pt.singlet_population << "\n";
        }

        ASSERT_TRUE(results.size() == 5, "5-point sweep returns 5 points");

        // Verify Bz values
        std::vector<double> expected_Bz = {-2.0, -1.0, 0.0, 1.0, 2.0};
        for (size_t i = 0; i < results.size(); i++)
        {
            ASSERT_NEAR(results[i].Bz, expected_Bz[i], 1e-10,
                        "Bz[" + std::to_string(i) + "] value");
        }

        // All populations should be finite
        for (const auto& pt : results)
        {
            ASSERT_TRUE(std::isfinite(pt.singlet_population),
                        "finite population at Bz=" + std::to_string(pt.Bz));
        }
    }

    // Test 2: Single-point sweep (Bz_min == Bz_max)
    {
        SweepParams sweep;
        sweep.Bz_min = 1.0;
        sweep.Bz_max = 1.0;
        sweep.Bz_step = 0.5;

        std::vector<SimulationPoint> results = run_sweep(config, params, sweep, false);
        g_detail << "Single-point sweep: got " << results.size() << " points\n";

        ASSERT_TRUE(results.size() == 1, "single-point sweep returns 1 point");
        ASSERT_NEAR(results[0].Bz, 1.0, 1e-10, "single-point Bz value");
    }

    g_detail << "\n";
}

static void test_config_validation()
{
    g_detail << "=== SystemConfig validation ===\n";

    // Valid configs
    {
        SystemConfig c(1, 1);
        g_detail << "SystemConfig(1,1): electron_dim=" << c.electron_dim
                 << " nuclear_dim=" << c.nuclear_dim
                 << " total_dim=" << c.total_dim << "\n";
        ASSERT_TRUE(c.electron_dim == 2, "1e: electron_dim=2");
        ASSERT_TRUE(c.nuclear_dim == 2, "1n: nuclear_dim=2");
        ASSERT_TRUE(c.total_dim == 4, "1e1n: total_dim=4");
    }
    {
        SystemConfig c(2, 1);
        ASSERT_TRUE(c.electron_dim == 4, "2e: electron_dim=4");
        ASSERT_TRUE(c.total_dim == 8, "2e1n: total_dim=8");
        g_detail << "SystemConfig(2,1): total_dim=" << c.total_dim << "\n";
    }
    {
        SystemConfig c(3, 1);
        ASSERT_TRUE(c.electron_dim == 8, "3e: electron_dim=8");
        ASSERT_TRUE(c.total_dim == 16, "3e1n: total_dim=16");
        g_detail << "SystemConfig(3,1): total_dim=" << c.total_dim << "\n";
    }

    // Invalid configs should throw
    bool caught = false;
    try { SystemConfig c(0, 1); } catch (const std::invalid_argument&) { caught = true; }
    ASSERT_TRUE(caught, "n_electrons=0 rejected");
    g_detail << "SystemConfig(0,1): correctly rejected\n";

    caught = false;
    try { SystemConfig c(5, 1); } catch (const std::invalid_argument&) { caught = true; }
    ASSERT_TRUE(caught, "n_electrons=5 rejected");
    g_detail << "SystemConfig(5,1): correctly rejected\n";

    caught = false;
    try { SystemConfig c(2, 0); } catch (const std::invalid_argument&) { caught = true; }
    ASSERT_TRUE(caught, "n_nuclei=0 rejected");
    g_detail << "SystemConfig(2,0): correctly rejected\n\n";
}

// ===================================================================
//  MAIN
// ===================================================================

int main(int argc, char* argv[])
{
    bool write_report = false;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--report" || arg == "-r")
        {
            write_report = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "SLE Solver Test Suite\n\n";
            std::cout << "Usage: " << argv[0] << " [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --report, -r   Write detailed test report to output/\n";
            std::cout << "  --help, -h     Show this help message\n";
            return 0;
        }
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  SLE Solver Test Suite\n";
    std::cout << "========================================\n\n";

    // -- operators.cpp tests --
    run_test("get_pauli_matrices",                    test_pauli_matrices);
    run_test("extend_operator (1 particle)",          test_extend_operator_1particle);
    run_test("extend_operator (2 particles)",         test_extend_operator_2particle);
    run_test("extend_operator (3 particles)",         test_extend_operator_3particle);
    run_test("get_spin_operators",                    test_get_spin_operators);
    run_test("get_lifted_operators",                  test_get_lifted_operators);
    run_test("get_proj_operators",                    test_get_proj_operators);

    // -- hamiltonian.cpp tests (contrived operators) --
    run_test("construct_hamiltonian: formula",        test_hamiltonian_formula);
    run_test("construct_hamiltonian: 2e Zeeman sum",  test_hamiltonian_multi_electron_zeeman);
    run_test("construct_hamiltonian: Bz=0",           test_hamiltonian_zero_field);
    run_test("construct_hamiltonian: linearity in Bz",test_hamiltonian_linearity_in_Bz);

    // -- sle_solver.cpp tests (contrived matrices) --
    run_test("construct_sle_linear_system",           test_construct_sle_linear_system);
    run_test("solve_steady_state_cpu (known system)", test_solve_steady_state_cpu);
    run_test("solve: normalization & Hermiticity",    test_solve_normalization_hermiticity);
    run_test("compute_singlet_population",            test_compute_singlet_population);
    run_test("validate_density_matrix (valid)",       test_validate_density_matrix_valid);
    run_test("validate_density_matrix (invalid)",     test_validate_density_matrix_invalid);
    run_test("run_sweep (sweep logic)",               test_run_sweep);

    // -- config.hpp tests --
    run_test("SystemConfig validation",               test_config_validation);

    // -- Print results --
    int passed = 0, failed = 0;
    for (const auto& r : g_results)
    {
        if (r.passed)
        {
            std::cout << "  [PASS] " << r.name << "\n";
            passed++;
        }
        else
        {
            std::cout << "  [FAIL] " << r.name << "\n";
            std::cout << "         " << r.failure << "\n";
            failed++;
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "  Results: " << passed << " passed, " << failed << " failed, "
              << g_results.size() << " total\n";
    std::cout << "----------------------------------------\n";

    // -- Write detailed report --
    if (write_report)
    {
        // Generate timestamped filename
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream ts;
        ts << std::put_time(std::localtime(&time), "%Y%m%d_%H%M%S");

        std::string report_file = "output/test_report_" + ts.str() + ".txt";
        std::ofstream out(report_file);
        if (!out)
        {
            std::cerr << "Error: Cannot open report file '" << report_file << "'\n";
            return (failed > 0) ? 1 : 0;
        }

        out << "================================================================\n";
        out << "  SLE Solver Test Suite - Detailed Report\n";
        out << "  Generated: " << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "\n";
        out << "================================================================\n\n";

        out << "SUMMARY: " << passed << " passed, " << failed << " failed, "
            << g_results.size() << " total\n\n";

        for (const auto& r : g_results)
        {
            out << "------------------------------------------------------------\n";
            out << (r.passed ? "[PASS] " : "[FAIL] ") << r.name << "\n";
            out << "------------------------------------------------------------\n";
            if (!r.passed)
            {
                out << "FAILURE: " << r.failure << "\n";
            }
            if (!r.detail.empty())
            {
                out << "\n" << r.detail;
            }
            out << "\n";
        }

        out.close();
        std::cout << "\n  Report written to: " << report_file << "\n";
    }

    std::cout << "\n";
    return (failed > 0) ? 1 : 0;
}
