#include "operators.hpp"

void get_pauli_matrices(Mat2c& sigma_x, Mat2c& sigma_y, Mat2c& sigma_z)
{
    // Pauli matrices scaled by 1/2 for spin-1/2
    sigma_x << 0, 0.5,
               0.5, 0;

    sigma_y << 0, cplx(0, -0.5),
               cplx(0, 0.5), 0;

    sigma_z << 0.5, 0,
               0, -0.5;
}

MatXc extend_operator(const Mat2c& single_op, int n_particles, int position)
{
    // Build I ⊗ ... ⊗ I ⊗ op ⊗ I ⊗ ... ⊗ I
    // with op at the given position (0-indexed)
    
    MatXc result;
    
    if (position == 0)
    {
        result = single_op;
    }
    else
    {
        result = Mat2c::Identity();
    }

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

SpinOperators get_spin_operators(const SystemConfig& config)
{
    SpinOperators ops;
    ops.dim = config.electron_dim;

    Mat2c pauli_x, pauli_y, pauli_z;
    get_pauli_matrices(pauli_x, pauli_y, pauli_z);

    // Build operators for each electron
    for (int i = 0; i < config.n_electrons; i++)
    {
        ops.sx.push_back(extend_operator(pauli_x, config.n_electrons, i));
        ops.sy.push_back(extend_operator(pauli_y, config.n_electrons, i));
        ops.sz.push_back(extend_operator(pauli_z, config.n_electrons, i));
    }

    return ops;
}

KronOperators get_lifted_operators(const SpinOperators& spin_ops, const SystemConfig& config)
{
    KronOperators krons;
    krons.dim = config.total_dim;

    // Identity matrix for nuclear space
    MatXc id_nuclear = MatXc::Identity(config.nuclear_dim, config.nuclear_dim);

    // Lift electron operators: S_i = s_i ⊗ I_nuclear
    // From Mathematica: Sx1 = KroneckerProduct[sx1, IdentityMatrix[2]]
    for (int i = 0; i < config.n_electrons; i++)
    {
        krons.Sx.push_back(Eigen::kroneckerProduct(spin_ops.sx[i], id_nuclear).eval());
        krons.Sy.push_back(Eigen::kroneckerProduct(spin_ops.sy[i], id_nuclear).eval());
        krons.Sz.push_back(Eigen::kroneckerProduct(spin_ops.sz[i], id_nuclear).eval());
    }

    // Identity matrix for electron space
    MatXc id_electron = MatXc::Identity(config.electron_dim, config.electron_dim);

    // Nuclear spin operators: I_α = I_electron ⊗ σ_α/2
    // From Mathematica: Ix = KroneckerProduct[IdentityMatrix[4], 1/2 PauliMatrix[1]]
    Mat2c pauli_x, pauli_y, pauli_z;
    get_pauli_matrices(pauli_x, pauli_y, pauli_z);

    krons.Ix = Eigen::kroneckerProduct(id_electron, pauli_x).eval();
    krons.Iy = Eigen::kroneckerProduct(id_electron, pauli_y).eval();
    krons.Iz = Eigen::kroneckerProduct(id_electron, pauli_z).eval();

    return krons;
}

ProjOperators get_proj_operators(const SystemConfig& config)
{
    ProjOperators proj;
    
    int dim = config.total_dim;
    int singlet_states = config.nuclear_dim;  // 2^M

    // From Mathematica:
    // Ps = DiagonalMatrix[Join[Table[1, {2^1}], Table[0, {4*2^1 - 2^1}]]]
    // First nuclear_dim diagonal entries are 1, rest are 0
    proj.Ps = MatXc::Zero(dim, dim);
    for (int i = 0; i < singlet_states; i++)
    {
        proj.Ps(i, i) = 1.0;
    }

    // Pt = I - Ps
    proj.Pt = MatXc::Identity(dim, dim) - proj.Ps;

    return proj;
}
