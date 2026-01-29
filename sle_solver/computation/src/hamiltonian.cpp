#include "hamiltonian.hpp"

MatXc construct_hamiltonian(
    const KronOperators& krons,
    const SystemConfig& config,
    const PhysicalParams& params,
    double Bz)
{
    int dim = config.total_dim;

    // From Mathematica:
    // H = g·μB·Bz·(Sz1 + Sz2) + g·μB·a·(Ix·Sx1 + Iy·Sy1 + Iz·Sz1)

    // Term 1: Zeeman splitting for all electrons
    // g·μB·Bz·(Σ Sz_i)
    MatXc zeeman = MatXc::Zero(dim, dim);
    for (int i = 0; i < config.n_electrons; i++)
    {
        zeeman += krons.Sz[i];
    }
    zeeman = (params.g * params.mu * Bz) * zeeman;

    // Term 2: Hyperfine interaction with first electron only
    // g·μB·a·(Ix·Sx1 + Iy·Sy1 + Iz·Sz1)
    MatXc hyperfine = (krons.Ix * krons.Sx[0]) +
                      (krons.Iy * krons.Sy[0]) +
                      (krons.Iz * krons.Sz[0]);
    hyperfine = (params.g * params.mu * params.a) * hyperfine;

    return zeeman + hyperfine;
}
