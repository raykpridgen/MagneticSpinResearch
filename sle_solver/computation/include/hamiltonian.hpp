#ifndef SLE_HAMILTONIAN_HPP
#define SLE_HAMILTONIAN_HPP

#include "operators.hpp"
#include "config.hpp"

/**
 * Construct the Hamiltonian for N electrons + M nuclei
 * 
 * From Mathematica:
 *   H = g·μB·Bz·(Sz1 + Sz2) + g·μB·a·(Ix·Sx1 + Iy·Sy1 + Iz·Sz1)
 * 
 * Term 1: Zeeman splitting for all electrons
 * Term 2: Hyperfine interaction with first electron only
 * 
 * @param krons     Kronecker-lifted operators
 * @param config    System configuration
 * @param params    Physical parameters (g, mu, a)
 * @param Bz        Magnetic field strength (mT)
 * @return          Hamiltonian matrix (total_dim x total_dim)
 */
MatXc construct_hamiltonian(
    const KronOperators& krons,
    const SystemConfig& config,
    const PhysicalParams& params,
    double Bz
);

#endif // SLE_HAMILTONIAN_HPP
