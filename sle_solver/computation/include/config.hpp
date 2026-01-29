#ifndef SLE_CONFIG_HPP
#define SLE_CONFIG_HPP

#include <stdexcept>

/**
 * System configuration for N-electron, M-nuclear spin system
 * Based on Mathematica SLESolveS implementation
 */
struct SystemConfig
{
    int n_electrons;  // Number of electrons
    int n_nuclei;     // Number of nuclear spins
    int electron_dim; // 2^n_electrons
    int nuclear_dim;  // 2^n_nuclei
    int total_dim;    // electron_dim * nuclear_dim
    int matrix_size;  // total_dim^2 (vectorized density matrix size)

    SystemConfig(int n_e, int n_n = 1)
        : n_electrons(n_e), n_nuclei(n_n),
          electron_dim(1 << n_e),
          nuclear_dim(1 << n_n),
          total_dim((1 << n_e) * (1 << n_n)),
          matrix_size(((1 << n_e) * (1 << n_n)) * ((1 << n_e) * (1 << n_n)))
    {
        if (n_e < 1 || n_e > 4)
        {
            throw std::invalid_argument("n_electrons must be between 1 and 4");
        }
        if (n_n < 1)
        {
            throw std::invalid_argument("n_nuclei must be at least 1");
        }
    }
};

/**
 * Physical parameters for the simulation
 * Default values match Mathematica code
 */
struct PhysicalParams
{
    double g = 2.003;           // g-factor
    double mu = 5.788e-8;       // Bohr magneton (eV/mT)
    double hbar = 6.582e-16;    // Reduced Planck constant (eV·s)
    double a = 1.0;             // Hyperfine coupling constant
    double Ks = 4e6;            // Singlet recombination rate
    double Kd = 1e6;            // Dephasing rate
    double fudge = 1.0;         // Fudge factor for output scaling
};

/**
 * Magnetic field sweep parameters
 */
struct SweepParams
{
    double Bz_min = -10.0;      // Minimum Bz (mT)
    double Bz_max = 10.0;       // Maximum Bz (mT)
    double Bz_step = 0.02;      // Step size (mT)
};

#endif // SLE_CONFIG_HPP
