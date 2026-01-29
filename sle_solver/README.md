# SLE Solver

Stochastic Liouville Equation (SLE) solver for spin systems, implementing the NZFMR (Near Zero Field Magnetoresistance) simulation from the Mathematica reference code.

## Overview

This project is organized into two decoupled components:

1. **Computation** (C++/CUDA): Solves the SLE and outputs results to CSV
2. **Visualization** (Python/matplotlib): Calls computation and displays plots

The components communicate via subprocess and CSV files, allowing them to be used independently.

## Directory Structure

```
sle_solver/
├── computation/
│   ├── include/          # C++ headers
│   │   ├── config.hpp
│   │   ├── operators.hpp
│   │   ├── hamiltonian.hpp
│   │   ├── sle_solver.hpp
│   │   └── cuda_solver.hpp
│   ├── src/              # C++ source files
│   │   ├── operators.cpp
│   │   ├── hamiltonian.cpp
│   │   ├── sle_solver.cpp
│   │   ├── cuda_solver.cu
│   │   └── main.cpp
│   └── build/            # Compiled binaries
├── visualization/
│   ├── __init__.py
│   ├── runner.py         # Subprocess wrapper
│   ├── plotter.py        # matplotlib functions
│   └── cli.py            # Command-line interface
├── output/               # CSV output files
├── Makefile
├── requirements.txt
└── README.md
```

## Dependencies

### C++ Computation
- **Eigen3** (required): Matrix operations
- **CUDA Toolkit** (optional): GPU acceleration

On Ubuntu/Debian:
```bash
sudo apt install libeigen3-dev

# For GPU support:
# Install CUDA Toolkit from NVIDIA
```

### Python Visualization
- numpy
- matplotlib

```bash
pip install -r requirements.txt
# or
make python-deps
```

## Building

```bash
cd sle_solver

# Build CPU-only version
make cpu

# Build GPU version (requires CUDA)
make gpu

# Build both
make both
```

## Usage

### C++ CLI (Computation Only)

```bash
# CPU solver
./computation/build/sle_solver_cpu --cpu [options]

# GPU solver
./computation/build/sle_solver_cuda --gpu [options]
```

**Required**: Must specify `--cpu` or `--gpu`. The solver will fail if `--gpu` is requested but no CUDA device is available.

**Options**:
```
  -n, --electrons N  Number of electrons (1-4, default: 2)
  -g G_FACTOR        g-factor (default: 2.003)
  --mu MU            Bohr magneton in eV/mT (default: 5.788e-8)
  --hbar HBAR        Reduced Planck constant in eV*s (default: 6.582e-16)
  -a HYPERFINE       Hyperfine coupling constant (default: 1.0)
  --ks KS            Singlet recombination rate (default: 4e6)
  --kd KD            Dephasing rate (default: 1e6)
  --bz-min MIN       Minimum Bz in mT (default: -10)
  --bz-max MAX       Maximum Bz in mT (default: 10)
  --bz-step STEP     Bz step size in mT (default: 0.02)
  --fudge F          Fudge factor (default: 1.0)
  -o, --output FILE  Output CSV file (default: output/results.csv)
  --save             Append timestamp to filename (don't overwrite)
  -v, --verbose      Print detailed progress
  -h, --help         Show help
```

**Examples**:
```bash
# Basic 2-electron simulation
./computation/build/sle_solver_cpu --cpu -n 2 -v

# 3-electron with custom field range
./computation/build/sle_solver_cpu --cpu -n 3 --bz-min -20 --bz-max 20 --bz-step 0.1

# GPU simulation with saved output
./computation/build/sle_solver_cuda --gpu -n 2 --save -o output/my_results.csv
```

### Python CLI (Visualization)

```bash
# Run simulation and display plot
python -m sle_solver.visualization.cli --cpu
python -m sle_solver.visualization.cli --gpu -n 3

# Plot existing CSV without running simulation
python -m sle_solver.visualization.cli --plot-only output/results.csv

# Save plot to file
python -m sle_solver.visualization.cli --cpu --save-plot plot.png

# Run simulation without plotting
python -m sle_solver.visualization.cli --cpu --no-plot
```

### Python API

```python
from sle_solver.visualization import run_simulation, load_results, plot_singlet_vs_bz, show_plot

# Run simulation
csv_path = run_simulation(
    mode='cpu',
    n_electrons=2,
    bz_min=-10,
    bz_max=10,
    verbose=True
)

# Load and plot results
bz, singlet = load_results(csv_path)
fig = plot_singlet_vs_bz(bz, singlet, title='My Simulation')
show_plot(fig)
```

## Physics Background

This solver implements the Stochastic Liouville Equation for radical pair spin dynamics:

```
-(i/ℏ)[H,ρ] - (1/2)(Ks+Kd){Ps,ρ} - (1/2)Kd{Pt,ρ} + (1/dim)I = 0
```

Where:
- **H**: Hamiltonian (Zeeman + hyperfine interaction)
- **ρ**: Density matrix (steady-state solution)
- **Ps, Pt**: Singlet/triplet projection operators
- **Ks**: Singlet recombination rate
- **Kd**: Dephasing rate

The Hamiltonian is:
```
H = g·μB·Bz·(Sz1 + Sz2 + ...) + g·μB·a·(Ix·Sx1 + Iy·Sy1 + Iz·Sz1)
```

The observable is the singlet population: `Tr(Ps·ρ)`

## Output Format

CSV with two columns:
```csv
Bz,singlet_population
-10.0,0.1234567890
-9.98,0.1234567891
...
```

## Testing

```bash
# Test CPU build
make test-cpu

# Test GPU build
make test-gpu

# Benchmark CPU vs GPU
make benchmark
```

## Correspondence to Mathematica Code

| Mathematica | C++ File | Function |
|-------------|----------|----------|
| `sz1, sz2, sx1, sy1...` | operators.cpp | `get_spin_operators()` |
| `KroneckerProduct[sx1, IdentityMatrix[2]]` | operators.cpp | `get_lifted_operators()` |
| `Ps, Pt` | operators.cpp | `get_proj_operators()` |
| `g mu B Bz * (Sz1 + Sz2) + ...` | hamiltonian.cpp | `construct_hamiltonian()` |
| `Commute[H, rho]` | sle_solver.cpp | Vectorized via `(I⊗H - H^T⊗I)` |
| `NSolve[... == 0]` | sle_solver.cpp | `solve_steady_state_cpu/gpu()` |
| `Tr[Ps . rho]` | sle_solver.cpp | `compute_singlet_population()` |
| `ParallelTable[{Bz, ...}]` | main.cpp | Loop in `run_sweep()` |
