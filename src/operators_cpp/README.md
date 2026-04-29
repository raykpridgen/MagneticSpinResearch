# `sle.py` C++ Port (`src/operators_cpp`)

This module ports the stochastic Liouville sweep in `src/sle.py` to C++.

## Files

- `operators.hpp` / `operators.cpp`: struct and solver implementation
- `sweep_plot.cpp`: executable that runs a field sweep and plots with `matplotlib-cpp`
- `verify_cpp_vs_python.py`: compares C++ sweep output with Python `sle.py` output
- `requirements.txt`: Python deps used by verification

## Dependency setup

1. Header-only `matplotlib-cpp`:
   - place `matplotlibcpp.h` at `third_party/matplotlib-cpp/matplotlibcpp.h`
2. Python venv and packages:
   - `make operators-venv`

## Build and run

```bash
make operators-cpp
./build/operators_cpp_sweep --no-show
```

Solver backend selection:
- default: auto (uses CUDA solve path when compiled and a GPU is available)
- `--cpu`: force Eigen CPU solve path
- `--gpu`: require CUDA solve path

Problem-size controls:
- `--j-numerators 1,1,1,1` sets per-nucleus spin numerators directly
- `--n-nuclei N --nuclear-j-numerator J` builds a uniform set of `N` nuclei with numerator `J`
- if `--h-number` is not supplied with `--n-nuclei`, it defaults to `N`

## CPU vs GPU timing

```bash
src/operators_cpp/benchmark_backends.sh
```

Size-ladder examples:

```bash
# Uniform spin-1/2 nuclei counts
src/operators_cpp/benchmark_backends.sh --nuclei-list "1,2,3,4,5" -- --bz-min -5 --bz-max 5 --bz-step 0.05

# Explicit per-size j-numerator sets
src/operators_cpp/benchmark_backends.sh --sizes "1;1,1;1,1,1;1,3,3,3"
```

## Verify C++ vs Python

```bash
make operators-verify
```

That target:
- builds the C++ sweep binary
- creates/updates `.venv`
- runs both C++ and Python sweeps
- checks max absolute error against tolerance (default `1e-6`)
