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

## Verify C++ vs Python

```bash
make operators-verify
```

That target:
- builds the C++ sweep binary
- creates/updates `.venv`
- runs both C++ and Python sweeps
- checks max absolute error against tolerance (default `1e-6`)
