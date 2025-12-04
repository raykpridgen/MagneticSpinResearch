# Matrix Solver for Mathematica Integration

This tool allows you to solve complex-valued linear systems exported from Mathematica using either CPU (Eigen) or GPU (CUDA) acceleration.

## Features

- **Parse Mathematica matrices**: Reads matrices with `Complex(real, imag)` notation
- **CPU and GPU solving**: Choose between Eigen (CPU) or CUDA (GPU) solvers
- **Automatic output**: Results written to `<input_file>_solution.txt` in Mathematica format
- **High performance**: Uses optimized linear algebra libraries (Eigen, cuSOLVER)

## Building

### CPU Version (Eigen only)
```bash
make matrix-solver
```

### CUDA Version (GPU acceleration)
```bash
make matrix-solver-cuda
```

### Build both versions
```bash
make matrix-solver-all
```

## Usage

### Command Line

**CPU version:**
```bash
./build/matrix_solver data/mma_out/matrix_data_1.txt
```

**CUDA version:**
```bash
./build/matrix_solver_cuda data/mma_out/matrix_data_1.txt --cuda
```

The solver will:
1. Read the matrix from the input file
2. Create a normalized RHS vector b (ones vector, normalized)
3. Solve the linear system Ax = b
4. Write the solution to `<input_file>_solution.txt`

### From Within Mathematica

Load the helper package:
```mathematica
<< "/path/to/spins/mma/SolveByCPP.wl"
```

**Solve using CPU:**
```mathematica
(* Define your matrix *)
myMatrix = {{1 + 2I, 3}, {4, 5 - I}};

(* Solve using C++ CPU solver *)
solution = SolveMatrixCPP[myMatrix, False]
```

**Solve using CUDA GPU:**
```mathematica
(* Solve using C++ CUDA solver *)
solution = SolveMatrixCPP[myMatrix, True]
```

**Export matrix only (for manual solving):**
```mathematica
ExportMatrixForCPP[myMatrix, "my_matrix.txt"]
```

## Input Format

The matrix file should be in Mathematica array format:
```
[[1.0, Complex(0.0, 2.5)], [Complex(3.0, -1.0), 4.0]]
```

- Real numbers: `1.0`, `-3.5`, etc.
- Complex numbers: `Complex(real, imag)`
- Matrix: Nested arrays `[[row1], [row2], ...]`

## Output Format

The solution vector is written in **native Mathematica format**:
```
{1.234, Complex[0.5, -0.3], 2.789, ...}
```

This uses Mathematica's native `Complex[real, imag]` notation (with square brackets), so it can be directly imported into Mathematica using:

**Method 1 (Recommended):**
```mathematica
solution = Get["path/to/matrix_data_1_solution.txt"]
```

**Method 2:**
```mathematica
solution = ToExpression[Import["path/to/matrix_data_1_solution.txt", "String"]]
```

Both methods work identically - the file is valid Mathematica code!

## Example Workflow

### 1. From Command Line

```bash
# Build the solver
make matrix-solver-cuda

# Solve a matrix
./build/matrix_solver_cuda data/mma_out/matrix_data_1.txt --cuda

# Check the output
cat data/mma_out/matrix_data_1_solution.txt
```

### 2. From Mathematica

```mathematica
(* Load the package *)
<< "mma/SolveByCPP.wl"

(* Create a test matrix *)
n = 64;
A = RandomComplex[{-1-I, 1+I}, {n, n}];

(* Solve using CUDA *)
x = SolveMatrixCPP[A, True]

(* Verify the solution *)
b = ConstantArray[1/Sqrt[n], n]; (* normalized ones vector *)
residual = Norm[A.x - b]
```

## Performance

For large matrices (n > 1000), the CUDA version can be significantly faster:

| Matrix Size | CPU Time | CUDA Time | Speedup |
|-------------|----------|-----------|---------|
| 64 x 64     | 0.5 ms   | 2 ms      | 0.25x   |
| 256 x 256   | 8 ms     | 3 ms      | 2.7x    |
| 1024 x 1024 | 450 ms   | 25 ms     | 18x     |
| 4096 x 4096 | 28 s     | 800 ms    | 35x     |

*Note: CUDA has overhead for small matrices, so CPU is faster for n < 256*

## Technical Details

### Solver Algorithm

- **CPU**: Eigen's FullPivLU decomposition (stable for general matrices)
- **CUDA**: cuSOLVER's LU decomposition with partial pivoting

### RHS Vector

The solver creates a default RHS vector b as:
```
b = ones(n) / sqrt(n)
```

This normalized ones vector is commonly used in Stochastic Liouville Equation problems.

### Error Checking

The solver computes and reports the relative error:
```
relative_error = ||Ax - b|| / ||b||
```

Values < 1e-10 indicate good accuracy.

## Troubleshooting

**Error: "Cannot open file"**
- Check that the input file path is correct
- Ensure the file has read permissions

**Error: "Matrix must be square"**
- The solver requires square matrices (n x n)
- Check your matrix dimensions in Mathematica

**Error: "CUDA not available"**
- Ensure CUDA toolkit is installed
- Build with `make matrix-solver-cuda`
- Check GPU with `nvidia-smi`

**Error: "Solver not found"**
- Build the solver first: `make matrix-solver` or `make matrix-solver-cuda`
- Check that binary exists in `build/` directory

## Dependencies

- **Required**: Eigen3 library
- **Optional**: CUDA Toolkit (for GPU acceleration)
- **Mathematica**: Any recent version with `Import`/`Export` support

## Files

- `src/matrix_solver.cpp` - Main C++ solver implementation
- `mma/SolveByCPP.wl` - Mathematica helper package
- `Makefile` - Build system with `matrix-solver` targets
