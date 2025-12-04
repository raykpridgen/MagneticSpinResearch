# Matrix Solver Format Guide

## Input Format (From Mathematica Export)

Your input files use **Python-style** complex notation with **parentheses**:
```
[[1.0, Complex(0.0, 0.25)], [Complex(2.0, -1.0), 3.0]]
```

This format is parsed correctly by the C++ solver.

## Output Format (To Mathematica Import)

The solver outputs in **native Mathematica format** with **square brackets**:
```
{1.0, Complex[0.0, 0.25], Complex[2.0, -1.0], 3.0}
```

### Why the Difference?

- **Input**: Your Mathematica code exports using `Complex(real, imag)` - possibly from Python/NumPy-style export
- **Output**: We use `Complex[real, imag]` - Mathematica's native function syntax

This ensures the output is **valid Mathematica code** that can be directly evaluated.

## Importing Solutions in Mathematica

### Quick Import (One Line)

```mathematica
solution = Get["data/mma_out/matrix_data_1_solution.txt"]
```

That's it! The file is valid Mathematica code.

### Example Session

```mathematica
(* Import solution *)
solution = Get["data/mma_out/matrix_data_1_solution.txt"];

(* Check dimensions *)
Length[solution]
(* Output: 64 *)

(* Verify it's a list of numbers *)
Head[solution[[1]]]
(* Output: Real *)

Head[solution[[2]]]
(* Output: Complex *)

(* Use the solution *)
Norm[solution]
(* Output: 0.708975 *)

Max[Abs[solution]]
(* Output: 0.146693 *)

(* Plot if desired *)
ListPlot[Abs[solution], Joined -> True]
```

### Alternative: Using Import[]

```mathematica
(* Import as string, then evaluate *)
solution = ToExpression[Import["data/mma_out/matrix_data_1_solution.txt", "String"]]
```

Both methods work identically because the output file contains pure Mathematica syntax.

## Format Comparison

| Aspect | Input Format | Output Format | Mathematica Compatible? |
|--------|--------------|---------------|------------------------|
| Complex notation | `Complex(a, b)` | `Complex[a, b]` | Output: Yes ✓ |
| List brackets | `[[...]]` | `{...}` | Output: Yes ✓ |
| Direct evaluation | No | Yes | Output: Yes ✓ |

## Testing

A test script is provided to verify import compatibility:

```bash
# Run from Mathematica
<< "mma/test_import.wl"
```

Or from command line:
```bash
wolframscript -file mma/test_import.wl
```

This will verify that the solution file can be imported and used correctly.

## Summary

✓ **Input**: C++ solver correctly parses your `Complex(a,b)` format
✓ **Output**: C++ solver outputs valid Mathematica `Complex[a,b]` format
✓ **Import**: Use `Get[]` to directly load solutions into Mathematica
✓ **Compatible**: Output is native Mathematica code - no parsing needed!
