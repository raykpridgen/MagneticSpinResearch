#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_csv(path: Path):
    bz = []
    y = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bz.append(float(row["Bz"]))
            y.append(float(row["singlet_trace"]))
    return np.array(bz), np.array(y)


def run_cpp(binary: Path, out_csv: Path, args):
    cmd = [
        str(binary),
        "--ks",
        str(args.ks),
        "--kd",
        str(args.kd),
        "--h-number",
        str(args.h_number),
        "--j-numerators",
        args.j_numerators,
        "--bz-min",
        str(args.bz_min),
        "--bz-max",
        str(args.bz_max),
        "--bz-step",
        str(args.bz_step),
        "--out",
        str(out_csv),
        "--no-plot",
        "--no-show",
    ]
    subprocess.run(cmd, check=True)


def run_python_reference(args):
    os.environ.setdefault("MPLBACKEND", "Agg")
    src_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(src_dir))
    import sle  # pylint: disable=import-outside-toplevel

    j_nums = [int(x.strip()) for x in args.j_numerators.split(",") if x.strip()]
    solver = sle.Solver(args.ks, args.kd, args.h_number, *j_nums)
    solver.solve(args.bz_min, args.bz_max, args.bz_step)
    bz = np.array(solver.Bzs, dtype=float)
    y = np.array([complex(v).real for v in solver.tracesinglets], dtype=float)
    return bz, y


def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python SLE sweep outputs.")
    parser.add_argument("--cpp-bin", default="build/operators_cpp_sweep")
    parser.add_argument("--ks", type=float, default=4e6)
    parser.add_argument("--kd", type=float, default=1e6)
    parser.add_argument("--h-number", type=int, default=1)
    parser.add_argument("--j-numerators", default="1")
    parser.add_argument("--bz-min", type=float, default=-10.0)
    parser.add_argument("--bz-max", type=float, default=10.0)
    parser.add_argument("--bz-step", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    cpp_csv = Path("data/operators_cpp_sweep_verify.csv")
    cpp_csv.parent.mkdir(parents=True, exist_ok=True)

    run_cpp(Path(args.cpp_bin), cpp_csv, args)
    bz_cpp, y_cpp = parse_csv(cpp_csv)
    bz_py, y_py = run_python_reference(args)

    if bz_cpp.shape != bz_py.shape or not np.allclose(bz_cpp, bz_py, atol=1e-12):
        print("FAIL: Bz grids differ between C++ and Python runs.")
        return 1

    abs_err = np.abs(y_cpp - y_py)
    max_err = float(np.max(abs_err)) if abs_err.size else 0.0
    mean_err = float(np.mean(abs_err)) if abs_err.size else 0.0
    print(f"max_abs_error={max_err:.6e}")
    print(f"mean_abs_error={mean_err:.6e}")

    if max_err > args.tol:
        print(f"FAIL: error exceeds tolerance {args.tol}")
        return 1

    print("PASS: C++ and Python sweeps match within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
