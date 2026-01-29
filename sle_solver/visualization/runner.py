"""
Subprocess wrapper for the C++ SLE solver.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional


def find_binary(mode: str) -> Path:
    """
    Find the SLE solver binary for the given mode.
    
    Args:
        mode: "cpu" or "gpu"
    
    Returns:
        Path to the binary
    
    Raises:
        FileNotFoundError: If binary not found
    """
    # Get the directory containing this file
    this_dir = Path(__file__).parent.resolve()
    sle_solver_dir = this_dir.parent
    
    if mode == "cpu":
        binary_name = "sle_solver_cpu"
    elif mode == "gpu":
        binary_name = "sle_solver_cuda"
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'cpu' or 'gpu'")
    
    # Check in computation/build directory
    binary_path = sle_solver_dir / "computation" / "build" / binary_name
    
    if not binary_path.exists():
        raise FileNotFoundError(
            f"Binary not found: {binary_path}\n"
            f"Build it with: cd {sle_solver_dir} && make {mode}"
        )
    
    return binary_path


def run_simulation(
    mode: str,
    n_electrons: int = 2,
    g: float = 2.003,
    mu: float = 5.788e-8,
    hbar: float = 6.582e-16,
    a: float = 1.0,
    ks: float = 4e6,
    kd: float = 1e6,
    bz_min: float = -10.0,
    bz_max: float = 10.0,
    bz_step: float = 0.02,
    fudge: float = 1.0,
    output_file: Optional[str] = None,
    save: bool = False,
    verbose: bool = False,
) -> Path:
    """
    Run the C++ SLE solver via subprocess.
    
    Args:
        mode: "cpu" or "gpu"
        n_electrons: Number of electrons (1-4)
        g: g-factor
        mu: Bohr magneton (eV/mT)
        hbar: Reduced Planck constant (eV*s)
        a: Hyperfine coupling constant
        ks: Singlet recombination rate
        kd: Dephasing rate
        bz_min: Minimum magnetic field (mT)
        bz_max: Maximum magnetic field (mT)
        bz_step: Magnetic field step (mT)
        fudge: Fudge factor
        output_file: Output CSV path (default: output/results.csv)
        save: Append timestamp to filename
        verbose: Print detailed output
    
    Returns:
        Path to the output CSV file
    
    Raises:
        FileNotFoundError: If binary not found
        RuntimeError: If simulation fails
    """
    binary = find_binary(mode)
    
    # Build command
    cmd = [
        str(binary),
        f"--{mode}",
        "-n", str(n_electrons),
        "-g", str(g),
        "--mu", str(mu),
        "--hbar", str(hbar),
        "-a", str(a),
        "--ks", str(ks),
        "--kd", str(kd),
        "--bz-min", str(bz_min),
        "--bz-max", str(bz_max),
        "--bz-step", str(bz_step),
        "--fudge", str(fudge),
    ]
    
    if output_file:
        cmd.extend(["-o", output_file])
    
    if save:
        cmd.append("--save")
    
    if verbose:
        cmd.append("-v")
    
    # Run the command
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(binary.parent.parent.parent),  # sle_solver directory
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"Simulation failed: {error_msg}")
    
    # Parse output to get the CSV file path
    output = result.stdout.strip()
    if verbose:
        print(output)
    
    # The last line of output is the CSV file path (when not verbose)
    # When verbose, we need to look for "Output:" line
    if verbose:
        for line in output.split('\n'):
            if line.strip().startswith("Output:"):
                csv_path = line.split(":", 1)[1].strip()
                break
        else:
            csv_path = output_file or "output/results.csv"
    else:
        csv_path = output.split('\n')[-1].strip()
    
    # Resolve relative to sle_solver directory
    sle_solver_dir = binary.parent.parent.parent
    csv_path = sle_solver_dir / csv_path
    
    if not csv_path.exists():
        raise RuntimeError(f"Output file not created: {csv_path}")
    
    return csv_path
