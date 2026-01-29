#!/usr/bin/env python3
"""
Command-line interface for SLE Solver visualization.

Usage:
    python -m sle_solver.visualization.cli --cpu [options]
    python -m sle_solver.visualization.cli --gpu [options]
    python -m sle_solver.visualization.cli --plot-only results.csv
"""

import argparse
import sys
from pathlib import Path

from runner import run_simulation
from plotter import load_results, plot_singlet_vs_bz, show_plot, save_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description='SLE Solver - Run simulation and visualize results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CPU simulation and display plot
  python -m sle_solver.visualization.cli --cpu

  # Run GPU simulation with custom parameters
  python -m sle_solver.visualization.cli --gpu -n 3 --bz-min -20 --bz-max 20

  # Plot existing results without running simulation
  python -m sle_solver.visualization.cli --plot-only output/results.csv

  # Save plot to file instead of displaying
  python -m sle_solver.visualization.cli --cpu --save-plot plot.png
        """
    )
    
    # Mode selection (mutually exclusive group)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU solver (Eigen)'
    )
    mode_group.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU solver (CUDA)'
    )
    mode_group.add_argument(
        '--plot-only',
        metavar='FILE',
        type=str,
        help='Skip simulation, plot existing CSV file'
    )
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation parameters')
    sim_group.add_argument(
        '-n', '--electrons',
        type=int,
        default=2,
        help='Number of electrons (1-4, default: 2)'
    )
    sim_group.add_argument(
        '-g',
        type=float,
        default=2.003,
        help='g-factor (default: 2.003)'
    )
    sim_group.add_argument(
        '--mu',
        type=float,
        default=5.788e-8,
        help='Bohr magneton in eV/mT (default: 5.788e-8)'
    )
    sim_group.add_argument(
        '--hbar',
        type=float,
        default=6.582e-16,
        help='Reduced Planck constant in eV*s (default: 6.582e-16)'
    )
    sim_group.add_argument(
        '-a',
        type=float,
        default=1.0,
        help='Hyperfine coupling constant (default: 1.0)'
    )
    sim_group.add_argument(
        '--ks',
        type=float,
        default=4e6,
        help='Singlet recombination rate (default: 4e6)'
    )
    sim_group.add_argument(
        '--kd',
        type=float,
        default=1e6,
        help='Dephasing rate (default: 1e6)'
    )
    sim_group.add_argument(
        '--bz-min',
        type=float,
        default=-10.0,
        help='Minimum Bz in mT (default: -10)'
    )
    sim_group.add_argument(
        '--bz-max',
        type=float,
        default=10.0,
        help='Maximum Bz in mT (default: 10)'
    )
    sim_group.add_argument(
        '--bz-step',
        type=float,
        default=0.02,
        help='Bz step size in mT (default: 0.02)'
    )
    sim_group.add_argument(
        '--fudge',
        type=float,
        default=1.0,
        help='Fudge factor (default: 1.0)'
    )
    
    # Output options
    out_group = parser.add_argument_group('Output options')
    out_group.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output CSV file (default: output/results.csv)'
    )
    out_group.add_argument(
        '--save',
        action='store_true',
        help='Append timestamp to CSV filename'
    )
    out_group.add_argument(
        '--save-plot',
        metavar='FILE',
        type=str,
        help='Save plot to file instead of displaying'
    )
    out_group.add_argument(
        '--title',
        type=str,
        help='Custom plot title'
    )
    out_group.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    out_group.add_argument(
        '--no-plot',
        action='store_true',
        help='Run simulation without plotting'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine CSV file path
    if args.plot_only:
        # Plot existing file
        csv_path = Path(args.plot_only)
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}", file=sys.stderr)
            sys.exit(1)
    else:
        # Run simulation
        mode = 'gpu' if args.gpu else 'cpu'
        
        try:
            csv_path = run_simulation(
                mode=mode,
                n_electrons=args.electrons,
                g=args.g,
                mu=args.mu,
                hbar=args.hbar,
                a=args.a,
                ks=args.ks,
                kd=args.kd,
                bz_min=args.bz_min,
                bz_max=args.bz_max,
                bz_step=args.bz_step,
                fudge=args.fudge,
                output_file=args.output,
                save=args.save,
                verbose=args.verbose,
            )
            
            if args.verbose:
                print(f"Results saved to: {csv_path}")
                
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Skip plotting if requested
    if args.no_plot:
        print(f"Simulation complete: {csv_path}")
        return
    
    # Load and plot results
    try:
        bz, singlet = load_results(csv_path)
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create title
    if args.title:
        title = args.title
    elif not args.plot_only:
        mode_str = 'GPU' if args.gpu else 'CPU'
        title = f'SLE Simulation ({args.electrons} electrons, {mode_str})'
    else:
        title = None
    
    # Create plot
    fig = plot_singlet_vs_bz(bz, singlet, title=title)
    
    # Save or display
    if args.save_plot:
        save_plot(fig, Path(args.save_plot))
    else:
        show_plot(fig)


if __name__ == '__main__':
    main()
