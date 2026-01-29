"""
SLE Solver Visualization Module

Provides matplotlib-based plotting for SLE simulation results.
Calls the C++ computation component via subprocess.
"""

from .runner import run_simulation, find_binary
from .plotter import load_results, plot_singlet_vs_bz, show_plot, save_plot
from .cli import main

__all__ = [
    'run_simulation',
    'find_binary',
    'load_results',
    'plot_singlet_vs_bz',
    'show_plot',
    'save_plot',
    'main',
]
