"""
Matplotlib plotting functions for SLE simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import Tuple, Optional


def load_results(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load simulation results from CSV file.
    
    Args:
        csv_path: Path to CSV file with columns (Bz, singlet_population)
    
    Returns:
        Tuple of (Bz array, singlet_population array)
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    
    if data.ndim == 1:
        # Single data point
        bz = np.array([data[0]])
        singlet = np.array([data[1]])
    else:
        bz = data[:, 0]
        singlet = data[:, 1]
    
    return bz, singlet


def plot_singlet_vs_bz(
    bz: np.ndarray,
    singlet: np.ndarray,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    color: str = 'blue',
    linewidth: float = 1.5,
) -> Figure:
    """
    Create a matplotlib figure of singlet population vs magnetic field.
    
    Args:
        bz: Magnetic field values (mT)
        singlet: Singlet population values
        title: Plot title (optional)
        figsize: Figure size in inches
        color: Line color
        linewidth: Line width
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(bz, singlet, color=color, linewidth=linewidth)
    
    ax.set_xlabel('Magnetic Field Bz (mT)', fontsize=12)
    ax.set_ylabel('Singlet Population', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('SLE Simulation: Singlet Population vs Magnetic Field', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim(bz.min(), bz.max())
    
    # Add some padding to y-axis
    y_margin = (singlet.max() - singlet.min()) * 0.05
    if y_margin > 0:
        ax.set_ylim(singlet.min() - y_margin, singlet.max() + y_margin)
    
    fig.tight_layout()
    
    return fig


def show_plot(fig: Figure) -> None:
    """
    Display the plot in a window.
    
    Args:
        fig: matplotlib Figure to display
    """
    plt.show()


def save_plot(fig: Figure, path: Path, dpi: int = 150) -> None:
    """
    Save the plot to a file.
    
    Args:
        fig: matplotlib Figure to save
        path: Output file path (e.g., 'plot.png', 'plot.pdf')
        dpi: Resolution for raster formats
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {path}")


def plot_comparison(
    data_list: list,
    labels: list,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """
    Plot multiple simulation results for comparison.
    
    Args:
        data_list: List of (bz, singlet) tuples
        labels: List of labels for each dataset
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for (bz, singlet), label, color in zip(data_list, labels, colors):
        ax.plot(bz, singlet, label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Magnetic Field Bz (mT)', fontsize=12)
    ax.set_ylabel('Singlet Population', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('SLE Simulation Comparison', fontsize=14)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig
