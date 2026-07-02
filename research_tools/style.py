import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class PlotStyleDefaults:
    """Visual defaults shared across publication-style plots."""
    DPI: int = 600
    AXES_FONTSIZE: int = 10
    AXES_LABELSIZE: int = 12
    LEGEND_FONTSIZE: int = 10
    LW_CONTOUR: float = 1.5
    FIGWIDTH: float = 6.08948     # LaTeX \linewidth in inches
    FIGSIZE_SINGLE: tuple = (6.08948, 4.2)


STYLE_DEFAULTS: Final = PlotStyleDefaults()


def configure_publication_style(usetex: bool = False) -> None:
    """Apply matplotlib rcParams suitable for journal publication figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'text.usetex': usetex,
        'axes.linewidth': 0.8,
        'axes.labelsize': STYLE_DEFAULTS.AXES_LABELSIZE,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.labelsize': STYLE_DEFAULTS.AXES_FONTSIZE,
        'ytick.labelsize': STYLE_DEFAULTS.AXES_FONTSIZE,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'lines.linewidth': 1.5,
        'legend.fontsize': STYLE_DEFAULTS.LEGEND_FONTSIZE,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'figure.dpi': 100,
        'figure.figsize': STYLE_DEFAULTS.FIGSIZE_SINGLE,
        'savefig.dpi': STYLE_DEFAULTS.DPI,
    })
