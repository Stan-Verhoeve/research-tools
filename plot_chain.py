"""
MCMC Chain Analysis and Visualization

This script loads and processes MCMC chains from Cobaya runs,
adds derived parameters (Brans-Dicke parameters), and creates
triangle plots comparing different cosmological model runs.
"""
# pyright: reportUnusedCallResult=false
import argparse
import json
import logging
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from pathlib import Path
from dataclasses import dataclass
from typing import Final, override
from collections.abc import Sequence
from getdist import plots, MCSamples, loadMCSamples
from research_tools.style import configure_publication_style, PlotStyleDefaults, STYLE_DEFAULTS

# getdist 1.7.x added ParamBounds.periodic, but both 1.4.x and 1.7.x share
# pickle_version=22, so 1.7.x loads 1.4.x caches and finds objects where
# periodic was never set. Patching __setstate__ injects it when missing.
try:
    from getdist.parampriors import ParamBounds as _ParamBounds
    _orig_pb_setstate = getattr(_ParamBounds, '__setstate__', None)

    def _pb_compat_setstate(self, state: dict) -> None:
        if _orig_pb_setstate is not None:
            _orig_pb_setstate(self, state)
        else:
            self.__dict__.update(state)
        if 'periodic' not in self.__dict__:
            self.__dict__['periodic'] = set()

    _ParamBounds.__setstate__ = _pb_compat_setstate
    del _ParamBounds, _orig_pb_setstate, _pb_compat_setstate
except (ImportError, AttributeError):
    pass


# Setup logging with colored output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored log levels."""
    
    COLORS: Final[dict[str, str]] = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Orange/Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET: Final[str] = '\033[0m'
    
    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color: bool = use_color
    
    @override
    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            log_color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname = f"{log_color}[{record.levelname}]{self.RESET}"
        else:
            record.levelname = f"[{record.levelname}]"
        return super().format(record)

# Check if terminal supports colors (not redirected to file)
use_color = sys.stdout.isatty()

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(levelname)s %(message)s', use_color=use_color))
logger = logging.getLogger(__name__)
logger.handlers.clear()  # Clear any existing handlers
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger


# Constants
@dataclass(frozen=True)
class PlotDefaults(PlotStyleDefaults):
    """Default configuration values for plot_chain."""
    IGNORE_ROWS: str = "0.3"
    CHAIN_BASE: str = "."
    PARAMS_TO_PLOT: tuple[str, ...] = ("logA", "ns", "H0", "ombh2", "omch2", "tau")


DEFAULTS = PlotDefaults()

RENAME_MAP: Final[dict[str, str]] = {
    "RPHalphaM0": "alphaM",
    "RPHbraiding0": "alphaB",
    "RPHalphaM_ODE0": "alphaM",
    "RPHbraiding_ODE0": "alphaB"
}

LATEX_LABELS: Final[dict[str, str]] = {
    "alphaB": "c_B",
    "alphaM": "c_M",
    "mnu": r"\sum m_{\nu}"
}

PLANCK_PARAMS: Final[dict[str, float]] = {
    "logA": 3.040,
    "As": exp(3.040) * 1e-10,
    "ns": 0.9681,
    "H0": 67.64,
    "ombh2": 0.02226,
    "omch2": 0.1188,
    "tau": 0.0580,
    "alphaB_BS": 0.0,
    "RPHalphaM0": 0.0,
    "RPHbraiding0": 0.0,
    "RPHalphaM_ODE0": 0.0,
    "RPHbraiding_ODE0": 0.0,
}


def load_json_file(path: str) -> dict:
    """
    Load a JSON file and return its contents as a dict.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary with the file contents

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not valid JSON or not an object
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with p.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def resolve_reference_renames(
    reference: dict[str, float],
    rename_map: dict[str, str]
) -> dict[str, float]:
    """
    Translate reference dict keys from old parameter names to new ones.

    Since apply_parameter_renames commits renames directly into param.name,
    GetDist's marker lookup (which uses param.name) expects the new names.
    Both old names (e.g. from PLANCK_PARAMS) and new names pass through
    correctly: old names are translated, new names are left unchanged.

    Args:
        reference: Dictionary of reference parameter values
        rename_map: Dictionary mapping old parameter names to new names

    Returns:
        New dictionary with renamed keys
    """
    return {rename_map.get(key, key): value for key, value in reference.items()}


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    chain_bases: Sequence[str]
    chain_names: Sequence[str]
    chain_labels: Sequence[str] | None
    savename: str
    ignore_rows: str = DEFAULTS.IGNORE_ROWS
    params_to_plot: Sequence[str] | None = None
    dpi: int = DEFAULTS.DPI
    axes_fontsize: int = DEFAULTS.AXES_FONTSIZE
    axes_labelsize: int = DEFAULTS.AXES_LABELSIZE
    title: str | None = None
    projection_plot: bool = False
    projection2d: bool = False
    output_format: str = "png"
    colors: Sequence[str] | None = None
    usetex: bool = False
    no_legend: bool = False

    def __post_init__(self):
        if self.params_to_plot is None:
            self.params_to_plot = DEFAULTS.PARAMS_TO_PLOT
        if self.chain_labels is None:
            self.chain_labels = list(self.chain_names)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Analyze and visualize MCMC chains from Cobaya runs"
    )
    parser.add_argument(
        "-b", "--chain-bases",
        type=str,
        nargs="+",
        default=[DEFAULTS.CHAIN_BASE],
        help="Path(s) to search for chains. Each entry is a directory that may contain the requested chain names. (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./figures",
        help="Directory to save figures (default: ./figures)"
    )
    parser.add_argument(
        "-c", "--chain-names",
        type=str,
        nargs="+",
        required=True,
        help="Names of chains to load (space-separated)"
    )
    parser.add_argument(
        "-l", "--chain-labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for chains in plot (default: uses chain names)"
    )
    parser.add_argument(
        "-s", "--save-name",
        type=str,
        default=None,
        help="Name for saved figure (default: first chain name)"
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default=None,
        help="Title for the plot (default: no title)"
    )
    parser.add_argument(
        "-i", "--ignore-rows",
        type=str,
        default=DEFAULTS.IGNORE_ROWS,
        help=f"Fraction of rows to ignore as burn-in (default: {DEFAULTS.IGNORE_ROWS})"
    )
    parser.add_argument(
        "-d", "--dpi",
        type=int,
        default=DEFAULTS.DPI,
        help=f"DPI for saved figure (default: {DEFAULTS.DPI})"
    )
    parser.add_argument(
        "-p", "--params",
        type=str,
        nargs="+",
        default=list(DEFAULTS.PARAMS_TO_PLOT),
        help="Parameters to plot (default: logA ns H0 ombh2 omch2 tau)"
    )
    parser.add_argument(
        "--projection",
        action="store_true",
        help="Create 2D triangle plot with MAP diamonds overlaid"
    )
    parser.add_argument(
        "--projection1d",
        action="store_true",
        help="Create 1D projection plot with error bars (original projection mode)"
    )
    parser.add_argument(
        "--rename",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to JSON file with parameter rename map (merged with and overriding built-in renames)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to JSON file with reference/baseline values for markers (merged with and overriding built-in Planck values; may use renamed parameter names)"
    )
    parser.add_argument(
        "--latex-labels",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to JSON file with LaTeX labels for parameters (merged with and overriding built-in labels)"
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Suppress all reference markers (including built-in Planck values)"
    )
    parser.add_argument(
        "-L", "--list-params",
        action="store_true",
        help="Print available parameter names after loading and exit"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output figure format (default: png)"
    )
    parser.add_argument(
        "--colors",
        type=str,
        nargs="+",
        default=None,
        metavar="COLOR",
        help="Colors for each chain's contours (matplotlib color names or hex codes)"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Suppress the legend in the plot"
    )
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Use LaTeX for text rendering (requires a working LaTeX installation)"
    )
    parser.add_argument(
        "--plot-settings",
        type=str,
        default=None,
        metavar="JSON",
        help='JSON string of GetDistPlotSettings overrides, e.g. \'{"fig_width_inch": 3.04}\''
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()





def configure_plot_settings(
    fontsize: int = DEFAULTS.AXES_FONTSIZE,
    labelsize: int = DEFAULTS.AXES_LABELSIZE
) -> plots.GetDistPlotSettings:
    """
    Create and configure plot settings.

    Args:
        fontsize: Font size for axes
        labelsize: Font size for labels

    Returns:
        Configured GetDistPlotSettings object
    """
    plot_settings = plots.GetDistPlotSettings()
    plot_settings.scaling = False
    plot_settings.axes_fontsize = fontsize
    plot_settings.axes_labelsize = labelsize
    plot_settings.legend_fontsize = DEFAULTS.LEGEND_FONTSIZE
    plot_settings.lw_contour = DEFAULTS.LW_CONTOUR
    plot_settings.fig_width_inch = DEFAULTS.FIGWIDTH
    plot_settings.axis_tick_x_rotation = 45
    plot_settings.axis_tick_y_rotation = 45
    return plot_settings


def chain_exists(chain_path: Path) -> bool:
    """
    Check if chain files exist at the given path.
    
    Args:
        chain_path: Path to check (without extension)
        
    Returns:
        True if chain files exist, False otherwise
    """
    return (chain_path.parent.exists() and 
            bool(list(chain_path.parent.glob(f"{chain_path.name}*.txt"))))


def find_chain_in_bases(
    base_dirs: Sequence[str],
    chain_name: str
) -> Path | None:
    """
    Search for a chain across multiple base directories.

    Args:
        base_dirs: List of paths to search (absolute or relative to CWD)
        chain_name: Name of the chain to find

    Returns:
        Full path to the chain if found, None otherwise
    """
    for base_dir in base_dirs:
        chain_path = Path(base_dir) / chain_name
        if chain_exists(chain_path):
            return chain_path
    return None


def load_chain(
    chain_path: Path,
    chain_name: str,
    settings: dict[str, str]
) -> MCSamples:
    """
    Load a single MCMC chain with error handling.
    
    Args:
        chain_path: Full path to the chain (without extension)
        chain_name: Name of the chain (for error messages)
        settings: Dictionary of chain loading settings
        
    Returns:
        Loaded MCSamples object
        
    Raises:
        RuntimeError: If chain loading fails
    """
    try:
        return loadMCSamples(str(chain_path), settings=settings)
    except Exception as e:
        raise RuntimeError(f"Failed to load chain {chain_name}: {e}")


def add_derived_braiding_param(chain: MCSamples) -> None:
    """
    Add derived Brans-Dicke parameter to chain.
    
    Args:
        chain: MCSamples object to modify in-place
    """
    braiding_names = [
        p.name for p in chain.paramNames.names
        if "braiding" in p.name
    ]
    if braiding_names:
        aB = getattr(chain.getParams(), braiding_names[0])
        chain.addDerived(-2 * aB, name="alphaB_BS", label=r"c_B^{B&S}")


def apply_parameter_renames(
    chain: MCSamples,
    rename_map: dict[str, str]
) -> None:
    """
    Commit parameter renames directly into param.name and rebuild the index.

    Unlike updateRenames (which only adds aliases into par.renames), this
    overwrites par.name with the new name so that GetDist's canonical name
    — used for marker lookup, density caching, and statistics — is the
    renamed one. The name→column index is rebuilt afterwards.

    Args:
        chain: MCSamples object to modify
        rename_map: Dictionary mapping old parameter names to new names
    """
    for par in chain.paramNames.names:
        if par.name in rename_map:
            par.name = rename_map[par.name]
    # Rebuild the name→column index to reflect the new canonical names
    chain.index = {par.name: i for i, par in enumerate(chain.paramNames.names)}


def update_latex_labels(
    chain: MCSamples,
    latex_labels: dict[str, str]
) -> None:
    """
    Update LaTeX labels for parameters.

    Since apply_parameter_renames commits renames into par.name, labels can
    be looked up directly by the (possibly renamed) parameter name.

    Args:
        chain: MCSamples object to modify
        latex_labels: Dictionary mapping parameter names to LaTeX labels
    """
    for parname, label in latex_labels.items():
        param = chain.paramNames.parWithName(parname)
        if param:
            param.label = label


def process_chain(
    chain: MCSamples,
    rename_map: dict[str, str],
    latex_labels: dict[str, str]
) -> None:
    """
    Apply renaming and add derived parameters to chain.

    Args:
        chain: MCSamples object to process
        rename_map: Dictionary mapping old parameter names to new names
        latex_labels: Dictionary mapping parameter names to LaTeX labels
    """
    add_derived_braiding_param(chain)
    apply_parameter_renames(chain, rename_map)
    update_latex_labels(chain, latex_labels)


def load_map_from_bestfit(
    chain: MCSamples,
    rename_map: dict[str, str],
    max_posterior: bool = True,
) -> dict[str, float] | None:
    """
    Load MAP values from the chain's .minimum (or .bestfit) file via getdist.

    getdist writes pre-rename Cobaya parameter names into the file, so the
    rename map is applied to the returned keys. Python-side derived params
    (e.g. alphaB_BS added by our script) are absent from the file; callers
    should fall back per-parameter for those.

    Args:
        chain: MCSamples object (must have chain.root set, i.e. loaded from file).
        rename_map: The effective rename map used when processing the chain.
        max_posterior: True to load .minimum (MAP), False for .bestfit (MLE).

    Returns:
        Dict mapping renamed param names to MAP values, or None if unavailable.
    """
    try:
        raw = chain.getBestFit(max_posterior=max_posterior).getParamDict(include_derived=True)
        result = {rename_map.get(k, k): v for k, v in raw.items()
                  if k not in ('weight', 'loglike')}
        ext = '.minimum' if max_posterior else '.bestfit'
        logger.debug(f"  Loaded MAP from {ext} ({len(result)} params)")
        return result
    except Exception as exc:
        logger.debug(f"  No bestfit file or failed to load: {exc}")
        return None


def build_parameter_suggestions(
    missing_params: Sequence[str],
    available_params: set[str]
) -> dict[str, list[str]]:
    """
    Build suggestions for missing parameters based on similarity.
    
    Args:
        missing_params: List of parameters that weren't found
        available_params: Set of available parameter names
        
    Returns:
        Dictionary mapping missing params to similar available params
    """
    suggestions = {}
    for missing in missing_params:
        similar = [
            p for p in available_params 
            if missing.lower() in p.lower() or p.lower() in missing.lower()
        ][:5]
        if similar:
            suggestions[missing] = similar
    return suggestions


def validate_parameters(
    chains: Sequence[MCSamples],
    params: Sequence[str] | None
) -> Sequence[str] | None:
    """
    Validate that requested parameters exist in all chains.

    Args:
        chains: chains to check
        params: parameter names to validate (None means all)

    Raises:
        ValueError: if any parameter is missing from any chain
    """
    if params is None:
        return None

    chain_missing: dict[str, list[str]] = {}
    for chain in chains:
        param_names = chain.getParamNames()
        missing = [p for p in params if param_names.parWithName(p) is None]
        if missing:
            chain_missing[chain.label] = missing

    if not chain_missing:
        return params

    all_missing = sorted({p for missing in chain_missing.values() for p in missing})
    available_params = set(chains[0].getParamNames().list())
    suggestions = build_parameter_suggestions(all_missing, available_params)

    error_parts = ["Parameters not found:"]
    for label, missing in chain_missing.items():
        error_parts.append(f"  {label}: {', '.join(missing)}")
    error_parts.append(f"\nAvailable: {', '.join(sorted(available_params))}")

    if suggestions:
        error_parts.append("\nSuggestions:")
        error_parts.extend(
            f"  {p}: {', '.join(similar)}"
            for p, similar in suggestions.items()
        )

    raise ValueError('\n'.join(error_parts))


def create_triangle_plot(
    chains: Sequence[MCSamples],
    params: Sequence[str] | None,
    markers: dict[str, float],
    settings: plots.GetDistPlotSettings,
    save_path: Path,
    dpi: int,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    no_legend: bool = False,
) -> None:
    """
    Create and save triangle plot.

    Args:
        chains: List of MCSamples to plot
        params: List of parameter names to include (or None for all)
        markers: Dictionary of marker positions
        settings: Plot settings object
        save_path: Path where to save the figure
        dpi: Resolution for saved figure
        title: Optional title for the plot
        colors: Optional list of colors for each chain's contours
        no_legend: Suppress the legend
    """
    try:
        g = plots.get_subplot_plotter(settings=settings)
        contour_colors = list(colors[:len(chains)]) if colors else None
        g.triangle_plot(chains, params, filled=True, markers=markers,
                        contour_colors=contour_colors,
                        legend_labels=[] if no_legend else None)
        
        # Add title if provided
        if title:
            fig = plt.gcf()
            
            # First apply tight_layout to get proper spacing
            plt.tight_layout()
            
            # Add title with proper positioning
            # Use a text object positioned in figure coordinates
            fig.text(0.5, 0.98, title, 
                    ha='center', va='top',
                    fontsize=settings.axes_labelsize + 2,
                    transform=fig.transFigure)
            
            # Add some padding at the top
            fig.subplots_adjust(top=0.95)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close("all")


def create_projection_plot(
    chains: Sequence[MCSamples],
    params: Sequence[str] | None,
    markers: dict[str, float],
    settings: plots.GetDistPlotSettings,
    save_path: Path,
    dpi: int,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    chain_maps: Sequence[dict[str, float] | None] | None = None,
) -> None:
    """
    Create and save a projection plot of parameter constraints.

    Args:
        chains: chains to plot
        params: parameters to include (None for all)
        markers: reference marker values keyed by parameter name
        settings: plot settings
        save_path: output path
        dpi: figure resolution
        title: optional figure title
        colors: optional list of colors for each chain
        chain_maps: per-chain MAP dicts from .minimum files; falls back to
            best sample when None or when a parameter is absent
    """
    try:
        # Get parameter list
        if params is None:
            params = chains[0].getParamNames().list()

        # Get parameter labels
        param_labels = []
        for param in params:
            p = chains[0].paramNames.parWithName(param)
            if p and p.label:
                param_labels.append(f"${p.label}$")
            else:
                param_labels.append(param)

        n_params = len(params)
        n_chains = len(chains)

        # Calculate subplot layout: at most 2 columns
        n_cols = min(2, n_params)
        n_rows = int(np.ceil(n_params / n_cols))

        # Create figure with subplots
        panel_h = n_chains * 0.45 + 0.5
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(STYLE_DEFAULTS.FIGWIDTH, panel_h * n_rows),
            squeeze=False,
            constrained_layout=True,
            sharey=True,
        )

        # Flatten axes array for easier iteration
        axes_flat = axes.flatten()

        # Resolve per-chain colors, falling back to tab10 for any unspecified chains
        if colors:
            import matplotlib.colors as mcolors
            resolved_colors = [
                mcolors.to_rgba(colors[i]) if i < len(colors)
                else plt.cm.tab10(i)
                for i in range(n_chains)
            ]
        else:
            resolved_colors = [plt.cm.tab10(i) for i in range(n_chains)]
        
        # Precompute statistics for all chains
        chain_stats = []
        for i, chain in enumerate(chains):
            file_map = chain_maps[i] if chain_maps is not None else None
            # Fallback: best sample by likelihood (used when file MAP is absent)
            map_index = np.argmin(chain.loglikes)
            map_samples = chain.samples[map_index]

            param_stats = {}
            for param in params:
                param_obj = chain.paramNames.parWithName(param)
                if param_obj is not None:
                    param_index = chain.index[param_obj.name]
                    samples = chain.samples[:, param_index]
                    mean = np.average(samples, weights=chain.weights)
                    marge_stats = chain.getMargeStats().parWithName(param_obj.name)
                    map_val = (
                        file_map[param]
                        if file_map is not None and param in file_map
                        else map_samples[param_index]
                    )
                    param_stats[param] = {
                        'mean': mean,
                        'map': map_val,
                        'lower_95': marge_stats.limits[1].lower,
                        'upper_95': marge_stats.limits[1].upper,
                    }

            chain_stats.append(param_stats)
        
        # For each parameter, create a subplot
        for i, (param, param_label) in enumerate(zip(params, param_labels)):
            ax = axes_flat[i]
            
            # Y-positions for each chain (reversed so first chain is at top)
            y_positions = np.arange(n_chains)[::-1]
            
            for j, (chain, stats) in enumerate(zip(chains, chain_stats)):
                if param not in stats:
                    continue
                
                # Get precomputed statistics
                param_stat = stats[param]
                mean = param_stat['mean']
                map_value = param_stat['map']
                lower = param_stat['lower_95']
                upper = param_stat['upper_95']
                
                # Plot mean with 95% error bars
                ax.errorbar(
                    mean, y_positions[j],
                    xerr=[[mean - lower], [upper - mean]],
                    fmt='o',
                    color=resolved_colors[j],
                    capsize=3,
                    capthick=0.8,
                    lw=STYLE_DEFAULTS.LW_CONTOUR,
                    markersize=5,
                    label=chain.label if i == 0 else None,
                    alpha=0.8
                )

                # Plot MAP as a diamond marker
                ax.plot(
                    map_value, y_positions[j],
                    marker='D',
                    markerfacecolor='none',
                    markeredgecolor=resolved_colors[j],
                    markeredgewidth=1.0,
                    markersize=4,
                    alpha=0.9
                )
            
            if param in markers:
                ax.axvline(markers[param], color='black', linestyle='--', linewidth=1, alpha=0.5)

            ax.set_xlabel(param_label, fontsize=settings.axes_labelsize)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([chain.label for chain in chains],
                               fontsize=settings.axes_fontsize)
            ax.tick_params(axis='x', labelsize=settings.axes_fontsize, rotation=45)
            ax.set_ylim(-0.5, n_chains - 0.5)
        
        # Hide unused subplots
        for i in range(n_params, len(axes_flat)):
            axes_flat[i].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=settings.axes_labelsize + 2)
        
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close("all")


def _contrasting_edge(color: str) -> str:
    """Return 'white' or 'black' depending on the luminance of color."""
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(color)
    return 'white' if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else 'black'


def create_projection2d_plot(
    chains: Sequence[MCSamples],
    params: Sequence[str] | None,
    markers: dict[str, float],
    settings: plots.GetDistPlotSettings,
    save_path: Path,
    dpi: int,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    chain_maps: Sequence[dict[str, float] | None] | None = None,
    no_legend: bool = False,
) -> None:
    """
    Triangle plot with filled MAP diamonds overlaid on every 2D panel.

    Args:
        chains: chains to plot
        params: parameters to include (None for all)
        markers: reference marker values for 1D diagonal panels
        settings: GetDist plot settings (fig_width_inch already set)
        save_path: output path
        dpi: figure resolution
        title: optional figure title
        colors: contour colors for each chain; defaults to tab10 if not given
        chain_maps: per-chain MAP dicts from .minimum files
    """
    if params is None:
        params = list(chains[0].getParamNames().list())

    n_chains = len(chains)
    resolved_colors: list[str] = (
        [str(colors[i]) for i in range(min(len(colors), n_chains))]
        if colors
        else [f"C{i}" for i in range(n_chains)]
    )

    try:
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(chains, params, filled=True, markers=markers,
                        contour_colors=resolved_colors,
                        legend_labels=[] if no_legend else None)

        if chain_maps is not None:
            n_params = len(params)
            for i in range(1, n_params):
                for j in range(i):
                    ax = g.subplots[i, j]
                    if ax is None:
                        continue
                    for color, cmap in zip(resolved_colors, chain_maps):
                        if cmap is None:
                            continue
                        col_p, row_p = params[j], params[i]
                        if col_p in cmap and row_p in cmap:
                            ax.plot(
                                cmap[col_p], cmap[row_p],
                                marker='D', ms=3, zorder=5,
                                mfc=color,
                                mec=_contrasting_edge(color),
                                mew=0.5, ls='none',
                            )

        if title:
            plt.gcf().suptitle(title, fontsize=settings.axes_labelsize + 2, y=1.01)

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    finally:
        plt.close('all')


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    configure_publication_style(usetex=args.usetex)

    if args.chain_labels and len(args.chain_labels) != len(args.chain_names):
        raise ValueError(
            f"Number of labels ({len(args.chain_labels)}) must match "
            f"number of chains ({len(args.chain_names)})"
        )

    savename = args.save_name if args.save_name else args.chain_names[0]

    config = PlotConfig(
        chain_bases=args.chain_bases,
        chain_names=args.chain_names,
        chain_labels=args.chain_labels,
        savename=savename,
        ignore_rows=args.ignore_rows,
        params_to_plot=args.params,
        dpi=args.dpi,
        title=args.title,
        projection_plot=args.projection1d,
        projection2d=args.projection,
        output_format=args.format,
        colors=args.colors,
        usetex=args.usetex,
        no_legend=args.no_legend,
    )
    
    plot_settings = configure_plot_settings(
        fontsize=config.axes_fontsize,
        labelsize=config.axes_labelsize
    )

    if args.plot_settings:
        for key, val in json.loads(args.plot_settings).items():
            if not hasattr(plot_settings, key):
                warnings.warn(f"--plot-settings: unknown key '{key}', ignored")
            else:
                setattr(plot_settings, key, val)

    effective_rename: dict[str, str] = dict(RENAME_MAP)
    if args.rename:
        effective_rename.update(load_json_file(args.rename))

    effective_latex: dict[str, str] = dict(LATEX_LABELS)
    if args.latex_labels:
        effective_latex.update(load_json_file(args.latex_labels))

    if args.no_reference:
        effective_reference: dict[str, float] = {}
    else:
        effective_reference = resolve_reference_renames(dict(PLANCK_PARAMS), effective_rename)
        if args.reference:
            loaded_ref: dict[str, float] = load_json_file(args.reference)
            effective_reference.update(resolve_reference_renames(loaded_ref, effective_rename))

    chain_settings = {"ignore_rows": config.ignore_rows}

    chains = []
    for chain_name, chain_label in zip(config.chain_names, config.chain_labels):
        logger.info(f"Searching for chain: {chain_name}")
        chain_path = find_chain_in_bases(config.chain_bases, chain_name)
        if chain_path is None:
            error_msg = (
                f"Chain '{chain_name}' not found in any of the base directories: "
                f"{', '.join(config.chain_bases)}"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        logger.info(f"  Found in: {chain_path.parent.name}")
        chain = load_chain(chain_path, chain_name, chain_settings)
        chain.label = chain_label
        logger.debug(f"  {chain.samples.shape[0]} samples, {len(chain.paramNames.names)} parameters")
        chains.append(chain)

    renamed = [k for k in effective_rename if any(
        p.name == k for c in chains for p in c.paramNames.names
    )]
    if renamed:
        logger.debug(f"Renaming: {', '.join(f'{k} -> {effective_rename[k]}' for k in renamed)}")

    for chain in chains:
        process_chain(chain, effective_rename, effective_latex)

    chain_maps = [load_map_from_bestfit(chain, effective_rename) for chain in chains]
    if any(m is not None for m in chain_maps):
        logger.debug("Using MAP from .minimum file(s) for projection plot")

    if args.list_params:
        available = chains[0].getParamNames().list()
        print("\n".join(sorted(available)))
        return

    figpath = Path(args.output).expanduser().resolve()
    figpath.mkdir(parents=True, exist_ok=True)

    if config.params_to_plot:
        try:
            validated_params = validate_parameters(chains, config.params_to_plot)
        except ValueError as e:
            logger.error(f"\n{e}")
            return
    else:
        validated_params = None
        logger.info("Plotting all available parameters")

    save_path = figpath / f"{config.savename}.{config.output_format}"

    if config.projection2d:
        create_projection2d_plot(
            chains=chains,
            params=validated_params,
            markers=effective_reference,
            settings=plot_settings,
            save_path=save_path,
            dpi=config.dpi,
            title=config.title,
            colors=config.colors,
            chain_maps=chain_maps,
            no_legend=config.no_legend,
        )
        plot_type = "projection2d"
    elif config.projection_plot:
        create_projection_plot(
            chains=chains,
            params=validated_params,
            markers=effective_reference,
            settings=plot_settings,
            save_path=save_path,
            dpi=config.dpi,
            title=config.title,
            colors=config.colors,
            chain_maps=chain_maps,
        )
        plot_type = "projection1d"
    else:
        create_triangle_plot(
            chains=chains,
            params=validated_params,
            markers=effective_reference,
            settings=plot_settings,
            save_path=save_path,
            dpi=config.dpi,
            title=config.title,
            colors=config.colors,
            no_legend=config.no_legend,
        )
        plot_type = "triangle"
    logger.info(f"{plot_type.capitalize()} plot saved to: {save_path}")
    logger.info(f"Plotted {len(chains)} chain(s)")


if __name__ == "__main__":
    main()
