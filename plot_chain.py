"""
MCMC Chain Analysis and Visualization

This script loads and processes MCMC chains from Cobaya runs,
adds derived parameters (Brans-Dicke parameters), and creates
triangle plots comparing different cosmological model runs.
"""
# pyright: reportUnusedCallResult=false
import argparse
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from pathlib import Path
from dataclasses import dataclass
from typing import Final, override
from collections.abc import Sequence
from getdist import plots, MCSamples, loadMCSamples


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
class PlotDefaults:
    """Default configuration values for plotting."""
    IGNORE_ROWS: str = "0.3"
    DPI: int = 600
    CHAIN_BASE: str = "."
    PARAMS_TO_PLOT: tuple[str, ...] = ("logA", "ns", "H0", "ombh2", "omch2", "tau")
    AXES_FONTSIZE: int = 16
    AXES_LABELSIZE: int = 18


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
    "logA": 3.044,
    "As": exp(3.044) * 1e-10,
    "ns": 0.966,
    "H0": 67.4,
    "ombh2": 0.0224,
    "omch2": 0.120,
    "tau": 0.054,
    "alphaB_BS": 0.0,
    "RPHalphaM0": 0.0,
    "RPHbraiding0": 0.0,
    "RPHalphaM_ODE0": 0.0,
    "RPHbraiding_ODE0": 0.0,
}


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
    
    def __post_init__(self):
        if self.params_to_plot is None:
            object.__setattr__(self, 'params_to_plot', DEFAULTS.PARAMS_TO_PLOT)
        if self.chain_labels is None:
            object.__setattr__(self, 'chain_labels', self.chain_names)


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
        help="Create projection plot instead of triangle plot"
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
    plot_settings.axes_fontsize = fontsize
    plot_settings.axes_labelsize = labelsize
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
    Apply parameter name aliases to chain.
    
    Args:
        chain: MCSamples object to modify
        rename_map: Dictionary mapping old parameter names to new names
    """
    chain.updateRenames(rename_map)


def update_latex_labels(
    chain: MCSamples,
    rename_map: dict[str, str],
    latex_labels: dict[str, str]
) -> None:
    """
    Update LaTeX labels for parameters.
    
    Args:
        chain: MCSamples object to modify
        rename_map: Dictionary of parameter renames
        latex_labels: Dictionary mapping parameter names to LaTeX labels
    """
    # Update labels for renamed parameters
    for old_name, new_name in rename_map.items():
        param = chain.paramNames.parWithName(old_name)
        if param and new_name in latex_labels:
            param.label = latex_labels[new_name]
    
    # Update labels for direct parameter specifications
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
    update_latex_labels(chain, rename_map, latex_labels)


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
    Validate that requested parameters exist in the chains.
    
    Args:
        chains: List of MCSamples to check
        params: List of parameter names to validate (or None for all)
        
    Returns:
        Validated parameter list or None
        
    Raises:
        ValueError: If any requested parameter doesn't exist in the chains
    """
    if params is None:
        return None
    
    param_names = chains[0].getParamNames()
    missing_params = [p for p in params if param_names.parWithName(p) is None]
    
    if not missing_params:
        return params
    
    # Build error message with suggestions
    available_params = set(param_names.list())
    suggestions = build_parameter_suggestions(missing_params, available_params)
    
    error_parts = [
        f"Parameters not found: {', '.join(missing_params)}",
        f"\nAvailable: {', '.join(sorted(available_params))}"
    ]
    
    if suggestions:
        error_parts.append("\nSuggestions:")
        error_parts.extend(
            f"  {missing} -> {', '.join(similar)}" 
            for missing, similar in suggestions.items()
        )
    
    raise ValueError('\n'.join(error_parts))


def create_triangle_plot(
    chains: Sequence[MCSamples],
    params: Sequence[str] | None,
    markers: dict[str, float],
    settings: plots.GetDistPlotSettings,
    save_path: Path,
    dpi: int,
    title: str | None = None
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
    """
    try:
        g = plots.get_subplot_plotter(settings=settings)
        g.triangle_plot(chains, params, filled=True, markers=markers)
        
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
    title: str | None = None
) -> None:
    """
    Create and save projection plot showing parameter constraints.
    
    Args:
        chains: List of MCSamples to plot
        params: List of parameter names to include (or None for all)
        markers: Dictionary of marker positions for reference values
        settings: Plot settings object
        save_path: Path where to save the figure
        dpi: Resolution for saved figure
        title: Optional title for the plot
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
        
        # Calculate subplot layout: at most 3 columns
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(4 * n_cols, max(3, n_chains * 0.6) * n_rows),
            squeeze=False
        )
        
        # Flatten axes array for easier iteration
        axes_flat = axes.flatten()
        
        # Define colors for different statistics
        colors = plt.cm.tab10(np.arange(n_chains))
        
        # Precompute statistics for all chains
        chain_stats = []
        for chain in chains:
            # Get MAP (Maximum A Posteriori) - sample with highest likelihood
            map_index = np.argmin(chain.loglikes)
            map_samples = chain.samples[map_index]
            
            # Get statistics for each parameter
            param_stats = {}
            for param in params:
                # logger.warning(param)
                # Use parWithName which handles renames
                param_obj = chain.paramNames.parWithName(param)
                if param_obj is not None:
                    # param_index = chain.paramNames.numberOfName(param)
                    orig_name = chain.paramNames.parWithName(param).name
                    param_index = chain.index[orig_name]
                    samples = chain.samples[:, param_index]
                    weights = chain.weights
                    mean = np.average(samples, weights=weights)

                    stats = chain.getMargeStats().parWithName(orig_name)
                    lower = stats.limits[1].lower
                    upper = stats.limits[1].upper
                    # logger.warning(map_samples[param_index])
                    # TODO: double-check if mean is correct here
                    param_stats[param] = {
                        'mean': mean,
                        'map': map_samples[param_index],
                        'lower_95': lower,
                        'upper_95': upper,
                        # 'lower_95': np.percentile(samples, 2.5),
                        # 'upper_95': np.percentile(samples, 97.5)
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
                    color=colors[j],
                    capsize=5,
                    capthick=2,
                    markersize=10,
                    label=chain.label if i == 0 else None,
                    alpha=0.5
                )
                
                # Plot MAP as a diamond marker
                ax.plot(
                    map_value, y_positions[j],
                    marker='D',
                    markerfacecolor='none',
                    markeredgecolor=colors[j],
                    markeredgewidth=1.5,
                    markersize=6,
                    alpha=0.8
                )
            
            # Add reference value if available
            if param in markers:
                ax.axvline(markers[param], color='black', linestyle='--', 
                          linewidth=1, alpha=0.5, label='Planck' if i == 0 else None)
            
            # Formatting
            ax.set_xlabel(param_label, fontsize=settings.axes_labelsize)
            ax.set_yticks(y_positions)
            # Reverse the labels to match reversed y_positions
            ax.set_yticklabels([chain.label for chain in chains], 
                              fontsize=settings.axes_fontsize)
            ax.tick_params(axis='x', labelsize=settings.axes_fontsize, rotation=45)
            # ax.grid(True, alpha=0.3, axis='x')
            ax.set_ylim(-0.5, n_chains - 0.5)
            
            # Only show legend on first subplot
            # if i == 0:
            #     # Create custom legend
            #     from matplotlib.lines import Line2D
            #     legend_elements = [
            #         Line2D([0], [0], marker='o', color='w', 
            #               markerfacecolor='gray', markersize=8, 
            #               label='Mean ± 95% CI'),
            #         Line2D([0], [0], marker='D', color='w',
            #               markerfacecolor='gray', markersize=6,
            #               label='MAP')
            #     ]
            #     if any(param in markers for param in params):
            #         legend_elements.append(
            #             Line2D([0], [0], color='black', linestyle='--',
            #                   linewidth=1, label='Planck')
            #         )
            #     ax.legend(handles=legend_elements, loc='best', 
            #              fontsize=settings.axes_fontsize - 2)
        
        # Hide unused subplots
        for i in range(n_params, len(axes_flat)):
            axes_flat[i].axis('off')
        
        # Add overall title if provided
        if title:
            fig.suptitle(title, fontsize=settings.axes_labelsize + 2, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()
        
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close("all")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate chain labels
    if args.chain_labels and len(args.chain_labels) != len(args.chain_names):
        raise ValueError(
            f"Number of labels ({len(args.chain_labels)}) must match "
            f"number of chains ({len(args.chain_names)})"
        )
    
    # Use first chain name as savename if not specified
    savename = args.save_name if args.save_name else args.chain_names[0]
    
    # Setup configuration
    config = PlotConfig(
        chain_bases=args.chain_bases,
        chain_names=args.chain_names,
        chain_labels=args.chain_labels,
        savename=savename,
        ignore_rows=args.ignore_rows,
        params_to_plot=args.params,
        dpi=args.dpi,
        title=args.title,
        projection_plot=args.projection
    )
    
    # Setup paths and settings
    figpath = Path(args.output).expanduser().resolve()
    figpath.mkdir(parents=True, exist_ok=True)
    plot_settings = configure_plot_settings(
        fontsize=config.axes_fontsize,
        labelsize=config.axes_labelsize
    )
    
    # Chain loading settings
    chain_settings = {"ignore_rows": config.ignore_rows}
    
    # Load all chains
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
        chains.append(chain)
    
    # Process all chains
    for chain in chains:
        process_chain(chain, RENAME_MAP, LATEX_LABELS)
    
    # Validate parameters after processing
    if config.params_to_plot:
        try:
            validated_params = validate_parameters(chains, config.params_to_plot)
        except ValueError as e:
            logger.error(f"\n{e}")
            return
    else:
        validated_params = None
        logger.info("Plotting all available parameters")
    
    # Create and save plot
    save_path = figpath / f"{config.savename}.png"
    
    if config.projection_plot:
        create_projection_plot(
            chains=chains,
            params=validated_params,
            markers=PLANCK_PARAMS,
            settings=plot_settings,
            save_path=save_path,
            dpi=config.dpi,
            title=config.title
        )
    else:
        create_triangle_plot(
            chains=chains,
            params=validated_params,
            markers=PLANCK_PARAMS,
            settings=plot_settings,
            save_path=save_path,
            dpi=config.dpi,
            title=config.title
        )
    
    plot_type = "projection" if config.projection_plot else "triangle"
    logger.info(f"{plot_type.capitalize()} plot saved to: {save_path}")
    logger.info(f"Plotted {len(chains)} chain(s)")


if __name__ == "__main__":
    main()
