"""
Nested Sampling Analysis and Visualization

This script loads and processes nested sampling runs from Cobaya/PolyChord,
adds derived parameters (Brans-Dicke parameters), and creates
triangle plots comparing different cosmological model runs.

Uses anesthetic for loading and plotting nested sampling chains.
"""
# pyright: reportUnusedCallResult=false
import argparse
import json
import logging
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import exp
from pathlib import Path
from dataclasses import dataclass
from typing import Final, override
from collections.abc import Sequence
from anesthetic import read_chains, NestedSamples
from anesthetic.plot import make_2d_axes


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
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# Constants
@dataclass(frozen=True)
class PlotDefaults:
    """Default configuration values for plotting."""
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
    "alphaB": r"c_B",
    "alphaM": r"c_M",
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
    params_to_plot: Sequence[str] | None = None
    dpi: int = DEFAULTS.DPI
    axes_fontsize: int = DEFAULTS.AXES_FONTSIZE
    axes_labelsize: int = DEFAULTS.AXES_LABELSIZE
    title: str | None = None
    projection_plot: bool = False
    output_format: str = "png"

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
        description="Analyze and visualize nested sampling chains from Cobaya/PolyChord runs"
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
    parser.add_argument(
        "--tension",
        action="store_true",
        help="Print Bayesian tension statistics between the first two chains"
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
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def chain_exists(chain_path: Path) -> bool:
    """
    Check if nested sampling chain files exist at the given path.

    Anesthetic expects either a .txt dead file (PolyChord) or similar
    flat files. We look for the dead-birth file or standard chain files.

    Args:
        chain_path: Path to check (without extension)

    Returns:
        True if chain files exist, False otherwise
    """
    parent = chain_path.parent
    name = chain_path.name
    if not parent.exists():
        return False
    return bool(
        list(parent.glob(f"{name}_dead*.txt"))
        or list(parent.glob(f"{name}.txt"))
        or list(parent.glob(f"{name}*.hdf5"))
    )


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


def load_chain(chain_path: Path, chain_name: str) -> NestedSamples:
    """
    Load a single nested sampling chain with error handling.

    Uses anesthetic's read_chains which auto-detects PolyChord /
    MultiNest / plain formats.

    Args:
        chain_path: Full path to the chain root (without extension)
        chain_name: Name of the chain (for error messages)

    Returns:
        Loaded NestedSamples object

    Raises:
        RuntimeError: If chain loading fails
    """
    try:
        return read_chains(str(chain_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load chain '{chain_name}': {e}") from e


def add_derived_braiding_param(samples: NestedSamples) -> None:
    """
    Add derived Brans-Dicke braiding parameter (alphaB_BS = -2 * alphaB)
    to the NestedSamples DataFrame.

    Args:
        samples: NestedSamples object to modify in-place
    """
    braiding_cols = [c for c in samples.columns if "braiding" in str(c)]
    if braiding_cols:
        col = braiding_cols[0]
        samples["alphaB_BS"] = -2.0 * samples[col]
        samples.set_label("alphaB_BS", r"c_B^{B\&S}")


def apply_parameter_renames(
    samples: NestedSamples,
    rename_map: dict[str, str]
) -> None:
    """
    Rename columns in the NestedSamples DataFrame.

    Args:
        samples: NestedSamples object to modify in-place
        rename_map: Dictionary mapping old column names to new names
    """
    existing = {k: v for k, v in rename_map.items() if k in samples.columns}
    if existing:
        samples.rename(columns=existing, inplace=True)


def update_latex_labels(
    samples: NestedSamples,
    rename_map: dict[str, str],
    latex_labels: dict[str, str]
) -> None:
    """
    Update LaTeX labels for parameters in NestedSamples.

    Args:
        samples: NestedSamples object to modify in-place
        rename_map: Dictionary of parameter renames (old -> new)
        latex_labels: Dictionary mapping parameter names to LaTeX labels
    """
    for new_name in rename_map.values():
        if new_name in latex_labels and new_name in samples.columns:
            samples.set_label(new_name, latex_labels[new_name])

    for param, label in latex_labels.items():
        if param in samples.columns:
            samples.set_label(param, label)


def process_chain(
    samples: NestedSamples,
    rename_map: dict[str, str],
    latex_labels: dict[str, str]
) -> None:
    """
    Apply renaming and add derived parameters to NestedSamples.

    Args:
        samples: NestedSamples object to process in-place
        rename_map: Dictionary mapping old parameter names to new names
        latex_labels: Dictionary mapping parameter names to LaTeX labels
    """
    add_derived_braiding_param(samples)
    apply_parameter_renames(samples, rename_map)
    update_latex_labels(samples, rename_map, latex_labels)


def build_parameter_suggestions(
    missing_params: Sequence[str],
    available_params: set[str]
) -> dict[str, list[str]]:
    """
    Build suggestions for missing parameters based on name similarity.

    Args:
        missing_params: Parameters that were not found
        available_params: Set of available parameter names

    Returns:
        Dictionary mapping missing params to suggested available params
    """
    suggestions: dict[str, list[str]] = {}
    for missing in missing_params:
        similar = [
            p for p in available_params
            if missing.lower() in p.lower() or p.lower() in missing.lower()
        ][:5]
        if similar:
            suggestions[missing] = similar
    return suggestions


def validate_parameters(
    chains: Sequence[NestedSamples],
    params: Sequence[str] | None
) -> Sequence[str] | None:
    """
    Validate that requested parameters exist in the chains.

    Args:
        chains: List of NestedSamples to check
        params: List of parameter names to validate (or None for all)

    Returns:
        Validated parameter list, or None if params is None

    Raises:
        ValueError: If any requested parameter is missing from the chains
    """
    if params is None:
        return None

    cols = chains[0].columns
    if isinstance(cols, pd.MultiIndex):
        available = set(cols.get_level_values(0).astype(str))
    else:
        available = set(cols.astype(str))
    missing_params = [p for p in params if p not in available]

    if not missing_params:
        return params

    suggestions = build_parameter_suggestions(missing_params, available)

    error_parts = [
        f"Parameters not found: {', '.join(missing_params)}",
        f"\nAvailable: {', '.join(sorted(available))}"
    ]
    if suggestions:
        error_parts.append("\nSuggestions:")
        error_parts.extend(
            f"  {m} -> {', '.join(s)}"
            for m, s in suggestions.items()
        )

    raise ValueError('\n'.join(error_parts))


def _resolve_params(
    chains: Sequence[NestedSamples],
    params: Sequence[str] | None
) -> list[str]:
    """Return the parameter list, defaulting to all numeric columns."""
    if params is not None:
        return list(params)
    cols = chains[0].columns
    if isinstance(cols, pd.MultiIndex):
        all_cols = cols.get_level_values(0).astype(str).tolist()
    else:
        all_cols = cols.astype(str).tolist()
    exclude = {"weight", "logL", "logL_birth", "nlive"}
    return [
        c for c in all_cols
        if c not in exclude
        and np.issubdtype(chains[0][c].dtype, np.number)
        and chains[0][c].std() > 0
    ]


def create_triangle_plot(
    chains: Sequence[NestedSamples],
    labels: Sequence[str],
    params: Sequence[str] | None,
    markers: dict[str, float],
    save_path: Path,
    dpi: int,
    axes_fontsize: int = DEFAULTS.AXES_FONTSIZE,
    axes_labelsize: int = DEFAULTS.AXES_LABELSIZE,
    title: str | None = None
) -> None:
    """
    Create and save a triangle (corner) plot using anesthetic.

    Args:
        chains: List of NestedSamples to plot
        labels: Display labels for each chain
        params: Parameters to include (None = all)
        markers: Reference values to mark on each axis
        save_path: Where to save the figure
        dpi: Figure resolution
        axes_fontsize: Tick label font size
        axes_labelsize: Axis label font size
        title: Optional figure title
    """
    param_list = _resolve_params(chains, params)
    n = len(param_list)

    tex_labels: dict[str, str] = {}
    for p in param_list:
        raw = chains[0].get_label(p) if hasattr(chains[0], "get_label") else None
        if raw and raw != p:
            tex_labels[p] = raw

    try:
        def has_variance(param: str) -> bool:
            return all(
                chain[param].std() > 0
                for chain in chains
                if param in chain.columns
            )

        dropped = [p for p in param_list if not has_variance(p)]
        if dropped:
            logger.warning(f"Dropping zero-variance parameters: {', '.join(dropped)}")
        param_list = [p for p in param_list if has_variance(p)]
        n = len(param_list)
        tex_labels = {k: v for k, v in tex_labels.items() if k in param_list}

        fig, axes = make_2d_axes(
            param_list,
            labels=tex_labels,
            figsize=(2.5 * n, 2.5 * n),
            upper=False,
        )

        colors = plt.cm.tab10(np.arange(len(chains)))

        for chain, label, color in zip(chains, labels, colors):
            chain.plot_2d(
                axes,
                kinds={"diagonal": "kde_1d", "lower": "kde_2d"},
                label=label,
                color=color,
                alpha=0.8,
            )

        marker_subset = {p: v for p, v in markers.items() if p in param_list}
        if marker_subset:
            axes.axlines(marker_subset, color="black", linestyle="--",
                         linewidth=1, alpha=0.5)

        axes.tick_params(labelsize=axes_fontsize, axis="both")
        axes.tick_params(axis="x", rotation=45)
        axes.tick_params(axis="y", rotation=45)

        axes.iloc[-1, 0].legend(
            loc="lower center",
            bbox_to_anchor=(n / 2, n),
            fontsize=axes_fontsize,
            frameon=False,
        )

        if title:
            fig.suptitle(title, fontsize=axes_labelsize + 2, y=1.01)

        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close("all")


def create_projection_plot(
    chains: Sequence[NestedSamples],
    labels: Sequence[str],
    params: Sequence[str] | None,
    markers: dict[str, float],
    save_path: Path,
    dpi: int,
    axes_fontsize: int = DEFAULTS.AXES_FONTSIZE,
    axes_labelsize: int = DEFAULTS.AXES_LABELSIZE,
    title: str | None = None
) -> None:
    """
    Create and save a projection plot showing per-parameter constraints.

    Args:
        chains: List of NestedSamples to plot
        labels: Display labels for each chain
        params: Parameters to include (None = all)
        markers: Reference values to mark on each axis
        save_path: Where to save the figure
        dpi: Figure resolution
        axes_fontsize: Tick label font size
        axes_labelsize: Axis label font size
        title: Optional figure title
    """
    param_list = _resolve_params(chains, params)
    n_params = len(param_list)
    n_chains = len(chains)

    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    colors = plt.cm.tab10(np.arange(n_chains))

    try:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, max(3, n_chains * 0.7) * n_rows),
            squeeze=False
        )
        axes_flat = axes.flatten()

        chain_stats: list[dict[str, dict[str, float]]] = []
        for chain in chains:
            stats: dict[str, dict[str, float]] = {}
            for param in param_list:
                if param not in chain.columns:
                    continue
                col_data = chain[param]
                weights = chain.get_weights() if hasattr(chain, "get_weights") else chain["weight"]

                mean = float(np.average(col_data, weights=weights))

                sorted_idx = np.argsort(col_data)
                cum_w = np.cumsum(weights[sorted_idx])
                cum_w /= cum_w[-1]
                lower = float(col_data.iloc[sorted_idx[np.searchsorted(cum_w, 0.025)]])
                upper = float(col_data.iloc[sorted_idx[np.searchsorted(cum_w, 0.975)]])

                if "logL" in chain.columns:
                    map_idx = int(chain["logL"].idxmax())
                    map_val = float(chain.loc[map_idx, param])
                else:
                    map_val = mean

                stats[param] = {
                    "mean": mean,
                    "lower_95": lower,
                    "upper_95": upper,
                    "map": map_val,
                }
            chain_stats.append(stats)

        for i, param in enumerate(param_list):
            ax = axes_flat[i]
            y_pos = np.arange(n_chains)[::-1]

            raw_label = (
                chains[0].get_label(param)
                if hasattr(chains[0], "get_label")
                else param
            )
            param_label = f"${raw_label}$" if raw_label else param

            for j, (chain, stats, color) in enumerate(zip(chains, chain_stats, colors)):
                if param not in stats:
                    continue
                s = stats[param]
                mean, lower, upper, map_val = (
                    s["mean"], s["lower_95"], s["upper_95"], s["map"]
                )

                ax.errorbar(
                    mean, y_pos[j],
                    xerr=[[mean - lower], [upper - mean]],
                    fmt="o",
                    color=color,
                    capsize=5,
                    capthick=2,
                    markersize=10,
                    alpha=0.6,
                )
                ax.plot(
                    map_val, y_pos[j],
                    marker="D",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    markersize=7,
                    alpha=0.9,
                )

            if param in markers:
                ax.axvline(
                    markers[param], color="black", linestyle="--",
                    linewidth=1, alpha=0.5,
                    label="Planck" if i == 0 else None
                )

            ax.set_xlabel(param_label, fontsize=axes_labelsize)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(labels), fontsize=axes_fontsize)
            ax.tick_params(axis="x", labelsize=axes_fontsize, rotation=45)
            ax.set_ylim(-0.5, n_chains - 0.5)

        for k in range(n_params, len(axes_flat)):
            axes_flat[k].axis("off")

        if title:
            fig.suptitle(title, fontsize=axes_labelsize + 2, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        else:
            plt.tight_layout()

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close("all")


def print_tension_statistics(
    chain_a: NestedSamples,
    chain_b: NestedSamples,
    label_a: str,
    label_b: str,
) -> None:
    """
    Print Bayesian tension statistics between two nested sampling runs.

    Args:
        chain_a: First NestedSamples object
        chain_b: Second NestedSamples object
        label_a: Display label for chain A
        label_b: Display label for chain B
    """
    try:
        from anesthetic.tension import tension_stats  # type: ignore[import]
        stats = tension_stats(chain_a, chain_b)
        logger.info(f"Tension statistics: {label_a!r} vs {label_b!r}")
        for key, val in stats.items():
            logger.info(f"  {key}: {val:.4g}")
    except ImportError:
        logger.warning(
            "anesthetic.tension not available — "
            "update anesthetic to access tension statistics."
        )
    except Exception as e:
        logger.error(f"Tension statistics failed: {e}")


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

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
        params_to_plot=args.params,
        dpi=args.dpi,
        title=args.title,
        projection_plot=args.projection,
        output_format=args.format,
    )

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

    chains: list[NestedSamples] = []
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
        chain = load_chain(chain_path, chain_name)
        logger.debug(f"  {len(chain)} samples, {len(chain.columns)} columns")
        chains.append(chain)

    renamed = [k for k in effective_rename if any(k in c.columns for c in chains)]
    if renamed:
        logger.debug(f"Renaming: {', '.join(f'{k} -> {effective_rename[k]}' for k in renamed)}")

    for chain in chains:
        process_chain(chain, effective_rename, effective_latex)

    if args.list_params:
        cols = chains[0].columns
        available = (
            cols.get_level_values(0).astype(str).tolist()
            if isinstance(cols, pd.MultiIndex)
            else cols.astype(str).tolist()
        )
        print("\n".join(sorted(set(available))))
        return

    if args.tension and len(chains) >= 2:
        print_tension_statistics(
            chains[0], chains[1],
            config.chain_labels[0], config.chain_labels[1]  # type: ignore[index]
        )

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

    plot_kwargs = dict(
        chains=chains,
        labels=config.chain_labels,
        params=validated_params,
        markers=effective_reference,
        save_path=save_path,
        dpi=config.dpi,
        axes_fontsize=config.axes_fontsize,
        axes_labelsize=config.axes_labelsize,
        title=config.title,
    )

    if config.projection_plot:
        create_projection_plot(**plot_kwargs)  # type: ignore[arg-type]
    else:
        create_triangle_plot(**plot_kwargs)  # type: ignore[arg-type]

    plot_type = "projection" if config.projection_plot else "triangle"
    logger.info(f"{plot_type.capitalize()} plot saved to: {save_path}")
    logger.info(f"Plotted {len(chains)} chain(s)")


if __name__ == "__main__":
    main()
