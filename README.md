# research-tools

General-purpose scripts and shared utilities for research workflows.

## `research_tools` package

Importable Python package providing shared plotting utilities. Install into a Nix devshell via `buildPythonPackage` in your `flake.nix` (see below), then import from any script in that environment.

```python
from research_tools.style import configure_publication_style, PlotStyleDefaults, STYLE_DEFAULTS
```

### `research_tools.style`

| Symbol | Description |
|---|---|
| `configure_publication_style(usetex=False)` | Apply publication-quality `rcParams` (serif fonts, inward ticks, consistent sizes). Pass `usetex=True` for LaTeX rendering. |
| `PlotStyleDefaults` | Frozen dataclass with visual defaults: `DPI=600`, `AXES_FONTSIZE=14`, `AXES_LABELSIZE=16`, `LEGEND_FONTSIZE=13`, `LW_CONTOUR=1.5`. |
| `STYLE_DEFAULTS` | Module-level instance of `PlotStyleDefaults`. |

Script-specific overrides (different sizes, extra rcParams) can be applied via a follow-up `plt.rcParams.update({...})` after calling `configure_publication_style()`.

### Installation

```bash
pip install -e .
```

---

## Contents

### `plot_chain.py`

Load and visualize MCMC chains from [Cobaya](https://cobaya.readthedocs.io) runs. Produces triangle plots and projection plots via [GetDist](https://getdist.readthedocs.io). Supports derived parameter computation, parameter renaming, and multi-chain comparisons.

**Dependencies:** `getdist`, `matplotlib`, `numpy`

```bash
# Compare two chains, plotting a subset of parameters
python plot_chain.py -b /path/to/chains -c chain1 chain2 -l "label1" "label2" -p param1 param2

# Projection plot across all default parameters
python plot_chain.py -b /path/to/chains -c chain1 chain2 --projection

python plot_chain.py --help  # full list of options
```

---

### `grab_data.sh`

Sync large output data (chains, grids) from an HPC cluster via rsync. Reads a `sync.conf` from the project directory.

```bash
bash grab_data.sh <project-dir>   # reads <project-dir>/sync.conf
bash grab_data.sh                 # reads ./sync.conf
```

**`sync.conf` format:**

```bash
REMOTE=user@hostname:/remote/path
LOCAL=local/destination           # relative to sync.conf location
DIRS=(
  subdir1
  subdir2
)
```
