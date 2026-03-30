# research-tools

General-purpose scripts for research workflows.

## Contents

### `plot_chain.py`

Load and visualize MCMC chains from [Cobaya](https://cobaya.readthedocs.io) runs. Produces triangle plots and projection plots via [GetDist](https://getdist.readthedocs.io). Supports derived parameter computation, parameter renaming, and multi-chain comparisons.

**Dependencies:** `getdist`, `matplotlib`, `numpy`

```bash
# Compare two chains, plot modified gravity parameters
python plot_chain.py -b /path/to/chains -c ODE_P18 ODE_P18+FS+BAO -l "PR4" "PR4+FS+BAO" -p alphaB alphaM

# Full parameter projection plot
python plot_chain.py -b /path/to/chains -c ODE_P18 ODE_P18+FS+BAO --projection

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
