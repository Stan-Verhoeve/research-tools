# research-tools

General-purpose scripts for research workflows.

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
