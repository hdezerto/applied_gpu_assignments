Q2 – Parallel Reduction (CUDA)
------------------------------

Files
-----
- `reduction.cu` – configurable CUDA reduction kernel with CPU reference + timing.
- `Makefile` – builds five variants via NVCC (`sm_75` by default).
- `plot_timing.py` – reads `reduction_timing.csv` and plots CPU vs GPU timing (pandas + matplotlib required).

Build
-----
```bash
make            # build every variant below
make TARGET     # build a single variant, e.g. TARGET=reduction_shared
```

Run
---
Every executable accepts an optional array length (default: `2^20 = 1048576`). Examples:
```bash
./reduction_base 1048576      # explicit size
./reduction_shared            # uses default size (1048576)
```
Variants:
- `reduction_base`: pure global atomics baseline.
- `reduction_twoload`: baseline plus two elements per thread.
- `reduction_shared`: shared-memory tree per block, host finalizes sum.
- `reduction_shared_atomic`: shared tree plus one `atomicAdd` per block.
- `reduction_full`: default build with all optimizations enabled.

Output
------
Each run prints CPU vs GPU sums, elapsed time, and error metrics; sweep mode emits CSV lines compatible with `plot_timing.py`.

Sweep mode
----------
Generate CSV-friendly timings by doubling lengths from a starting value:
```bash
./reduction_full --sweep 512 9 > reduction_timing.csv
```
The CSV header matches the format expected by `plot_timing.py`, which produces a CPU/GPU bar chart.
