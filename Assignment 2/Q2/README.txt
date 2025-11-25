Q2 – Parallel Reduction in CUDA
------------------------------

Files:
- reduction.cu : CUDA implementation of parallel reduction (sum) with timing and result comparison.
- Makefile     : builds multiple optimization variants for benchmarking.

Build
-----
```bash
make            # builds every variant listed below
make TARGET     # builds a single variant
```

Run
Each binary accepts an optional array length (`2^20` default). Examples:
```bash
./reduction_base 1048576
./reduction_shared      # uses default length
```
Sweep mode
----------
Generate CSV-friendly timings by doubling lengths from a starting value:
```bash
./reduction_full --sweep 512 9 > reduction_timing.csv
```
The output header matches the format expected by a plotting script.

Variants
--------
- `reduction_base` (`OPT_TWO_LOADS=0`, `OPT_SHARED_REDUCTION=0`, `OPT_BLOCK_ATOMIC=0`): pure global atomics, used as the timing baseline.
- `reduction_twoload` (`OPT_TWO_LOADS=1`): baseline plus two elements per thread.
- `reduction_shared` (`OPT_SHARED_REDUCTION=1`, `OPT_BLOCK_ATOMIC=0`): shared-memory tree per block, host finalizes the partial sums.
- `reduction_shared_atomic` (`OPT_SHARED_REDUCTION=1`, `OPT_BLOCK_ATOMIC=1`): shared memory plus one `atomicAdd` per block.
- `reduction_full`: default build, all optimizations enabled.

Output
------
- Prints CPU and GPU reduction results, timing, and error metrics to the console.
