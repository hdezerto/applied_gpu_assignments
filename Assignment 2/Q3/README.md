Q3 – Tiled Matrix Multiplication (CUDA)
--------------------------------------

### Files
- `matrixMul.cu` – baseline GEMM plus tiled shared-memory kernel, CPU reference, sweep+CSV logging.
- `Makefile` – builds `matrixMul` with NVCC (default `sm_75`).
- `plot_timings.py` – reads the CSV emitted by `matrixMul --sweep` and plots runtimes (matplotlib + numpy required).

### Build
```bash
make        # build matrixMul
make run    # build (if needed) and run with defaults
```

### Run a single configuration
Tile sizes remain hard-coded in the `tiles` vector; pass matrix dimensions as arguments:
```bash
./matrixMul --m 1024 --k 1024 --n 1024 --no-print
./matrixMul --m 513 --k 8192 --n 1023
```
Options:
- `--m <rowsA>` – rows of A (and C)
- `--k <shared>` – columns of A / rows of B
- `--n <colsB>` – columns of B (and C)
- `--sweep` – ignore `--m/k/n` and run the predefined list of ~10 square problems, logging to `matrixMul_sweep.csv`
- `--print` / `--no-print` – show or hide compact 3×3 previews of each matrix snapshot
- `-h`, `--help` – print the option summary

To test other tile sizes, edit the `tiles` vector near the top of `main()` and rebuild.

### Program output
Each run:
1. Initializes matrices, prints the CPU reference preview, and reports CPU timing.
2. Launches the baseline `gemm` kernel and prints its timing/max error.
3. Launches every `[tileX, tileY]` pair defined in `tiles`, reporting timing and validation error.
4. Skips tiles that exceed the GPU’s thread or shared-memory limits (message shown).

### CSV + plotting workflow
1. Collect the built-in sweep timings (logged automatically to `matrixMul_sweep.csv`):
   ```bash
   ./matrixMul --sweep --no-print
   ```
2. Plot CPU vs GPU timings (log-scale y-axis) as grouped bars:
   ```bash
   python plot_timings.py matrixMul_sweep.csv --output sweep.png
   ```
   The plot displays CPU, baseline GEMM, and each tiled kernel for every matrix size in the sweep.
