# Assignment 4 - Q2: Heat Equation (cuBLAS, cuSPARSE, Unified Memory)

### Files
- `hw4-heat.cu`: Heat equation solver using cuSPARSE SpMV and cuBLAS AXPY/NRM2 with Unified Memory; optional prefetch toggle via argv[3].
- `Makefile`: build and run helpers (`ARCH` selectable, default `sm_75`).
- `run_error_sweep.py`: Sweeps `nsteps` for a fixed `dimX` (default 1024), writes `error_sweep.csv` and `error_sweep.png`.
- `run_prefetch_compare.py`: Runs multiple trials with/without prefetch for selected `dimX` values, writes `prefetch_compare.csv`.

### Build
```bash
make ARCH=-arch=sm_75   # adjust arch for your GPU
```

### Run
```bash
./heat <dimX> <nsteps> [prefetch]
# prefetch: 1 (default) to enable UM prefetch, 0 to disable
```
Examples:
- Prefetch on: `./heat 1024 2000 1`
- Prefetch off: `./heat 1024 2000 0`
- Using Makefile helper: `make run DIMX=1024 NSTEPS=2000 PREFETCH=1`

### Scripts
- Relative error sweep (Q2.2):
  ```bash
  python run_error_sweep.py   # outputs error_sweep.csv, error_sweep.png
  ```
- Prefetch comparison with averaging (Q2.3):
  ```bash
  python run_prefetch_compare.py   # outputs prefetch_compare.csv
  ```
  Edit `DIMX_LIST`, `NSTEPS`, `REPEATS` in the script as needed.