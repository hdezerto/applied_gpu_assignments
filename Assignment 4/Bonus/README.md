# Assignment 4 - Bonus: WMMA Tensor Core GEMM

## Files
- `wmma_bonus.cu`: CPU reference (double), baseline GEMM (float), tiled GEMM, and WMMA Tensor Core GEMM (FP16 inputs, FP32 accumulate) with timing and CSV logging.
- `Makefile`: builds `wmma_bonus` (defaults: `ARCH=sm_75`, `-O3`).
- `plot_bonus.py`: plots CPU vs GEMM vs WMMA runtimes from the sweep CSV (log-scale y-axis).

## Build
```bash
make ARCH=sm_75   # adjust arch for your GPU
```

## Run
Single case (defaults: m=k=n=256, prints small matrix preview):
```bash
./wmma_bonus --m 512 --k 512 --n 512 --block-warps-m 2 --block-warps-n 2 --iters 5
```

Preset sweep (512, 1024, 2048, 4096; writes `wmma_bonus_sweep.csv`):
```bash
./wmma_bonus --sweep --no-print --iters 5
```
Flags of interest:
- `--block-warps-m`, `--block-warps-n`: warp grid per block for WMMA (tile 16x16x16).
- `--iters`: timing iterations for WMMA averaging.
- `--print/--no-print`: enable or suppress matrix previews.

## Plot
```bash
python plot_bonus.py --out bonus_runtime.png wmma_bonus_sweep.csv
```
Produces a log-scale runtime bar chart for CPU, baseline GEMM, and WMMA.
