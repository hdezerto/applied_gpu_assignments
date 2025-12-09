## Q2 – Vector Addition with Streams

### Files
- `vectorAdd.cu`: CUDA vector add with optional 4-stream segmented execution, GPU timing, sweeps.
- `plot_timing.py`: Plots Q1 (streamed vs baseline) and Q3 (segment size impact) from `timing_results.csv`.
- `Makefile`: build and helper run targets.

### Build
- `make` (adjust `ARCH` in the Makefile if needed)

### Run Modes
- Baseline (no streams, default N): `./vectorAdd`
- Baseline sweep over N (powers of two): `./vectorAdd --sweepN`
- Streamed (4 streams, default N and S_seg): `./vectorAdd --streamed`
- Streamed sweep over N (Q1 comparison): `./vectorAdd --streamed --sweepN`
- Streamed sweep over segment sizes (Q3; uses max N=1,048,576 unless `--sweepN` also set): `./vectorAdd --streamed --sweepSseg`

### Output
- CSV `timing_results.csv` with columns: `N, GPUElapsed_ms, MaxError, Streamed (0|1), S_seg` (GPU time via CUDA events over H2D + kernel + D2H).
- Stdout mirrors the CSV per run.

### Plotting (Q1 and Q3)
- Both plots from default CSV: `python plot_timing.py`
- Only Q1 (baseline vs streamed): `python plot_timing.py --q1`
- Only Q3 (segment size impact): `python plot_timing.py --q3`
- Use a different CSV: `python plot_timing.py --csv my_results.csv --q1`

### Generated Plots
- `vectoradd_stream_vs_baseline.png`: GPU time vs N for baseline vs streamed (Q1 performance gain).
- `vectoradd_segment_size.png`: GPU time vs segment size at the largest N present (Q3 impact of S_seg).
