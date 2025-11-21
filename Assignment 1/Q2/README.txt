Q2 – Vector Addition in CUDA
----------------------------

Files:
- vectorAdd.cu : CUDA implementation with sweep mode, timing, and CSV output.
- plot_timing.py : Python script to plot timing breakdown from CSV output.
- Makefile     : to compile and run the code.

How to compile:
    make

How to run:
    ./vectorAdd           # Runs a sweep of N values (default)
    ./vectorAdd <N>       # Runs for a single vector length N

Output:
- Results are saved to timing_results.csv
- Each row: N, HostToDevice_ms, Kernel_ms, DeviceToHost_ms, MaxError

Plotting:
Run the Python script to generate a stacked bar chart with total time labels:
    python plot_timing.py

Description:
This program allocates host and device memory, transfers data to the GPU,
launches a kernel that adds two vectors, copies the result back to the CPU,
compares it with a CPU reference implementation, and prints the maximum error.
It also measures and records the timing for each major step and supports batch runs for multiple vector sizes.
