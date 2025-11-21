Q3 - 2D Dense Matrix Multiplication
===================================

This program multiplies two 2D matrices (A and B) on the GPU using CUDA and stores the result in matrix C. It benchmarks memory transfer and kernel execution times for multiple matrix sizes, and outputs results to a CSV file for plotting.

Files:
- matrixMul.cu: Main CUDA C++ source file (uses typedef T for float/double precision).
- plot_timing.py: Python script to plot timing breakdown from CSV output.
- Makefile: For building the program with `make`.

Usage:
1. Build the program:
   make
2. Run the program:
   ./matrixMul           # For float (default)
   ./matrixMul --no-error  # Skip error check for faster timing
   # To use double, change 'typedef float T;' to 'typedef double T;' in matrixMul.cu and rebuild

Output:
- Results are saved to timing_results.csv
- Each row: NARows, NBColumns, HostToDevice_ms, Kernel_ms, DeviceToHost_ms, MaxError

Plotting:
Run the Python script to generate a stacked bar chart:
   python plot_timing.py

Notes:
- Matrix sizes are set in the source code (test_sizes array)
- The program checks the result against a CPU reference unless --no-error is used
- For very large matrices, ensure your GPU has enough memory
