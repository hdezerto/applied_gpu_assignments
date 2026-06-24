# Applied GPU Programming Assignments

Coursework for **DD2360 Applied GPU Programming** at KTH Royal Institute of Technology.

This repository collects CUDA assignments covering basic kernels, memory transfers, shared memory, reductions, streams, cuBLAS/cuSPARSE, Unified Memory, profiling, and Tensor Core programming. Each question folder contains its own source code, `Makefile`, and short local README where relevant.

## Contents

| Path | Topic |
| --- | --- |
| `Assignment 1` | CUDA basics, vector addition, matrix multiplication, and Rodinia benchmark profiling. |
| `Assignment 2` | Histograms, atomics/shared memory, parallel reduction, and tiled matrix multiplication. |
| `Assignment 3` | CUDA streams and an iPIC3D-mini GPU mover bonus task. |
| `Assignment 4` | Convolution, cuBLAS/cuSPARSE heat equation solver, Unified Memory, and WMMA Tensor Cores. |

## Selected Takeaways

- Reduction: combined two-loads, shared memory, and block-level atomics reached about `0.176 ms` for `N = 2^20`.
- Streams: segmented four-stream vector addition improved large-vector runtime by up to about `30%`.
- iPIC3D-mini: GPU particle mover achieved about `12x` mover speedup and about `2x` end-to-end speedup.
- WMMA: Tensor Core GEMM was about `5x-11x` faster than the FP32 GPU GEMM baseline, with the expected FP16 accuracy trade-off.

## Running

Most folders build from inside the question directory:

```bash
cd "Assignment 2/Q3"
make
./matrixMul --sweep --no-print
```

The exact commands differ by assignment; see the local README in each folder.

## Notes

The assignments target a Linux CUDA environment with an NVIDIA GPU. Generated binaries, profiling files, CSVs, plots, and benchmark outputs are ignored going forward, though some historical artifacts remain from the original coursework snapshot.
