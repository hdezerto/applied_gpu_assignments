#include <cuda_runtime.h> // CUDA runtime API
#include <stdio.h>        // Standard I/O
#include <stdlib.h>       // Standard library
#include <sys/time.h>     // For timing on CPU
#include <cmath>          // Math functions
#include <random>         // Random number generation
#include <string.h>       // String utilities for CLI parsing
#include <limits>         // Guard against overflow in sweep mode

#ifndef OPT_TWO_LOADS
#define OPT_TWO_LOADS 1 // Load two elements per thread for higher arithmetic intensity
#endif

#ifndef OPT_SHARED_REDUCTION
#define OPT_SHARED_REDUCTION 1 // Use shared-memory tree to reduce per block
#endif

#ifndef OPT_BLOCK_ATOMIC
#define OPT_BLOCK_ATOMIC 1 // If shared reduction is enabled, emit one atomicAdd per block
#endif

#define CUDA_CHECK(call)                                                                        \
  do {                                                                                          \
    cudaError_t _err = (call);                                                                  \
    if (_err != cudaSuccess) {                                                                  \
      /* Print error message with file and line number, then exit */                            \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
      exit(EXIT_FAILURE);                                                                       \
    }                                                                                           \
  } while (0)



// Returns current time in milliseconds
static double get_time_ms() {
  timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<double>(tv.tv_sec) * 1000.0 + static_cast<double>(tv.tv_usec) / 1000.0;
}


// CUDA kernel for parallel reduction (sum) using shared memory
__global__ void reduction_kernel(const float *input,
                                 float *result,
                                 float *blockPartials,
                                 int length) {
#if OPT_SHARED_REDUCTION
  extern __shared__ float shared[]; // Shared memory for block
#endif

  const unsigned int tid = threadIdx.x; // Thread index within block
  const unsigned int elementsPerThread = OPT_TWO_LOADS ? 2 : 1;
  const unsigned int globalIdx = blockIdx.x * blockDim.x * elementsPerThread + threadIdx.x;

  float localSum = 0.0f;
  // Load first element if in bounds
  if (globalIdx < length) {
    localSum += input[globalIdx];
  }
  // Load second element if enabled and in bounds
#if OPT_TWO_LOADS
  if (globalIdx + blockDim.x < length) {
    localSum += input[globalIdx + blockDim.x];
  }
#endif

#if OPT_SHARED_REDUCTION
  // Store local sum in shared memory
  shared[tid] = localSum;
  __syncthreads();

  // Perform reduction in shared memory
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  // Write block's result to global memory using atomic add
  if (tid == 0) {
    const float blockSum = shared[0];
#if OPT_BLOCK_ATOMIC
    atomicAdd(result, blockSum);
#else
    if (blockPartials) {
      blockPartials[blockIdx.x] = blockSum;
    } else {
      atomicAdd(result, blockSum); // Fallback if partial buffer not provided
    }
#endif
  }

#else // OPT_SHARED_REDUCTION == 0
  // Without shared-memory tree we simply atomically accumulate each thread's partial sum
  atomicAdd(result, localSum);
#endif
}



// Simple CPU implementation of reduction (sum)
static float cpu_reduction(const float *data, int length) {
  double sum = 0.0;
  for (int i = 0; i < length; ++i) {
    sum += static_cast<double>(data[i]);
  }
  return static_cast<float>(sum);
}



struct RunStats {
  double cpuMs;
  double gpuMs;
  float absDiff;
  float relDiff;
  float cpuSum;
  float gpuSum;
};

static int run_reduction_once(int inputLength, RunStats *stats, bool verbose) {
  if (inputLength <= 0) {
    fprintf(stderr, "Input length must be positive.\n");
    return EXIT_FAILURE;
  }

  if (verbose) {
    printf("The input length is %d\n", inputLength);
    printf("Optimizations -> two-loads:%d shared:%d block-atomic:%d\n",
           OPT_TWO_LOADS,
           OPT_SHARED_REDUCTION,
           OPT_BLOCK_ATOMIC);
  }

  const size_t bytes = static_cast<size_t>(inputLength) * sizeof(float);
  float *h_input = static_cast<float *>(malloc(bytes));
  float *h_gpuResult = static_cast<float *>(malloc(sizeof(float)));
  if (!h_input || !h_gpuResult) {
    fprintf(stderr, "Failed to allocate host memory.\n");
    free(h_input);
    free(h_gpuResult);
    return EXIT_FAILURE;
  }

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (int i = 0; i < inputLength; ++i) {
    h_input[i] = dist(rng);
  }

  const double cpuStart = get_time_ms();
  const float cpuResult = cpu_reduction(h_input, inputLength);
  const double cpuElapsed = get_time_ms() - cpuStart;

  float *d_input = nullptr;
  float *d_result = nullptr;
  float *d_blockPartials = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_result, 0, sizeof(float)));

  const int threadsPerBlock = 256;
  const int elementsPerThread = OPT_TWO_LOADS ? 2 : 1;
  const int itemsPerBlock = threadsPerBlock * elementsPerThread;
  int blocksPerGrid = (inputLength + itemsPerBlock - 1) / itemsPerBlock;
  if (blocksPerGrid == 0) {
    blocksPerGrid = 1;
  }
  const bool useShared = OPT_SHARED_REDUCTION != 0;
  const bool useBlockAtomic = useShared && (OPT_BLOCK_ATOMIC != 0);
  const bool needBlockPartials = useShared && !useBlockAtomic;
  const size_t sharedBytes = useShared ? threadsPerBlock * sizeof(float) : 0;

  if (needBlockPartials) {
    CUDA_CHECK(cudaMalloc(&d_blockPartials, static_cast<size_t>(blocksPerGrid) * sizeof(float)));
  }

  double gpuStart = get_time_ms();
  reduction_kernel<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(d_input,
                                                                    d_result,
                                                                    d_blockPartials,
                                                                    inputLength);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  const double gpuElapsed = get_time_ms() - gpuStart;

  float gpuResult = 0.0f;
  if (needBlockPartials) {
    float *h_blockPartials = static_cast<float *>(malloc(static_cast<size_t>(blocksPerGrid) * sizeof(float)));
    if (!h_blockPartials) {
      fprintf(stderr, "Failed to allocate host buffer for block partials.\n");
      CUDA_CHECK(cudaFree(d_blockPartials));
      CUDA_CHECK(cudaFree(d_input));
      CUDA_CHECK(cudaFree(d_result));
      free(h_input);
      free(h_gpuResult);
      return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaMemcpy(h_blockPartials,
                          d_blockPartials,
                          static_cast<size_t>(blocksPerGrid) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    gpuResult = cpu_reduction(h_blockPartials, blocksPerGrid);
    free(h_blockPartials);
  } else {
    CUDA_CHECK(cudaMemcpy(h_gpuResult, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    gpuResult = *h_gpuResult;
  }

  const float absDiff = std::fabs(cpuResult - gpuResult);
  const float relDiff = absDiff / (std::fabs(cpuResult) + 1e-12f);

  if (verbose) {
    printf("CPU sum: %.6f (%.3f ms)\n", cpuResult, cpuElapsed);
    printf("GPU sum: %.6f (%.3f ms)\n", gpuResult, gpuElapsed);
    printf("Absolute difference: %.6e | Relative difference: %.6e\n", absDiff, relDiff);
  }

  if (d_blockPartials) {
    CUDA_CHECK(cudaFree(d_blockPartials));
  }
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_result));
  free(h_input);
  free(h_gpuResult);

  if (stats) {
    stats->cpuMs = cpuElapsed;
    stats->gpuMs = gpuElapsed;
    stats->absDiff = absDiff;
    stats->relDiff = relDiff;
    stats->cpuSum = cpuResult;
    stats->gpuSum = gpuResult;
  }

  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  const int defaultLength = 1 << 20;
  bool sweepMode = false;
  int sweepStart = 512;
  int sweepRuns = 9;
  int inputLength = defaultLength;

  if (argc >= 2 && strcmp(argv[1], "--sweep") == 0) {
    sweepMode = true;
    if (argc >= 3) {
      sweepStart = atoi(argv[2]);
    }
    if (argc >= 4) {
      sweepRuns = atoi(argv[3]);
    }
  } else if (argc >= 2) {
    inputLength = atoi(argv[1]);
  }

  if (sweepMode) {
    if (sweepStart <= 0 || sweepRuns <= 0) {
      fprintf(stderr, "Sweep arguments must be positive integers.\n");
      return EXIT_FAILURE;
    }
    printf("length,cpu_ms,gpu_ms,abs_diff,rel_diff\n");
    int currentLength = sweepStart;
    for (int i = 0; i < sweepRuns; ++i) {
      RunStats stats{};
      if (run_reduction_once(currentLength, &stats, false) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
      }
      printf("%d,%.6f,%.6f,%.6e,%.6e\n",
             currentLength,
             stats.cpuMs,
             stats.gpuMs,
             stats.absDiff,
             stats.relDiff);
      if (currentLength > std::numeric_limits<int>::max() / 2) {
        break; // Prevent overflow
      }
      currentLength *= 2;
    }
    return EXIT_SUCCESS;
  }

  RunStats stats{};
  if (run_reduction_once(inputLength, &stats, true) != EXIT_SUCCESS) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

