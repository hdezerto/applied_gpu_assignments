#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err), __FILE__, __LINE__);       \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

using DataType = float;

//------------------------------------------------------------------------------
// Host utilities
//------------------------------------------------------------------------------

void matMulCPU(const DataType *A, const DataType *B, DataType *C, int numARows,
               int numAColumns, int numBColumns) {
    // Loop over each row of A
    for (int row = 0; row < numARows; ++row) {
        // Loop over each column of B
        for (int col = 0; col < numBColumns; ++col) {
            DataType acc = 0.0f; // Accumulator for the dot product
            // Compute the dot product of row 'row' of A and column 'col' of B
            for (int k = 0; k < numAColumns; ++k) {
                // A[row, k] * B[k, col]
                acc += A[row * numAColumns + k] * B[k * numBColumns + col];
            }
            // Store the result in C[row, col]
            C[row * numBColumns + col] = acc;
        }
    }
}

void initializeMatrix(DataType *mat, int rows, int cols, DataType scale) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Compute the linear index for row-major storage
            const int idx = r * cols + c;
            // Generate a value that smoothly varies with idx
            // - scale * sin(0.01 * idx): sine component
            // - (1 - scale) * cos(0.005 * idx): cosine component
            // The scale parameter blends between the two functions
            mat[idx] = scale * std::sin(0.01f * static_cast<float>(idx)) +
                       (1.0f - scale) * std::cos(0.005f * static_cast<float>(idx));
        }
    }
}

double maxAbsDiff(const DataType *lhs, const DataType *rhs, std::size_t count) {
    double max_err = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        max_err = std::max<double>(max_err, std::fabs(lhs[i] - rhs[i]));
    }
    return max_err;
}

void printMatrix(const DataType *mat, int rows, int cols, const char *label,
                 bool enabled, int maxRows = 3, int maxCols = 3) {
    if (!enabled) {
        return;
    }
    std::printf("%s (%dx%d):\n", label, rows, cols);
    const int showRows = std::min(rows, maxRows);
    const int showCols = std::min(cols, maxCols);
    for (int r = 0; r < showRows; ++r) {
        for (int c = 0; c < showCols; ++c) {
            std::printf("%8.3f ", mat[r * cols + c]);
        }
        if (showCols < cols) {
            std::printf("...");
        }
        std::printf("\n");
    }
    if (showRows < rows) {
        std::printf("...\n");
    }
    std::printf("\n");
}

//------------------------------------------------------------------------------
// Argument parsing (matrix sizes, print flag)
//------------------------------------------------------------------------------

struct SizeSpec {
    int m;
    int k;
    int n;
};

struct Options {
    int m = 256; // rows of A / C
    int k = 256; // columns of A / rows of B
    int n = 256; // columns of B / C
    bool printMatrices = true;
    bool sweep = false; // run a predefined sweep instead of single size
};

const std::vector<SizeSpec> kSweepSizes = {
    {64, 64, 64},   {96, 96, 96},   {128, 128, 128}, {192, 192, 192},
    {256, 256, 256},{384, 384, 384},{512, 512, 512}, {768, 768, 768},
    {1024, 1024, 1024}, {1536, 1536, 1536}
};

constexpr const char *kSweepCsvPath = "matrixMul_sweep.csv";

void printUsage(const char *prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --m <rowsA>     Rows of A / C (default 256)\n"
        "  --k <shared>    Columns of A / rows of B (default 256)\n"
        "  --n <colsB>     Columns of B / C (default 256)\n"
    "  --sweep         Run the predefined size sweep and log to CSV\n"
        "  --print         Enable compact matrix previews (default)\n"
        "  --no-print      Disable matrix previews\n"
        "  -h, --help      Show this message\n",
        prog);
}

Options parseArgs(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--print") {
            opts.printMatrices = true;
        } else if (arg == "--no-print") {
            opts.printMatrices = false;
        } else if ((arg == "--m" || arg == "-m") && i + 1 < argc) {
            opts.m = std::atoi(argv[++i]);
        } else if ((arg == "--k" || arg == "-k") && i + 1 < argc) {
            opts.k = std::atoi(argv[++i]);
        } else if ((arg == "--n" || arg == "-n") && i + 1 < argc) {
            opts.n = std::atoi(argv[++i]);
        } else if (arg == "--sweep") {
            opts.sweep = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return opts;
}

void appendCsvRow(const std::string &path, bool &headerWritten,
                  const SizeSpec &size, double cpuMs, float gemmMs,
                  const std::vector<float> &tiledTimes,
                  const std::vector<std::pair<int, int>> &tiles) {
    if (path.empty()) {
        return;
    }
    std::ios_base::openmode mode = headerWritten ? std::ios::app : std::ios::out;
    std::ofstream file(path, mode);
    if (!file) {
        std::fprintf(stderr, "Could not open CSV file %s\n", path.c_str());
        return;
    }
    if (!headerWritten) {
        file << "m,k,n,cpu_ms,gemm_ms";
        for (const auto &tile : tiles) {
            file << ",tiled_" << tile.first << 'x' << tile.second << "_ms";
        }
        file << '\n';
        headerWritten = true;
    }
    file << size.m << ',' << size.k << ',' << size.n << ',' << cpuMs << ','
         << gemmMs;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        if (i < tiledTimes.size()) {
            file << ',' << tiledTimes[i];
        } else {
            file << ',';
        }
    }
    file << '\n';
}

//------------------------------------------------------------------------------
// CUDA kernels
//------------------------------------------------------------------------------

__global__ void gemm(const DataType *A, const DataType *B, DataType *C,
                     int numARows, int numAColumns, int numBColumns) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= numARows || col >= numBColumns) {
        return;
    }

    DataType acc = 0.0f;
    for (int k = 0; k < numAColumns; ++k) {
        acc += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numBColumns + col] = acc;
}


__global__ void tiled_gemm(const DataType *A, const DataType *B, DataType *C,
                           int numARows, int numAColumns, int numBColumns,
                           int tileX, int tileY) {
    // Allocate shared memory for tiles of A and B
    extern __shared__ DataType shared[]; // Shared memory size set at kernel launch
    DataType *tileA = shared; // Pointer to tile for matrix A
    DataType *tileB = tileA + tileX * tileY; // Pointer to tile for matrix B (placed after tileA)

    // Compute the global row and column indices for this thread in output matrix C
    const int row = blockIdx.y * tileY + threadIdx.y; // Global row index in C
    const int col = blockIdx.x * tileX + threadIdx.x; // Global column index in C

    // Thread info for parallel tile loading
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearTid = threadIdx.y * blockDim.x + threadIdx.x; // Linear thread ID within the block

    DataType acc = 0.0f; // Accumulator for the output value

    // Loop over tiles along the shared dimension (A's columns / B's rows)
    for (int tileIdx = 0; tileIdx < numAColumns; tileIdx += tileX) {
        // --- Load a tile of A into shared memory ---
        // Each thread loads multiple elements in parallel
        for (int idx = linearTid; idx < tileY * tileX; idx += threadsPerBlock) {
            const int localRow = idx / tileX; // Row within the tile
            const int localCol = idx % tileX; // Col within the tile
            const int globalRow = blockIdx.y * tileY + localRow; // Global row in A
            const int globalCol = tileIdx + localCol; // Global col in A
            // Bounds check for edge tiles
            if (globalRow < numARows && globalCol < numAColumns) {
                tileA[idx] = A[globalRow * numAColumns + globalCol];
            } else {
                tileA[idx] = 0.0f;
            }
        }

        // --- Load a tile of B into shared memory ---
        // Each thread loads multiple elements in parallel
        for (int idx = linearTid; idx < tileX * tileX; idx += threadsPerBlock) {
            const int localRow = idx / tileX; // Row within the tile
            const int localCol = idx % tileX; // Col within the tile
            const int globalRow = tileIdx + localRow; // Global row in B
            const int globalCol = blockIdx.x * tileX + localCol; // Global col in B
            // Bounds check for edge tiles
            if (globalRow < numAColumns && globalCol < numBColumns) {
                tileB[idx] = B[globalRow * numBColumns + globalCol];
            } else {
                tileB[idx] = 0.0f;
            }
        }
        // Synchronize to make sure all threads have loaded their tile data
        __syncthreads();

        // --- Compute partial dot product for this tile ---
        // Only iterate up to the valid number of elements in the tile (for edge tiles)
        const int maxK = min(tileX, numAColumns - tileIdx);
        for (int k = 0; k < maxK; ++k) {
            // Each thread computes one output element for C
            acc += tileA[threadIdx.y * tileX + k] *
                   tileB[k * tileX + threadIdx.x];
        }
        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = acc;
    }
}


//------------------------------------------------------------------------------
// Kernel launch helpers
//------------------------------------------------------------------------------

float launchBaselineKernel(const DataType *d_A, const DataType *d_B,
                           DataType *d_C, int numARows, int numAColumns,
                           int numBColumns) {
    dim3 block(16, 16);
    dim3 grid((numBColumns + block.x - 1) / block.x,
              (numARows + block.y - 1) / block.y);

    cudaEvent_t start{}, stop{};
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    gemm<<<grid, block>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms;
}

bool launchTiledKernel(const DataType *d_A, const DataType *d_B, DataType *d_C,
                       int numARows, int numAColumns, int numBColumns,
                       int tileX, int tileY, const cudaDeviceProp &prop,
                       float &timingMs) {
    if (tileX <= 0 || tileY <= 0) {
        std::fprintf(stderr, "Skipping tile [%d, %d]: dimensions must be > 0\n",
                     tileX, tileY);
        return false;
    }
    if (tileX > prop.maxThreadsDim[0] || tileY > prop.maxThreadsDim[1]) {
        std::fprintf(stderr,
                     "Skipping tile [%d, %d]: exceeds per-dimension thread limit\n",
                     tileX, tileY);
        return false;
    }
    if (tileX * tileY > prop.maxThreadsPerBlock) {
        std::fprintf(stderr,
                     "Skipping tile [%d, %d]: exceeds %d threads per block\n",
                     tileX, tileY, prop.maxThreadsPerBlock);
        return false;
    }

    const std::size_t sharedBytes =
        static_cast<std::size_t>(tileX * tileY + tileX * tileX) * sizeof(DataType);
    if (sharedBytes > prop.sharedMemPerBlock) {
        std::fprintf(stderr,
                     "Skipping tile [%d, %d]: needs %zu B shared (limit %zu B)\n",
                     tileX, tileY, sharedBytes,
                     static_cast<std::size_t>(prop.sharedMemPerBlock));
        return false;
    }

    dim3 block(tileX, tileY);
    dim3 grid((numBColumns + tileX - 1) / tileX,
              (numARows + tileY - 1) / tileY);

    cudaEvent_t start{}, stop{};
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    tiled_gemm<<<grid, block, sharedBytes>>>(d_A, d_B, d_C, numARows,
                                             numAColumns, numBColumns, tileX,
                                             tileY);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    CHECK(cudaEventElapsedTime(&timingMs, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return true;
}

void runCase(const SizeSpec &size, bool printMatrices,
             const std::vector<std::pair<int, int>> &tiles,
             const cudaDeviceProp &prop, const std::string &csvPath,
             bool &csvHeaderWritten) {
    const int numARows = size.m;
    const int numAColumns = size.k;
    const int numBRows = size.k;
    const int numBColumns = size.n;
    const int numCRows = numARows;
    const int numCColumns = numBColumns;

    std::printf("\n==== GEMM case: A=%dx%d, B=%dx%d ====\n", numARows,
                numAColumns, numBRows, numBColumns);

    const std::size_t elementsA = static_cast<std::size_t>(numARows) * numAColumns;
    const std::size_t elementsB = static_cast<std::size_t>(numBRows) * numBColumns;
    const std::size_t elementsC = static_cast<std::size_t>(numCRows) * numCColumns;

    DataType *h_A = static_cast<DataType *>(std::malloc(elementsA * sizeof(DataType)));
    DataType *h_B = static_cast<DataType *>(std::malloc(elementsB * sizeof(DataType)));
    DataType *h_C = static_cast<DataType *>(std::malloc(elementsC * sizeof(DataType)));
    DataType *h_ref = static_cast<DataType *>(std::malloc(elementsC * sizeof(DataType)));
    if (!h_A || !h_B || !h_C || !h_ref) {
        std::fprintf(stderr, "Host memory allocation failed for size %dx%dx%d.\n",
                     numARows, numAColumns, numBColumns);
        std::free(h_A);
        std::free(h_B);
        std::free(h_C);
        std::free(h_ref);
        return;
    }

    initializeMatrix(h_A, numARows, numAColumns, 0.7f);
    initializeMatrix(h_B, numBRows, numBColumns, 0.3f);

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    matMulCPU(h_A, h_B, h_ref, numARows, numAColumns, numBColumns);
    const auto cpuStop = std::chrono::high_resolution_clock::now();
    const double cpuMs =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    DataType *d_A = nullptr;
    DataType *d_B = nullptr;
    DataType *d_C = nullptr;
    const std::size_t bytesA = elementsA * sizeof(DataType);
    const std::size_t bytesB = elementsB * sizeof(DataType);
    const std::size_t bytesC = elementsC * sizeof(DataType);

    CHECK(cudaMalloc(&d_A, bytesA));
    CHECK(cudaMalloc(&d_B, bytesB));
    CHECK(cudaMalloc(&d_C, bytesC));

    CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));

    std::printf("Input matrix dim: A=%dx%d  B=%dx%d\n", numARows,
                numAColumns, numBRows, numBColumns);
    std::printf("CPU reference time: %.3f ms\n", cpuMs);
    printMatrix(h_ref, numCRows, numCColumns, "CPU reference result:",
                printMatrices);

    CHECK(cudaMemset(d_C, 0, bytesC));
    const float gemmMs = launchBaselineKernel(d_A, d_B, d_C, numARows,
                                              numAColumns, numBColumns);
    CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    const double gemmError = maxAbsDiff(h_C, h_ref, elementsC);

    std::printf("CUDA gemm result:\n");
    printMatrix(h_C, numCRows, numCColumns, "CUDA gemm result:",
                printMatrices);
    std::printf("timing: %.3f ms\n", gemmMs);
    std::printf("max error vs CPU: %.3e\n\n", gemmError);

    std::vector<float> tiledTimings;
    std::vector<double> tiledErrors;
    tiledTimings.reserve(tiles.size());
    tiledErrors.reserve(tiles.size());

    for (const auto &tile : tiles) {
        const int tileX = tile.first;
        const int tileY = tile.second;
        CHECK(cudaMemset(d_C, 0, bytesC));
        float tiledMs = 0.0f;
        if (!launchTiledKernel(d_A, d_B, d_C, numARows, numAColumns,
                               numBColumns, tileX, tileY, prop, tiledMs)) {
            tiledTimings.push_back(std::numeric_limits<float>::quiet_NaN());
            tiledErrors.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
        const double tiledError = maxAbsDiff(h_C, h_ref, elementsC);
        std::printf("CUDA tiled_gemm with tile [%d, %d] result:\n", tileX,
                    tileY);
        printMatrix(h_C, numCRows, numCColumns,
                    "CUDA tiled_gemm result:", printMatrices);
        std::printf("timing: %.3f ms\n", tiledMs);
        std::printf("max error vs CPU: %.3e\n\n", tiledError);
        tiledTimings.push_back(tiledMs);
        tiledErrors.push_back(tiledError);
    }

    appendCsvRow(csvPath, csvHeaderWritten, size, cpuMs, gemmMs,
                 tiledTimings, tiles);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_ref);
}

int main(int argc, char **argv) {
    const Options opts = parseArgs(argc, argv);
    const bool printMatrices = opts.printMatrices;
    const std::vector<std::pair<int, int>> tiles = {
        {8, 8}, {16, 16}, {32, 32}
    };

    std::vector<SizeSpec> runs;
    if (opts.sweep) {
        runs = kSweepSizes;
    } else {
        runs.push_back(SizeSpec{opts.m, opts.k, opts.n});
    }

    int device = 0;
    CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, device));

    bool csvHeaderWritten = false;
    const std::string csvPath = opts.sweep ? std::string(kSweepCsvPath) : std::string();
    for (const auto &spec : runs) {
        if (spec.k <= 0 || spec.m <= 0 || spec.n <= 0) {
            std::fprintf(stderr, "Skipping invalid size %dx%dx%d.\n", spec.m,
                         spec.k, spec.n);
            continue;
        }
        runCase(spec, printMatrices, tiles, prop, csvPath,
                csvHeaderWritten);
    }

    return EXIT_SUCCESS;
}
