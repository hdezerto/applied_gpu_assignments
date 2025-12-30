#include <cuda_fp16.h> // NEW: WMMA half support
#include <cuda_runtime.h>
#include <mma.h>        // NEW: WMMA API

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

// This file keeps the Assignment 2 Q3 structure (CPU ref, baseline gemm,
// tiled gemm, timing, CSV). Marked lines/blocks are NEW or CHANGED for WMMA.

using namespace nvcuda; // NEW
using DataType = float; // Device baseline uses float (same as original)

constexpr int WMMA_M = 16; // NEW
constexpr int WMMA_N = 16; // NEW
constexpr int WMMA_K = 16; // NEW

//------------------------------------------------------------------------------
// Host utilities (mostly unchanged)
//------------------------------------------------------------------------------

// CPU reference now in double for accuracy (NEW)
void matMulCPU(const double *A, const double *B, double *C, int numARows,
               int numAColumns, int numBColumns) {
    for (int row = 0; row < numARows; ++row) {
        for (int col = 0; col < numBColumns; ++col) {
            double acc = 0.0;
            for (int k = 0; k < numAColumns; ++k) {
                acc += A[row * numAColumns + k] * B[k * numBColumns + col];
            }
            C[row * numBColumns + col] = acc;
        }
    }
}

// CHANGED: initialize doubles
void initializeMatrix(double *mat, int rows, int cols, double scale) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            mat[idx] = scale * std::sin(0.01 * static_cast<double>(idx)) +
                       (1.0 - scale) * std::cos(0.005 * static_cast<double>(idx));
        }
    }
}

double maxAbsDiff(const double *lhs, const double *rhs, std::size_t count) {
    double max_err = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        max_err = std::max<double>(max_err, std::fabs(lhs[i] - rhs[i]));
    }
    return max_err;
}

// NEW: compare float output vs double reference
double maxAbsDiffFloatToDouble(const float *lhs, const double *rhs,
                               std::size_t count) {
    double max_err = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        max_err = std::max<double>(max_err, std::fabs(static_cast<double>(lhs[i]) - rhs[i]));
    }
    return max_err;
}

void printMatrix(const double *mat, int rows, int cols, const char *label,
                 bool enabled, int maxRows = 3, int maxCols = 3) {
    if (!enabled) {
        return;
    }
    std::printf("%s (%dx%d):\n", label, rows, cols);
    const int showRows = std::min(rows, maxRows);
    const int showCols = std::min(cols, maxCols);
    for (int r = 0; r < showRows; ++r) {
        for (int c = 0; c < showCols; ++c) {
            std::printf("%10.4f ", mat[r * cols + c]);
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

// NEW: helper to print float output safely
void printMatrixFromFloat(const float *mat, int rows, int cols, const char *label,
                          bool enabled, int maxRows = 3, int maxCols = 3) {
    if (!enabled) {
        return;
    }
    std::vector<double> tmp(static_cast<std::size_t>(rows) * cols);
    for (int i = 0; i < rows * cols; ++i) {
        tmp[static_cast<std::size_t>(i)] = static_cast<double>(mat[i]);
    }
    printMatrix(tmp.data(), rows, cols, label, enabled, maxRows, maxCols);
}

//------------------------------------------------------------------------------
// Argument parsing (kept structure, add WMMA options)
//------------------------------------------------------------------------------

struct SizeSpec {
    int m;
    int k;
    int n;
};

struct Options {
    int m = 256; // rows of A / C
    int k = 256; // cols of A / rows of B
    int n = 256; // cols of B / C
    bool printMatrices = true;
    bool sweep = false; // predefined sweep
    int blockWarpsM = 2; // NEW: warps in M per block for WMMA
    int blockWarpsN = 2; // NEW: warps in N per block for WMMA
    int iterations = 5;  // NEW: averaging count
};

const std::vector<SizeSpec> kSweepSizes = {
    {512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048},
    {4096, 4096, 4096}, {8192, 8192, 8192}
};

constexpr const char *kSweepCsvPath = "wmma_bonus_sweep.csv"; // CHANGED name

void printUsage(const char *prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --m <rowsA>     Rows of A / C (default 256)\n"
        "  --k <shared>    Columns of A / rows of B (default 256)\n"
        "  --n <colsB>     Columns of B / C (default 256)\n"
        "  --sweep         Run preset sizes (512..8192)\n"
        "  --print         Enable matrix previews (default)\n"
        "  --no-print      Disable matrix previews\n"
        "  --block-warps-m <int>  Warps along M per block (default 2)\n"
        "  --block-warps-n <int>  Warps along N per block (default 2)\n"
        "  --iters <int>          Kernel timing iterations (default 5)\n"
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
        } else if (arg == "--block-warps-m" && i + 1 < argc) { // NEW
            opts.blockWarpsM = std::atoi(argv[++i]);
        } else if (arg == "--block-warps-n" && i + 1 < argc) { // NEW
            opts.blockWarpsN = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) { // NEW
            opts.iterations = std::atoi(argv[++i]);
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
                  float wmmaMs, double gemmErr, double wmmaErr,
                  int warpsM, int warpsN) { // CHANGED columns
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
        file << "m,k,n,cpu_ms,gemm_ms,wmma_ms,gemm_err,wmma_err,warpsM,warpsN\n";
        headerWritten = true;
    }
    file << size.m << ',' << size.k << ',' << size.n << ',' << cpuMs << ','
         << gemmMs << ',' << wmmaMs << ',' << gemmErr << ',' << wmmaErr << ','
         << warpsM << ',' << warpsN << '\n';
}

//------------------------------------------------------------------------------
// CUDA kernels (baseline kept, WMMA added)
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
    extern __shared__ DataType shared[];
    DataType *tileA = shared;
    DataType *tileB = tileA + tileX * tileY;

    const int row = blockIdx.y * tileY + threadIdx.y;
    const int col = blockIdx.x * tileX + threadIdx.x;

    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int linearTid = threadIdx.y * blockDim.x + threadIdx.x;

    DataType acc = 0.0f;

    for (int tileIdx = 0; tileIdx < numAColumns; tileIdx += tileX) {
        for (int idx = linearTid; idx < tileY * tileX; idx += threadsPerBlock) {
            const int localRow = idx / tileX;
            const int localCol = idx % tileX;
            const int globalRow = blockIdx.y * tileY + localRow;
            const int globalCol = tileIdx + localCol;
            if (globalRow < numARows && globalCol < numAColumns) {
                tileA[idx] = A[globalRow * numAColumns + globalCol];
            } else {
                tileA[idx] = 0.0f;
            }
        }

        for (int idx = linearTid; idx < tileX * tileX; idx += threadsPerBlock) {
            const int localRow = idx / tileX;
            const int localCol = idx % tileX;
            const int globalRow = tileIdx + localRow;
            const int globalCol = blockIdx.x * tileX + localCol;
            if (globalRow < numAColumns && globalCol < numBColumns) {
                tileB[idx] = B[globalRow * numBColumns + globalCol];
            } else {
                tileB[idx] = 0.0f;
            }
        }
        __syncthreads();

        const int maxK = min(tileX, numAColumns - tileIdx);
        for (int k = 0; k < maxK; ++k) {
            acc += tileA[threadIdx.y * tileX + k] *
                   tileB[k * tileX + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = acc;
    }
}

// NEW: WMMA kernel

template <int BLOCK_WARPS_M, int BLOCK_WARPS_N>
__global__ void wmma_gemm(const half *A, const half *B, float *C,
                          int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int WARPS = BLOCK_WARPS_M * BLOCK_WARPS_N;
    const int warpId = threadIdx.y;
    const int laneId = threadIdx.x;
    if (warpId >= WARPS) {
        return;
    }

    const int warpRow = warpId / BLOCK_WARPS_N;
    const int warpCol = warpId % BLOCK_WARPS_N;

    const int globalRow = (blockIdx.y * BLOCK_WARPS_M + warpRow) * WMMA_M;
    const int globalCol = (blockIdx.x * BLOCK_WARPS_N + warpCol) * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;

        const bool validA = (globalRow + WMMA_M <= M) && (k0 + WMMA_K <= K);
        const bool validB = (globalCol + WMMA_N <= N) && (k0 + WMMA_K <= K);

        if (validA) {
            wmma::load_matrix_sync(aFrag, A + globalRow * lda + k0, lda);
        } else {
            wmma::fill_fragment(aFrag, __float2half(0.0f));
        }

        if (validB) {
            wmma::load_matrix_sync(bFrag, B + k0 * ldb + globalCol, ldb);
        } else {
            wmma::fill_fragment(bFrag, __float2half(0.0f));
        }

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    if (globalRow < M && globalCol < N) {
        float tmp[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(tmp, cFrag, WMMA_N, wmma::mem_row_major);
        for (int i = 0; i < WMMA_M; ++i) {
            const int r = globalRow + i;
            if (r >= M) break;
            for (int j = 0; j < WMMA_N; ++j) {
                const int c = globalCol + j;
                if (c >= N) break;
                C[r * ldc + c] = tmp[i * WMMA_N + j];
            }
        }
    }
    (void)laneId; // silence unused warning
}

//------------------------------------------------------------------------------
// Kernel launch helpers (baseline unchanged, add WMMA)
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

// NEW: WMMA launcher
float launchWmmaKernel(const half *d_A, const half *d_B, float *d_C, int M,
                       int K, int N, int warpsM, int warpsN, int iters) {
    const int warpsPerBlock = warpsM * warpsN;
    dim3 block(32, warpsPerBlock);
    dim3 grid((N + warpsN * WMMA_N - 1) / (warpsN * WMMA_N),
              (M + warpsM * WMMA_M - 1) / (warpsM * WMMA_M));

    cudaEvent_t start{}, stop{};
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    auto launch = [&]() {
        if (warpsM == 1 && warpsN == 1) {
            wmma_gemm<1, 1><<<grid, block>>>(d_A, d_B, d_C, M, N, K, K, N, N);
        } else if (warpsM == 2 && warpsN == 1) {
            wmma_gemm<2, 1><<<grid, block>>>(d_A, d_B, d_C, M, N, K, K, N, N);
        } else if (warpsM == 1 && warpsN == 2) {
            wmma_gemm<1, 2><<<grid, block>>>(d_A, d_B, d_C, M, N, K, K, N, N);
        } else if (warpsM == 2 && warpsN == 2) {
            wmma_gemm<2, 2><<<grid, block>>>(d_A, d_B, d_C, M, N, K, K, N, N);
        } else {
            wmma_gemm<2, 2><<<grid, block>>>(d_A, d_B, d_C, M, N, K, K, N, N);
        }
    };

    // Warmup
    launch();
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float totalMs = 0.0f;
    CHECK(cudaEventElapsedTime(&totalMs, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return totalMs / static_cast<float>(iters);
}

//------------------------------------------------------------------------------
// Single case runner (structure preserved, added WMMA path)
//------------------------------------------------------------------------------

void runCase(const SizeSpec &size, bool printMatrices,
             const std::vector<std::pair<int, int>> &tiles,
             const cudaDeviceProp &prop, const std::string &csvPath,
             bool &csvHeaderWritten, int warpsM, int warpsN, int iters) {
    const int numARows = size.m;
    const int numAColumns = size.k;
    const int numBRows = size.k;
    const int numBColumns = size.n;
    const int numCRows = numARows;
    const int numCColumns = numBColumns;

    std::printf("\n==== GEMM case: A=%dx%d, B=%dx%d ====\n", numARows,
                numAColumns, numBRows, numBColumns);
    std::printf("WMMA tile %dx%dx%d, block warps %d x %d\n", WMMA_M, WMMA_N,
                WMMA_K, warpsM, warpsN);

    const std::size_t elementsA = static_cast<std::size_t>(numARows) * numAColumns;
    const std::size_t elementsB = static_cast<std::size_t>(numBRows) * numBColumns;
    const std::size_t elementsC = static_cast<std::size_t>(numCRows) * numCColumns;

    // Host buffers
    std::vector<double> h_A(elementsA);
    std::vector<double> h_B(elementsB);
    std::vector<double> h_ref(elementsC);
    std::vector<float> h_C(elementsC);
    std::vector<float> h_C_tiled(elementsC);
    std::vector<float> h_C_wmma(elementsC);

    initializeMatrix(h_A.data(), numARows, numAColumns, 0.7);
    initializeMatrix(h_B.data(), numBRows, numBColumns, 0.3);

    const auto cpuStart = std::chrono::high_resolution_clock::now();
    matMulCPU(h_A.data(), h_B.data(), h_ref.data(), numARows, numAColumns,
              numBColumns);
    const auto cpuStop = std::chrono::high_resolution_clock::now();
    const double cpuMs =
        std::chrono::duration<double, std::milli>(cpuStop - cpuStart).count();

    // Convert to device types
    std::vector<DataType> h_A_f(elementsA);
    std::vector<DataType> h_B_f(elementsB);
    std::vector<half> h_A_h(elementsA);
    std::vector<half> h_B_h(elementsB);
    for (std::size_t i = 0; i < elementsA; ++i) {
        h_A_f[i] = static_cast<DataType>(h_A[i]);
        h_A_h[i] = __double2half(h_A[i]);
    }
    for (std::size_t i = 0; i < elementsB; ++i) {
        h_B_f[i] = static_cast<DataType>(h_B[i]);
        h_B_h[i] = __double2half(h_B[i]);
    }

    DataType *d_A = nullptr;
    DataType *d_B = nullptr;
    DataType *d_C = nullptr;
    DataType *d_C_tiled = nullptr;
    half *d_Ah = nullptr;
    half *d_Bh = nullptr;
    float *d_C_wmma = nullptr;

    const std::size_t bytesA = elementsA * sizeof(DataType);
    const std::size_t bytesB = elementsB * sizeof(DataType);
    const std::size_t bytesC = elementsC * sizeof(DataType);
    const std::size_t bytesAh = elementsA * sizeof(half);
    const std::size_t bytesBh = elementsB * sizeof(half);

    CHECK(cudaMalloc(&d_A, bytesA));
    CHECK(cudaMalloc(&d_B, bytesB));
    CHECK(cudaMalloc(&d_C, bytesC));
    CHECK(cudaMalloc(&d_C_tiled, bytesC));
    CHECK(cudaMalloc(&d_Ah, bytesAh));
    CHECK(cudaMalloc(&d_Bh, bytesBh));
    CHECK(cudaMalloc(&d_C_wmma, bytesC));

    CHECK(cudaMemcpy(d_A, h_A_f.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B_f.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Ah, h_A_h.data(), bytesAh, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Bh, h_B_h.data(), bytesBh, cudaMemcpyHostToDevice));

    // Baseline gemm
    CHECK(cudaMemset(d_C, 0, bytesC));
    const float gemmMs = launchBaselineKernel(d_A, d_B, d_C, numARows,
                                              numAColumns, numBColumns);
    CHECK(cudaMemcpy(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost));
    const double gemmError =
        maxAbsDiffFloatToDouble(h_C.data(), h_ref.data(), elementsC);

    // Tiled gemm (unchanged) – keep for completeness
    std::vector<float> tiledTimings;
    std::vector<double> tiledErrors;
    tiledTimings.reserve(tiles.size());
    tiledErrors.reserve(tiles.size());

    for (const auto &tile : tiles) {
        const int tileX = tile.first;
        const int tileY = tile.second;
        CHECK(cudaMemset(d_C_tiled, 0, bytesC));
        float tiledMs = 0.0f;
        if (!launchTiledKernel(d_A, d_B, d_C_tiled, numARows, numAColumns,
                               numBColumns, tileX, tileY, prop, tiledMs)) {
            tiledTimings.push_back(std::numeric_limits<float>::quiet_NaN());
            tiledErrors.push_back(std::numeric_limits<double>::quiet_NaN());
            continue;
        }
        CHECK(cudaMemcpy(h_C_tiled.data(), d_C_tiled, bytesC, cudaMemcpyDeviceToHost));
        const double tiledError = maxAbsDiffFloatToDouble(h_C_tiled.data(),
                                                          h_ref.data(), elementsC);
        tiledTimings.push_back(tiledMs);
        tiledErrors.push_back(tiledError);
    }

    // WMMA Tensor Core
    CHECK(cudaMemset(d_C_wmma, 0, bytesC));
    const float wmmaMs = launchWmmaKernel(d_Ah, d_Bh, d_C_wmma, numARows,
                                          numAColumns, numBColumns, warpsM,
                                          warpsN, iters);
    CHECK(cudaMemcpy(h_C_wmma.data(), d_C_wmma, bytesC, cudaMemcpyDeviceToHost));
    const double wmmaError =
        maxAbsDiffFloatToDouble(h_C_wmma.data(), h_ref.data(), elementsC);

    // Print summaries
    std::printf("CPU reference time: %.3f ms\n", cpuMs);
    printMatrix(h_ref.data(), numCRows, numCColumns, "CPU reference:",
                printMatrices);

    std::printf("CUDA gemm result:\n");
    printMatrixFromFloat(h_C.data(), numCRows, numCColumns, "gemm result:",
                         printMatrices);
    std::printf("timing: %.3f ms\n", gemmMs);
    std::printf("max error vs CPU: %.3e\n\n", gemmError);

    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const auto &tile = tiles[i];
        const float tMs = tiledTimings[i];
        const double tErr = tiledErrors[i];
        std::printf("CUDA tiled_gemm [%d,%d] timing: %.3f ms, max err: %.3e\n",
                    tile.first, tile.second, tMs, tErr);
    }

    std::printf("WMMA Tensor Core timing: %.3f ms, max err vs CPU: %.3e\n",
                wmmaMs, wmmaError);
    printMatrixFromFloat(h_C_wmma.data(), numCRows, numCColumns,
                         "wmma result:", printMatrices);

    appendCsvRow(csvPath, csvHeaderWritten, size, cpuMs, gemmMs, wmmaMs,
                 gemmError, wmmaError, warpsM, warpsN);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    CHECK(cudaFree(d_C_tiled));
    CHECK(cudaFree(d_Ah));
    CHECK(cudaFree(d_Bh));
    CHECK(cudaFree(d_C_wmma));
}

//------------------------------------------------------------------------------
// Entry point (structure preserved)
//------------------------------------------------------------------------------

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

    if (prop.major < 7) {
        std::fprintf(stderr, "Tensor Cores require sm_70+. Detected sm_%d%d.\n",
                     prop.major, prop.minor);
    }

    bool csvHeaderWritten = false;
    const std::string csvPath = opts.sweep ? std::string(kSweepCsvPath) : std::string();
    for (const auto &spec : runs) {
        if (spec.k <= 0 || spec.m <= 0 || spec.n <= 0) {
            std::fprintf(stderr, "Skipping invalid size %dx%dx%d.\n", spec.m,
                         spec.k, spec.n);
            continue;
        }
        runCase(spec, printMatrices, tiles, prop, csvPath, csvHeaderWritten,
                opts.blockWarpsM, opts.blockWarpsN, opts.iterations);
    }

    return EXIT_SUCCESS;
}
