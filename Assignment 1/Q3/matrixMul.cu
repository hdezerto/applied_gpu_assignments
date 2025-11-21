#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

// ----------- EDIT HERE: Choose precision -----------
//typedef double T; // Set to double for double precision
typedef float T; // Change to float for single precision


// CPU reference implementation
void matMulCPU(const T* A, const T* B, T* C, int numARows, int numAColumns, int numBColumns) {
    for (int row = 0; row < numARows; ++row) {
        for (int col = 0; col < numBColumns; ++col) {
            T sum = 0.0;
            for (int k = 0; k < numAColumns; ++k) {
                sum += A[row * numAColumns + k] * B[k * numBColumns + col];
            }
            C[row * numBColumns + col] = sum;
        }
    }
}

// GPU kernel. The __global__ keyword indicates that this function is a kernel (can be called from host and runs on device)
__global__ void matMulKernel(const T* A, const T* B, T* C, int numARows, int numAColumns, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numARows && col < numBColumns) {
        T sum = 0.0;
        for (int k = 0; k < numAColumns; ++k) {
            sum += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}


int main(int argc, char* argv[]) {
    // Check for argument to disable error computation
    bool compute_error = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--no-error") == 0) compute_error = false;
    }
    // Typical matrix sizes to test (A: rows x cols, B: rows x cols)
    int test_sizes[][4] = {
        {128, 128, 128, 128},
        {256, 256, 256, 256},
        {512, 512, 512, 512},
        {1024, 1024, 1024, 1024},
        {2048, 2048, 2048, 2048},
        {4096, 4096, 4096, 4096},
        {8192, 8192, 8192, 8192},
        {16384, 16384, 16384, 16384},
        // Custom cases (uncomment to enable):
        // {128, 256, 256, 32},
        // {1024, 8191, 8191, 8197},
    };
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    // Clear CSV file and write header at the start
    FILE* fout = fopen("timing_results.csv", "w");
    if (fout) {
        fprintf(fout, "NARows,NBColumns,HostToDevice_ms,Kernel_ms,DeviceToHost_ms,MaxError\n");
        fclose(fout);
    }

    for (int t = 0; t < num_tests; ++t) {
        int numARows = test_sizes[t][0];
        int numAColumns = test_sizes[t][1];
        int numBRows = test_sizes[t][2];
        int numBColumns = test_sizes[t][3];
        int numCRows = numARows;
        int numCColumns = numBColumns;

        size_t sizeA = numARows * numAColumns * sizeof(T);
        size_t sizeB = numBRows * numBColumns * sizeof(T);
        size_t sizeC = numCRows * numCColumns * sizeof(T);

        T *h_A = (T*)malloc(sizeA);
        T *h_B = (T*)malloc(sizeB);
        T *h_C = (T*)malloc(sizeC);
        T *h_ref = (T*)malloc(sizeC);
        if (!h_A || !h_B || !h_C || !h_ref) {
            fprintf(stderr, "Host memory allocation failed!\n");
            exit(1);
        }

        for (int i = 0; i < numARows * numAColumns; ++i) h_A[i] = static_cast<T>(i % 100);
        for (int i = 0; i < numBRows * numBColumns; ++i) h_B[i] = static_cast<T>((i * 2) % 100);

        if (compute_error) {
            matMulCPU(h_A, h_B, h_ref, numARows, numAColumns, numBColumns);
        }

        T *d_A, *d_B, *d_C;
        CHECK(cudaMalloc((void**)&d_A, sizeA));
        CHECK(cudaMalloc((void**)&d_B, sizeB));
        CHECK(cudaMalloc((void**)&d_C, sizeC));

        auto t_h2d_start = std::chrono::high_resolution_clock::now();
        CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
        auto t_h2d_end = std::chrono::high_resolution_clock::now();
        double time_h2d = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

        auto t_kernel_start = std::chrono::high_resolution_clock::now();
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((numCColumns + 15) / 16, (numCRows + 15) / 16);
        matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        auto t_kernel_end = std::chrono::high_resolution_clock::now();
        double time_kernel = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();

        auto t_d2h_start = std::chrono::high_resolution_clock::now();
        CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        auto t_d2h_end = std::chrono::high_resolution_clock::now();
        double time_d2h = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

        double maxError = 0.0;
        if (compute_error) {
            for (int i = 0; i < numCRows * numCColumns; ++i) {
                double diff = std::fabs(h_C[i] - h_ref[i]);
                if (diff > maxError) maxError = diff;
            }
            printf("Max error: %e\n", maxError);
        } else {
            maxError = -1.0; // Indicate not computed
        }

        printf("%d,%d,%.4f,%.4f,%.4f,%e\n", numARows, numBColumns, time_h2d, time_kernel, time_d2h, maxError);

        FILE* fout = fopen("timing_results.csv", "a");
        if (fout) {
            fprintf(fout, "%d,%d,%.4f,%.4f,%.4f,%e\n", numARows, numBColumns, time_h2d, time_kernel, time_d2h, maxError);
            fclose(fout);
        } else {
            fprintf(stderr, "Could not open timing_results.csv for writing!\n");
        }

        free(h_A); free(h_B); free(h_C); free(h_ref);
        CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    }
    return 0;
}