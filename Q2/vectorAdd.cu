#include <cstdio> // For printf
#include <cstdlib> // For malloc and free
#include <cmath> // For fabs
#include <chrono>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(1);                                    \
    }                                                    \
} while (0)

//--------------------------------------------------------------
// CPU reference version
//--------------------------------------------------------------
void vectorAddCPU(const float *a, const float *b, float *c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

//--------------------------------------------------------------
// GPU kernel
//--------------------------------------------------------------
__global__ void vectorAddKernel(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

//--------------------------------------------------------------
// Main program
//--------------------------------------------------------------
int main(int argc, char* argv[]) {
    // If a command-line argument is given, use it as N; otherwise, run a sweep of N values
    int single_N = -1;
    if (argc > 1) {
        single_N = std::atoi(argv[1]);
    }

    // Example sweep: 13 values from 2^9 to 2^24 (or adjust as needed)
    int N_values[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152};
    const int num_N = sizeof(N_values) / sizeof(N_values[0]);

    int sweep = (single_N <= 0);
    int num_runs = sweep ? num_N : 1;

    // Write header to CSV file if starting a new file (first run)
    FILE* fout = fopen("timing_results.csv", "w");
    if (fout) {
        fprintf(fout, "N,HostToDevice_ms,Kernel_ms,DeviceToHost_ms,MaxError\n");
        fclose(fout);
    } else {
        std::fprintf(stderr, "Could not open timing_results.csv for writing header!\n");
    }

    for (int run = 0; run < num_runs; ++run) {
        int N = sweep ? N_values[run] : single_N;
        size_t size = N * sizeof(float);

        //@@ 1. Allocate in host memory.
        float *h_a   = (float*)malloc(size);
        float *h_b   = (float*)malloc(size);
        float *h_c   = (float*)malloc(size);   // GPU result
        float *h_ref = (float*)malloc(size);   // CPU reference

        //@@ 2. Initialize host memory.
        for (int i = 0; i < N; i++) {
            h_a[i] = static_cast<float>(i);
            h_b[i] = static_cast<float>(2 * i);
        }

        // Compute CPU reference
        vectorAddCPU(h_a, h_b, h_ref, N);

        //@@ 3. Allocate in device memory.
        float *d_a, *d_b, *d_c;
        CHECK(cudaMalloc((void**)&d_a, size));
        CHECK(cudaMalloc((void**)&d_b, size));
        CHECK(cudaMalloc((void**)&d_c, size));

        // Timing using std::chrono
        double time_h2d = 0.0, time_kernel = 0.0, time_d2h = 0.0;

        //@@ 4. Copy from host memory to device memory (timed)
        auto t_h2d_start = std::chrono::high_resolution_clock::now();
        CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
        auto t_h2d_end = std::chrono::high_resolution_clock::now();
        time_h2d = std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

        //@@ 5. Initialize thread block and thread grid.
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        //@@ 6. Invoke the CUDA kernel (timed)
        auto t_kernel_start = std::chrono::high_resolution_clock::now();
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        auto t_kernel_end = std::chrono::high_resolution_clock::now();
        time_kernel = std::chrono::duration<double, std::milli>(t_kernel_end - t_kernel_start).count();

        //@@ 7. Copy results from GPU to CPU (timed)
        auto t_d2h_start = std::chrono::high_resolution_clock::now();
        CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
        auto t_d2h_end = std::chrono::high_resolution_clock::now();
        time_d2h = std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

        //@@ 8. Compare the results with the CPU reference result.
        double maxError = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = std::fabs(h_c[i] - h_ref[i]);
            if (diff > maxError) maxError = diff;
        }

        // Print results in CSV format for easy plotting
        std::printf("N,%d,HostToDevice_ms,%.4f,Kernel_ms,%.4f,DeviceToHost_ms,%.4f,MaxError,%e\n",
            N, time_h2d, time_kernel, time_d2h, maxError);

        // Append results to the CSV file (now always in append mode)
        FILE* fout = fopen("timing_results.csv", "a");
        if (fout) {
            fprintf(fout, "%d,%.4f,%.4f,%.4f,%e\n", N, time_h2d, time_kernel, time_d2h, maxError);
            fclose(fout);
        } else {
            std::fprintf(stderr, "Could not open timing_results.csv for writing!\n");
        }

        //@@ 9. Free host  memory.
        free(h_a);
        free(h_b);
        free(h_c);
        free(h_ref);

        //@@ 10. Free device memory.
        CHECK(cudaFree(d_a));
        CHECK(cudaFree(d_b));
        CHECK(cudaFree(d_c));
    }
    return 0;
}
