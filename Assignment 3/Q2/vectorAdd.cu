#include <cstdio> // For printf
#include <cstdlib> // For malloc and free
#include <cmath> // For fabs
#include <string>
#include <algorithm>
#include <cuda_runtime.h>


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
// Function to compute maximum error between two vectors
//--------------------------------------------------------------

double computeMaxError(const float* result, const float* reference, int N) {
    double maxError = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = std::fabs(result[i] - reference[i]);
        if (diff > maxError) maxError = diff;
    }
    return maxError;
}


//--------------------------------------------------------------
// Main program
//--------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Usage: vectorAdd [--streamed] [--sweepN] [--sweepSseg]
    // Hardcoded defaults
    const int N_default = 1048576;
    const int S_seg_default = 65536;
    bool use_streams = false;
    bool sweepN = false;
    bool sweepSseg = false;
    // Parse flags
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--streamed") {
            use_streams = true;
        } else if (std::string(argv[i]) == "--sweepN") {
            sweepN = true;
        } else if (std::string(argv[i]) == "--sweepSseg") {
            sweepSseg = true;
        }
    }

    // Sweep ranges
    int N_values[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152};
    int num_N = sizeof(N_values) / sizeof(N_values[0]);
    int S_seg_values[] = {4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288};
    int num_S = sizeof(S_seg_values) / sizeof(S_seg_values[0]);

    // Determine sweep mode
    int sweep_N_loops = sweepN ? num_N : 1;
    int sweep_S_loops = sweepSseg ? num_S : 1;

    const char* csv_file = "timing_results.csv";
    bool write_header = false;
    {
        FILE* fchk = fopen(csv_file, "r");
        if (!fchk) {
            write_header = true; // file does not exist
        } else {
            fseek(fchk, 0, SEEK_END);
            long sz = ftell(fchk);
            if (sz == 0) write_header = true; // empty file
            fclose(fchk);
        }
    }
    if (write_header) {
        FILE* fout = fopen(csv_file, "w");
        if (fout) {
            fprintf(fout, "N,GPUElapsed_ms,MaxError,Streamed,S_seg\n");
            fclose(fout);
        }
    }

    for (int iN = 0; iN < sweep_N_loops; ++iN) {
        int N = sweepN ? N_values[iN] : N_default;
        for (int iS = 0; iS < sweep_S_loops; ++iS) {
            int S_seg = sweepSseg ? S_seg_values[iS] : S_seg_default;
            size_t size = N * sizeof(float);

            //@@ 1. Allocate in host memory. Use pinned memory for transfer buffers to enable overlap.
            float *h_a   = nullptr;
            float *h_b   = nullptr;
            float *h_c   = nullptr;   // GPU result
            CHECK(cudaMallocHost((void**)&h_a, size));
            CHECK(cudaMallocHost((void**)&h_b, size));
            CHECK(cudaMallocHost((void**)&h_c, size));
            float *h_ref = (float*)malloc(size);   // CPU reference

            //@@ 2. Initialize input vectors
            for (int i = 0; i < N; i++) {
                h_a[i] = static_cast<float>(i);
                h_b[i] = static_cast<float>(2 * i);
            }

            // Compute CPU reference
            vectorAddCPU(h_a, h_b, h_ref, N);
            
            // ------- START TIMING HERE -------
            float gpu_time_ms_f = 0.0f;
            // Device timer
            cudaEvent_t ev_start, ev_stop;
            CHECK(cudaEventCreate(&ev_start));
            CHECK(cudaEventCreate(&ev_stop));
            CHECK(cudaEventRecord(ev_start));

            if (!use_streams) {
                // Non-streamed baseline
                float *d_a, *d_b, *d_c;
                CHECK(cudaMalloc((void**)&d_a, size));
                CHECK(cudaMalloc((void**)&d_b, size));
                CHECK(cudaMalloc((void**)&d_c, size));

                CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

                int threadsPerBlock = 256;
                int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
                vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
                CHECK(cudaGetLastError());
                CHECK(cudaDeviceSynchronize());

                CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

                CHECK(cudaFree(d_a));
                CHECK(cudaFree(d_b));
                CHECK(cudaFree(d_c));
            } else {
                // Streamed version
                int num_streams = 4;
                cudaStream_t streams[4];
                for (int i = 0; i < num_streams; ++i) CHECK(cudaStreamCreate(&streams[i]));

                // Allocate device memory for each segment
                float *d_a[4], *d_b[4], *d_c[4];
                int num_segments = (N + S_seg - 1) / S_seg;
                for (int i = 0; i < num_streams; ++i) {
                    CHECK(cudaMalloc((void**)&d_a[i], S_seg * sizeof(float)));
                    CHECK(cudaMalloc((void**)&d_b[i], S_seg * sizeof(float)));
                    CHECK(cudaMalloc((void**)&d_c[i], S_seg * sizeof(float)));
                }

                for (int seg = 0; seg < num_segments; ++seg) {
                    int stream_id = seg % num_streams;
                    int offset = seg * S_seg;
                    int seg_size = std::min(S_seg, N - offset);
                    // Async copy segment to device
                    CHECK(cudaMemcpyAsync(d_a[stream_id], h_a + offset, seg_size * sizeof(float), cudaMemcpyHostToDevice, streams[stream_id]));
                    CHECK(cudaMemcpyAsync(d_b[stream_id], h_b + offset, seg_size * sizeof(float), cudaMemcpyHostToDevice, streams[stream_id]));
                    // Launch kernel for segment
                    int threadsPerBlock = 256;
                    int blocksPerGrid = (seg_size + threadsPerBlock - 1) / threadsPerBlock;
                    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[stream_id]>>>(d_a[stream_id], d_b[stream_id], d_c[stream_id], seg_size);
                    CHECK(cudaGetLastError());
                    // Async copy result back to host
                    CHECK(cudaMemcpyAsync(h_c + offset, d_c[stream_id], seg_size * sizeof(float), cudaMemcpyDeviceToHost, streams[stream_id]));
                }
                // Synchronize all streams
                for (int i = 0; i < num_streams; ++i) CHECK(cudaStreamSynchronize(streams[i]));
                // Free device memory and destroy streams
                for (int i = 0; i < num_streams; ++i) {
                    CHECK(cudaFree(d_a[i]));
                    CHECK(cudaFree(d_b[i]));
                    CHECK(cudaFree(d_c[i]));
                    CHECK(cudaStreamDestroy(streams[i]));
                }
            }
            // Stop device timer
            CHECK(cudaEventRecord(ev_stop));
            CHECK(cudaEventSynchronize(ev_stop));
            CHECK(cudaEventElapsedTime(&gpu_time_ms_f, ev_start, ev_stop));
            CHECK(cudaEventDestroy(ev_start));
            CHECK(cudaEventDestroy(ev_stop));

            // ------- END TIMING HERE -------

            double gpu_time_ms = static_cast<double>(gpu_time_ms_f);
            double maxError = computeMaxError(h_c, h_ref, N);

            // Print results in CSV format for easy plotting
            std::printf("N,%d,GPUElapsed_ms,%.4f,MaxError,%e,Streamed,%d,S_seg,%d\n",
                N, gpu_time_ms, maxError, use_streams ? 1 : 0, S_seg);

            // Append results to the CSV file
            FILE* fout = fopen(csv_file, "a");
            if (fout) {
                fprintf(fout, "%d,%.4f,%e,%d,%d\n", N, gpu_time_ms, maxError, use_streams ? 1 : 0, S_seg);
                fclose(fout);
            } else {
                std::fprintf(stderr, "Could not open %s for writing!\n", csv_file);
            }

            //@@ 9. Free host  memory.
            CHECK(cudaFreeHost(h_a));
            CHECK(cudaFreeHost(h_b));
            CHECK(cudaFreeHost(h_c));
            free(h_ref);
        }
    }

    return 0;
}
