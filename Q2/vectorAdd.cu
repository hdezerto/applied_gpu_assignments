#include <cstdio> // For printf
#include <cstdlib> // For malloc and free
#include <cmath> // For fabs

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
int main() {
    //int N = 1 << 20; // 1,048,576 elements
    int N = 263149;
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

    //@@ 4. Copy from host memory to device memory.
    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    //@@ 5. Initialize thread block and thread grid.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    //@@ 6. Invoke the CUDA kernel.
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    //@@ 7. Copy results from GPU to CPU.
    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    //@@ 8. Compare the results with the CPU reference result.
    double maxError = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = std::fabs(h_c[i] - h_ref[i]);
        if (diff > maxError) maxError = diff;
    }
    std::printf("Max error between CPU and GPU results: %e\n", maxError);

    //@@ 9. Free host  memory.
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);

    //@@ 10. Free device memory.
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
