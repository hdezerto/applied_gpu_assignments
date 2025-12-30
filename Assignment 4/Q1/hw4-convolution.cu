#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <math.h>
#include <random>
#include <stdlib.h>
#include <sys/time.h>
#include <cstdio>

#define CHECK(call) do {                                 \
    cudaError_t err = (call);                            \
    if (err != cudaSuccess) {                            \
        std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
            cudaGetErrorString(err), __FILE__, __LINE__);\
        std::exit(1);                                    \
    }                                                    \
} while (0)

struct timeval t_start, t_end;
void cputimer_start(){
  gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info){
  gettimeofday(&t_end, 0);
  double time = (1000000.0*(t_end.tv_sec-t_start.tv_sec) + t_end.tv_usec-t_start.tv_usec);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

#define MASK_WIDTH 5
#define THREADS_PER_BLOCK 1024
#define TILE_WIDTH (THREADS_PER_BLOCK + MASK_WIDTH - 1)

__global__ void convolution_1D_basic(float *N, float *M, float *P, int width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (MASK_WIDTH/2);

    for(int j = 0; j < MASK_WIDTH; j++){
        if(N_start_point + j >= 0 && N_start_point + j < width){
            Pvalue += N[N_start_point + j]*M[j];
        }
    }

    P[i] = Pvalue;

}

__global__ void convolution_1D_tiled(float *N, float *M, float *P, int width)
{
  // shared memory
  __shared__ float input_tile[TILE_WIDTH];
  __shared__ float output_tile[THREADS_PER_BLOCK];

  // constants used
  const unsigned int s_idx = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int edge_length = (MASK_WIDTH/2);
  const int N_start_point = i - edge_length;

  // input standard cells
  input_tile[s_idx + edge_length] = N[i];

  // start halo cells
  if(s_idx < edge_length && i - edge_length > 0){
      input_tile[s_idx] = N[i-edge_length];
  }

  // end halo cells
  if(s_idx >= THREADS_PER_BLOCK - edge_length && i + edge_length < width){
      input_tile[s_idx + edge_length*2] = N[i+edge_length];
  }

  __syncthreads();

  float Pvalue = 0;

  for(int j = 0; j < MASK_WIDTH; j++){
      if(N_start_point + j >= 0 && N_start_point + j < width){
          Pvalue += input_tile[s_idx + j]*M[j];
      }
  }
  output_tile[s_idx] = Pvalue;
  __syncthreads();
  P[i] = output_tile[s_idx];
}

int main(int argc, char *argv[]) {

  // Read the arguments from the command line
  if (argc < 2){
      printf("needs input argument");
      return 1;
  }
  int N = atoi(argv[1]);


  float *hostN; // The input array N of length N
  float *hostM; // The 1D mask M of length MASK_WIDTH
  float *hostP; // The output array P of length N
  float *hostResBasic; // The output array P for basic
  float *hostResTiles; // The output array P for tiled

  cputimer_start();
  //@@ Allocate the host memory
  CHECK(cudaHostAlloc(&hostN, N*sizeof(float), cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&hostM, MASK_WIDTH*sizeof(float), cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&hostP, N*sizeof(float), cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&hostResBasic, N*sizeof(float), cudaHostAllocDefault));
  CHECK(cudaHostAlloc(&hostResTiles, N*sizeof(float), cudaHostAllocDefault));

  cputimer_stop("Allocated host memory");

  float *deviceN;
  float *deviceM;
  float *deviceP;

  cputimer_start();
  //@@ Allocate the device memory
  CHECK(cudaMalloc(&deviceN, N*sizeof(float)));
  CHECK(cudaMalloc(&deviceM, MASK_WIDTH*sizeof(float)));
  CHECK(cudaMalloc(&deviceP, N*sizeof(float)));
  cputimer_stop("Allocated device memory");


  cputimer_start();
  //@@ Initialize N with random values
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  for(int i = 0; i < N; i++){
      hostN[i] = distribution(generator);
  }
  //@@ Initialize M with [-0.25, 0.5, 1.0, 0.5, 0.25]
  float const hostM_default[5] = {-0.25, 0.5, 1.0, 0.5, 0.25};
  memcpy(hostM, hostM_default, MASK_WIDTH*sizeof(float));
  //@@ Initialize P with 0.0
  memset(hostP, 0, N*sizeof(float));
  cputimer_stop("Host memory values initialized");

  // CPU implementation
  cputimer_start();
  for(int i = 0; i < N; i++){
      int N_start_point = i - (MASK_WIDTH/2);
      for(int j = 0; j < MASK_WIDTH; j++){
          if(N_start_point + j >= 0 && N_start_point + j < N){
              hostP[i] += hostN[N_start_point + j]*hostM[j];
          }
      }
  }
  cputimer_stop("Finished CPU convolution");


  cputimer_start();
  //@@ INSERT CODE HERE
  CHECK(cudaMemcpy(deviceN, hostN, N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceM, hostM, MASK_WIDTH*sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(deviceP, hostP, N*sizeof(float), cudaMemcpyHostToDevice));
  cputimer_stop("Copying data to the GPU.");


  //@@  Define the execution configuration
  int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  //@@  Run the 1D convolution kernel (basic)
  cputimer_start();
  convolution_1D_basic<<<blocksPerGrid, THREADS_PER_BLOCK>>>(deviceN, deviceM, deviceP, N);
  cputimer_stop("Finished 1D convolution(basic)");

  cputimer_start();
  //@@ INSERT CODE HERE
  CHECK(cudaMemcpy(hostResBasic, deviceP, N*sizeof(float), cudaMemcpyDeviceToHost));
  // log results
  FILE *fbasic = fopen("convolution_basic.txt", "w");
  for(int i = 0; i < N; i++){
      fprintf(fbasic, "%f\n", hostResBasic[i]);
  }
  fclose(fbasic);
  cputimer_stop("Copying to CPU and print");

  // zero the device result array
  CHECK(cudaMemset(deviceP, 0, sizeof(float) * N));

  /* Call the tiled kernel */
  cputimer_start();
  //@@  Define the execution configuration
  //@@  Run the 1D convolution kernel (tiled)
  convolution_1D_tiled<<<blocksPerGrid, THREADS_PER_BLOCK>>>(deviceN, deviceM, deviceP, N);
  CHECK(cudaGetLastError());
  cputimer_stop("Finished 1D convolution(tiled)");

  cputimer_start();
  //@@ INSERT CODE HERE
  cudaMemcpy(hostResTiles, deviceP, N*sizeof(float), cudaMemcpyDeviceToHost);
  // log results
  FILE *ftiled = fopen("convolution_tiled.txt", "w");
  for(int i = 0; i < N; i++){
      fprintf(ftiled, "%f\n", hostResTiles[i]);
  }
  fclose(ftiled);
  cputimer_stop("Copying to CPU and print");


  //@@ Validate the results from the two implementations
  int basicCompare = memcmp(hostP, hostResBasic, N*sizeof(float));
  int tiledCompare = memcmp(hostP, hostResTiles, N*sizeof(float));
  int deviceCompare = memcmp(hostResBasic, hostResTiles, N*sizeof(float));

  printf("memcmp results (0 being equal) between CPU and basic : %d\n", basicCompare);
  printf("memcmp results (0 being equal) between CPU and tiled : %d\n", tiledCompare);
  printf("memcmp results (0 being equal) between basic and tiled: %d\n", deviceCompare);

  cputimer_start();
  //@@ INSERT CODE HERE
  cudaFree(hostN);
  cudaFree(hostM);
  cudaFree(hostP);
  cudaFree(hostResBasic);
  cudaFree(hostResTiles);
  cudaFree(deviceN);
  cudaFree(deviceN);
  cudaFree(deviceP);

  cputimer_stop("Free memory resources");

  return 0;
}
