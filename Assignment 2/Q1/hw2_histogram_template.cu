
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <endian.h>
#include <random>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <cuda_runtime.h>

#define NUM_BINS 4096
#define N 256 // threads per block
#define CUDAHOSTALLOC 1 // 1 for using CUDA host memory allocation, 0 for using malloc

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    //@@ Insert code below to compute histogram of input using shared memory and atomics
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    int s_i = threadIdx.x;
    __shared__ int s_input[N];
    s_input[s_i] = input[i];
    __syncthreads();

    int number = s_input[s_i];
    atomicInc(&bins[number], UINT_MAX);
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_bins && bins[i] > 127){
        bins[i] = 127;
    }
}


int main(int argc, char **argv) {

  // strings used for logging histograms
  std::string randomType;

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args

  if(argc < 3){
      printf("missing input length variable and/or choice of input value distribution\n");
      printf("{input length variable} {0,1,2} where:");
      printf("0: uniform distribution");
      printf("1: normal distribution");
      printf("2: debug input");
      return 1;
  }
  inputLength = atoi(argv[1]);

  //@@ Insert code below to allocate Host memory for input and output
  // //@ malloc method
  if(!CUDAHOSTALLOC){
    hostInput = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    resultRef = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
  }
  //@ cudaHostAlloc method
  else{
    cudaHostAlloc((void**)&hostInput, inputLength*sizeof(int), cudaHostRegisterDefault);
    cudaHostAlloc((void**)&hostBins, NUM_BINS*sizeof(int), cudaHostRegisterDefault);
    cudaHostAlloc((void**)&resultRef, NUM_BINS*sizeof(int), cudaHostRegisterDefault);
  }

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  // uniform rand distribution

  int inputChoice = atoi(argv[2]);
  if(inputChoice == 0){

    for(int i = 0; i < inputLength; i++){
        hostInput[i] = rand() % NUM_BINS;
    }
    randomType = "uniform";

  }


  // normal rand distribution
  else if(inputChoice == 1){

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.5,0.1);

    for(int i = 0; i < inputLength; i++){
        double mult = distribution(generator);
        int value = mult * NUM_BINS;
        // fix outliers
        if(value < 0) value = 0;
        else if (value > NUM_BINS-1) value = NUM_BINS -1;
        hostInput[i] = value;
    }

    randomType = "normal";

  }


  // all zeroes
  else if(inputChoice == 2){
      memset(hostInput, 0, inputLength);
      randomType = "debug";
  }


  //@@ Insert code below to create reference result in CPU

  memset(resultRef, 0, NUM_BINS); // array zeroed
  for(int i = 0; i < inputLength; i++){
      int number = hostInput[i];
      if (resultRef[number] < 127){
          resultRef[number]++;
      }
  }

  // log results
  FILE *f = fopen("resultRef.txt", "w");
  for(int i = 0; i < NUM_BINS; i++){
      fprintf(f, "%d\n", resultRef[i]);
  }
  fclose(f);

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  //@@ Initialize the grid and block dimensions here
  // 1D computation grid
  dim3 HistogramGrid((inputLength-1)/N + 1, 1, 1);
  dim3 HistogramBlock(N, 1, 1);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<HistogramGrid, HistogramBlock>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

  //@@ Initialize the second grid and block dimensions here
  dim3 convertGrid((NUM_BINS-1)/N + 1, 1, 1);
  dim3 convertBlock(N, 1, 1);


  //@@ Launch the second GPU Kernel here
  cudaDeviceSynchronize();
  convert_kernel<<<convertGrid, convertBlock>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  printf("memcmp results (0 being equal): %d\n" , memcmp(resultRef, hostBins, NUM_BINS));

  // log host bins

  std::string file = "data/" + randomType + "/" + argv[1] + ".csv";
  FILE *ff = fopen(file.c_str(), "w");
  for(int i = 0; i < NUM_BINS; i++){
      fprintf(ff, "%d, %d\n", i, hostBins[i]);
  }
  fclose(ff);


  //@@ Free the GPU memory here
  cudaFree(deviceBins);
  cudaFree(deviceInput);

  //@@ Free the CPU memory here
  //@ Malloc method
  if(!CUDAHOSTALLOC){
    free(hostBins);
    free(hostInput);
    free(resultRef);
  }
  //@ cudaHostAlloc method
  else{
    cudaFree(hostBins);
    cudaFree(hostInput);
    cudaFree(resultRef);
  }
  return 0;
}
