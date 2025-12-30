#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
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


// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX,
    double alpha) {
  // Stencil from the finete difference discretization of the equation
  double stencil[] = { 1, -2, 1 };
  // Variable holding the position to insert a new element
  size_t ptr = 0;
  // -----vvvvv Different from original template vvvvv-----
  // CSR format requires first row pointer to start at zero
  ArowPtr[0] = 0;
  // -----^^^^^ Different from original template ^^^^^-----
  // Insert a row of zeros at the beginning of the matrix
  ArowPtr[1] = ptr;
  // Fill the non zero entries of the matrix
  for (int i = 1; i < (dimX - 1); ++i) {
    // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
    for (int k = 0; k < 3; ++k) {
      // Set the value for A[i][i+k-1]
      A[ptr] = stencil[k];
      // Set the column index for A[i][i+k-1]
      AcolIndx[ptr++] = i + k - 1;
    }
    // Set the number of newly added elements
    ArowPtr[i + 1] = ptr;
  }
  // Insert a row of zeros at the end of the matrix
  ArowPtr[dimX] = ptr;
}

int main(int argc, char **argv) {
  int device = 0;            // Device to be used
  int dimX;                  // Dimension of the metal rod
  int nsteps;                // Number of time steps to perform
  int prefetchFlag = 1;      // Enable prefetch by default (can disable via arg)
  double alpha = 0.4;        // Diffusion coefficient
  double* temp;              // Array to store the final time step
  double* A;                 // Sparse matrix A values in the CSR format
  int* ARowPtr;              // Sparse matrix A row pointers in the CSR format
  int* AColIndx;             // Sparse matrix A col values in the CSR format
  int nzv;                   // Number of non zero values in the sparse matrix
  double* tmp;               // Temporal array of dimX for computations
  size_t bufferSize = 0;     // Buffer size needed by some routines
  void* buffer = nullptr;    // Buffer used by some routines in the libraries
  int concurrentAccessQ;     // Check if concurrent access flag is set
  double zero = 0;           // Zero constant
  double one = 1;            // One constant
  double norm;               // Variable for norm values
  double error;              // Variable for storing the relative error
  // --- NEW for question 1 ---
  float csrmvMsTotal = 0.f;  // Accumulated csrmv time in ms for FLOP calc
  int itersDone = 0;         // Number of csrmv iterations actually run
  cudaEvent_t evStart, evStop; // Events to time csrmv
  // --------------------------
  double tempLeft = 200.;    // Left heat source applied to the rod
  double tempRight = 300.;   // Right heat source applied to the rod
  cublasHandle_t cublasHandle;      // cuBLAS handle
  cusparseHandle_t cusparseHandle;  // cuSPARSE handle
  cusparseMatDescr_t Adescriptor;   // Mat descriptor needed by cuSPARSE
  cusparseSpMatDescr_t matA;        // cuSPARSE sparse matrix descriptor (SpMV API)
  cusparseDnVecDescr_t vecX, vecY;  // cuSPARSE dense vector descriptors

  // Read the arguments from the command line
  dimX = atoi(argv[1]);
  nsteps = atoi(argv[2]);
  if (argc > 3) {
    prefetchFlag = atoi(argv[3]) != 0;
  }

  // Print input arguments
  printf("The X dimension of the grid is %d \n", dimX);
  printf("The number of time steps to perform is %d \n", nsteps);

  // Get if the cudaDevAttrConcurrentManagedAccess flag is set
  gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));
  // Optional override to disable prefetch via third argument
  if (!prefetchFlag) {
    concurrentAccessQ = 0;
  }

  // Calculate the number of non zero values in the sparse matrix. This number
  // is known from the structure of the sparse matrix
  nzv = 3 * dimX - 6;

  //@@ Insert the code to allocate the temp, tmp and the sparse matrix
  //@@ arrays using Unified Memory
  cputimer_start();
  gpuCheck(cudaMallocManaged(&temp, dimX * sizeof(double)));
  gpuCheck(cudaMallocManaged(&tmp, dimX * sizeof(double)));
  gpuCheck(cudaMallocManaged(&A, nzv * sizeof(double)));
  gpuCheck(cudaMallocManaged(&ARowPtr, (dimX + 1) * sizeof(int)));
  gpuCheck(cudaMallocManaged(&AColIndx, nzv * sizeof(int)));
  cputimer_stop("Allocating device memory");

  // Check if concurrentAccessQ is non zero in order to prefetch memory
  if (concurrentAccessQ) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to CPU
    gpuCheck(cudaMemPrefetchAsync(A, nzv * sizeof(double), cudaCpuDeviceId, NULL));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), cudaCpuDeviceId, NULL));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), cudaCpuDeviceId, NULL));
    gpuCheck(cudaMemPrefetchAsync(temp, dimX * sizeof(double), cudaCpuDeviceId, NULL));
    gpuCheck(cudaMemPrefetchAsync(tmp, dimX * sizeof(double), cudaCpuDeviceId, NULL));
    cputimer_stop("Prefetching GPU memory to the host");
  }

  // Initialize the sparse matrix
  cputimer_start();
  matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
  cputimer_stop("Initializing the sparse matrix on the host");

  //Initiliaze the boundary conditions for the heat equation
  cputimer_start();
  memset(temp, 0, sizeof(double) * dimX);
  temp[0] = tempLeft;
  temp[dimX - 1] = tempRight;
  cputimer_stop("Initializing memory on the host");

  if (concurrentAccessQ) {
    cputimer_start();
    //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
    gpuCheck(cudaMemPrefetchAsync(A, nzv * sizeof(double), device, NULL));
    gpuCheck(cudaMemPrefetchAsync(ARowPtr, (dimX + 1) * sizeof(int), device, NULL));
    gpuCheck(cudaMemPrefetchAsync(AColIndx, nzv * sizeof(int), device, NULL));
    gpuCheck(cudaMemPrefetchAsync(temp, dimX * sizeof(double), device, NULL));
    gpuCheck(cudaMemPrefetchAsync(tmp, dimX * sizeof(double), device, NULL));
    cputimer_stop("Prefetching GPU memory to the device");
  }

  //@@ Insert code to create the cuBLAS handle
  cublasCheck(cublasCreate(&cublasHandle));

  //@@ Insert code to create the cuSPARSE handle
  cusparseCheck(cusparseCreate(&cusparseHandle));

  //@@ Insert code to set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST
  cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

  //@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
  cusparseCheck(cusparseCreateMatDescr(&Adescriptor));
  cusparseCheck(cusparseSetMatType(Adescriptor, CUSPARSE_MATRIX_TYPE_GENERAL));
  cusparseCheck(cusparseSetMatIndexBase(Adescriptor, CUSPARSE_INDEX_BASE_ZERO));

    // Create SpMV descriptors (modern API, available on Colab toolkits)
    cusparseCheck(cusparseCreateCsr(
      &matA,
      dimX,
      dimX,
      nzv,
      ARowPtr,
      AColIndx,
      A,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F));

    cusparseCheck(cusparseCreateDnVec(&vecX, dimX, temp, CUDA_R_64F));
    cusparseCheck(cusparseCreateDnVec(&vecY, dimX, tmp, CUDA_R_64F));

    // Query buffer size for SpMV
    cusparseCheck(cusparseSpMV_bufferSize(
      cusparseHandle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one,
      matA,
      vecX,
      &zero,
      vecY,
      CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT,
      &bufferSize));

    gpuCheck(cudaMallocManaged(&buffer, bufferSize));

  // --- NEW for question 1 ---
  // Create CUDA events for timing the csrmv (SMPV) to estimate FLOPs
  gpuCheck(cudaEventCreate(&evStart));
  gpuCheck(cudaEventCreate(&evStop));
  // --------------------------

  // Time the full time-stepping loop for wall-clock comparison
  cputimer_start();

  // Perform the time step iterations
  for (int it = 0; it < nsteps; ++it) {
    //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
    //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
    //@@ tmp = 1 * A * temp + 0 * tmp
    gpuCheck(cudaEventRecord(evStart)); // NEW for question 1
    cusparseCheck(cusparseSpMV(
      cusparseHandle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one,
      matA,
      vecX,
      &zero,
      vecY,
      CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT,
      buffer));
    // --- NEW for question 1 ---
    gpuCheck(cudaEventRecord(evStop));
    gpuCheck(cudaEventSynchronize(evStop));
    float iterMs = 0.f;
    gpuCheck(cudaEventElapsedTime(&iterMs, evStart, evStop));
    csrmvMsTotal += iterMs;
    ++itersDone;
    // --------------------------

    //@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
    //@@ This calculation corresponds to: temp = alpha * tmp + temp
    cublasCheck(cublasDaxpy(cublasHandle, dimX, &alpha, tmp, 1, temp, 1));

    //@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
    //@@ This calculation corresponds to: ||temp||
    cublasCheck(cublasDnrm2(cublasHandle, dimX, temp, 1, &norm));

    // If the norm of A*temp is smaller than 10^-4 exit the loop
    if (norm < 1e-4)
      break;
  }
  gpuCheck(cudaDeviceSynchronize());
  cputimer_stop("Time stepping loop");

  // --- NEW for question 1 ---
  // Report average csrmv time and achieved FLOPs for SMPV
  if (itersDone > 0 && csrmvMsTotal > 0.0f) {
    double flops = (2.0 * nzv * static_cast<double>(itersDone)) / (csrmvMsTotal / 1e3);
    double gflops = flops / 1e9;
    printf("SMPV: %d iterations, avg %.3f ms, throughput %.3f GFLOPS\n",
        itersDone, csrmvMsTotal / itersDone, gflops);
  }
  // --------------------------

  // Calculate the exact solution using thrust
  thrust::device_ptr<double> thrustPtr(tmp);
  thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
      (tempRight - tempLeft) / (dimX - 1));

  //  -----vvvvv Different from original template vvvvv-----

  // --------------------------------------------------------------------
  // IMPROVEMENT: Calculate the norm of the EXACT solution right now.
  // @@ We store it in 'norm' to use as the denominator (||exact||) later.
  // This is mathematically correct and saves it before we overwrite 'tmp'.
  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &norm));
  // --------------------------------------------------------------------

  // @@ Calculate the difference: tmp = -1 * temp + tmp
  // Now 'tmp' holds the error vector, and the exact solution is overwritten.
  one = -1;
  cublasCheck(cublasDaxpy(cublasHandle, dimX, &one, temp, 1, tmp, 1));

  // @@Calculate the norm of the error vector: ||temp - exact||
  cublasCheck(cublasDnrm2(cublasHandle, dimX, tmp, 1, &error));

  // NOTE: We do NOT need to calculate the norm of 'temp' anymore because
  // we already calculated the norm of the exact solution in the first step.

  // Calculate the relative error: ||error|| / ||exact||
  error = error / norm;
  printf("The relative error of the approximation is %f\n", error);

  //  -----^^^^^ Different from original template ^^^^^-----

  //@@ Insert the code to destroy the mat descriptor
  cusparseCheck(cusparseDestroyMatDescr(Adescriptor));

  // Destroy SpMV descriptors
  cusparseCheck(cusparseDestroySpMat(matA));
  cusparseCheck(cusparseDestroyDnVec(vecX));
  cusparseCheck(cusparseDestroyDnVec(vecY));

  //@@ Insert the code to destroy the cuSPARSE handle
  cusparseCheck(cusparseDestroy(cusparseHandle));

  //@@ Insert the code to destroy the cuBLAS handle
  cublasCheck(cublasDestroy(cublasHandle));

  // --- NEW for question 1 ---
  // Destroy CUDA events
  gpuCheck(cudaEventDestroy(evStart));
  gpuCheck(cudaEventDestroy(evStop));
  // --------------------------

  //@@ Insert the code for deallocating memory
  gpuCheck(cudaFree(temp));
  gpuCheck(cudaFree(tmp));
  gpuCheck(cudaFree(A));
  gpuCheck(cudaFree(ARowPtr));
  gpuCheck(cudaFree(AColIndx));
  gpuCheck(cudaFree(buffer));

  return 0;
}