Q2 – Vector Addition in CUDA
----------------------------

Files:
 - vectorAdd.cu : CUDA implementation with all //@@ comments.
 - Makefile     : to compile and run the code.

How to compile:
    make

How to run:
    ./vectorAdd

Description:
This program allocates host and device memory, transfers data to the GPU,
launches a kernel that adds two vectors, copies the result back to the CPU,
compares it with a CPU reference implementation, and prints the maximum error.
