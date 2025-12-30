CUDA_DIR := /usr/local/cuda

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hw4-convolution.cu

release: $(SRC)
	$(CC) $(SRC) -o a.out

clean: $(SRC)
	rm -f a.out convolution_basic.txt convolution_tiled.txt
