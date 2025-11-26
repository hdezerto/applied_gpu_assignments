CUDA_DIR := /usr/local/cuda

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hw2_histogram_template.cu

release: $(SRC)
	mkdir -p data/debug
	mkdir -p data/normal
	mkdir -p data/uniform
	$(CC) $(SRC) -o a.out
debug: $(SRC)
	mkdir -p data/debug
	mkdir -p data/normal
	mkdir -p data/uniform
	$(CC) -g $(SRC) -o debug.out


clean: $(SRC)
	rm -f $(EXE) resultRef.txt hostBins.txt
