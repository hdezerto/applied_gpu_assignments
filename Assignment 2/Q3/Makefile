# Makefile for Assignment 2 - Q3 (Tiled GEMM)

NVCC ?= nvcc
ARCH ?= -arch=sm_75
TARGET := matrixMul
SRC := matrixMul.cu
NVCCFLAGS ?= -O3 -std=c++17

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe
