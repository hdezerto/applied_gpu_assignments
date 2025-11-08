# Simple Makefile for Q3 - Matrix Multiplication

NVCC = nvcc
TARGET = matrixMul
SRC = matrixMul.cu
ARCH = -arch=sm_75

all:
	$(NVCC) $(ARCH) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
