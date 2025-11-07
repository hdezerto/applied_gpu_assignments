# Simple Makefile for Q2 - Vector Addition

NVCC = nvcc
TARGET = vectorAdd
SRC = vectorAdd.cu
ARCH = -arch=sm_75

all:
	$(NVCC) $(ARCH) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
