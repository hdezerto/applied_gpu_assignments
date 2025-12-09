# Simple Makefile for Q2 - Vector Addition

NVCC := nvcc
TARGET := vectorAdd
SRC := vectorAdd.cu
# Adjust the architecture to your GPU if needed
ARCH ?= -arch=sm_75

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

run_streamed: $(TARGET)
	./$(TARGET) --streamed

run_sweepN: $(TARGET)
	./$(TARGET) --sweepN

run_streamed_sweepN: $(TARGET)
	./$(TARGET) --streamed --sweepN

run_streamed_sweepSseg: $(TARGET)
	./$(TARGET) --streamed --sweepSseg

clean:
	rm -f $(TARGET)
