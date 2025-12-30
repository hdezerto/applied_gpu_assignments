NVCC ?= nvcc
ARCH ?= sm_75
TARGET ?= wmma_bonus
SRC := wmma_bonus.cu

NVCCFLAGS := -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -O3

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET) *.csv
