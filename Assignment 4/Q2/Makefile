NVCC       := nvcc
ARCH      ?= -arch=sm_75  # adjust to your GPU (e.g., sm_80 for A100, sm_90 for H100)
NVCCFLAGS  := -O3 -lineinfo $(ARCH)
LDFLAGS    := -lcusparse -lcublas
TARGET     := heat
SRC        := hw4-heat.cu
PREFETCH  ?= 1

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

# Example: make run DIMX=1024 NSTEPS=1000
run: $(TARGET)
	./$(TARGET) $(DIMX) $(NSTEPS) $(PREFETCH)

clean:
	rm -f $(TARGET) *.o
