# Makefile for Q2 - Reduction variants

NVCC := nvcc
SRC := reduction.cu
ARCH := -arch=sm_75

BASE := reduction_base
DUAL := reduction_twoload
SHARED := reduction_shared
SHARED_ATOMIC := reduction_shared_atomic
FULL := reduction_full

.PHONY: all clean run run-base run-dual run-shared run-shared-atomic run-full

all: $(BASE) $(DUAL) $(SHARED) $(SHARED_ATOMIC) $(FULL)

$(BASE): $(SRC)
	$(NVCC) $(ARCH) $(SRC) -o $@ -DOPT_TWO_LOADS=0 -DOPT_SHARED_REDUCTION=0 -DOPT_BLOCK_ATOMIC=0

$(DUAL): $(SRC)
	$(NVCC) $(ARCH) $(SRC) -o $@ -DOPT_TWO_LOADS=1 -DOPT_SHARED_REDUCTION=0 -DOPT_BLOCK_ATOMIC=0

$(SHARED): $(SRC)
	$(NVCC) $(ARCH) $(SRC) -o $@ -DOPT_TWO_LOADS=0 -DOPT_SHARED_REDUCTION=1 -DOPT_BLOCK_ATOMIC=0

$(SHARED_ATOMIC): $(SRC)
	$(NVCC) $(ARCH) $(SRC) -o $@ -DOPT_TWO_LOADS=0 -DOPT_SHARED_REDUCTION=1 -DOPT_BLOCK_ATOMIC=1

$(FULL): $(SRC)
	$(NVCC) $(ARCH) $(SRC) -o $@

run: run-full

run-base: $(BASE)
	./$(BASE)

run-dual: $(DUAL)
	./$(DUAL)

run-shared: $(SHARED)
	./$(SHARED)

run-shared-atomic: $(SHARED_ATOMIC)
	./$(SHARED_ATOMIC)

run-full: $(FULL)
	./$(FULL)

clean:
	rm -f $(BASE) $(DUAL) $(SHARED) $(SHARED_ATOMIC) $(FULL)
