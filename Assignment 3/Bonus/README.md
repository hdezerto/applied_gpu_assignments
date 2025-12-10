# Bonus: iPIC3D-mini GPU Mover

## Contents
- `iPIC3D-mini/`: mini PIC codebase
  - `src/Particles.cu`: CPU and GPU (`mover_PC_gpu`) movers
  - `src/sputniPIC.cpp`: selects CPU/GPU mover via `USE_GPU_MOVER`
  - `inputfiles/GEM_2D.inp`: input used for runs
  - `Makefile`: build configuration (`ARCH` set for GPU)

## Build
CPU baseline (default mover):
```bash
make
```

GPU mover enabled:
```bash
make clean
make CXXFLAGS="-std=c++11 -I./include -O3 -g -Xcompiler -Wall -DUSE_GPU_MOVER" \
     NVCCFLAGS="-I./include -arch=sm_75 -std=c++11 -O3 -g -Xcompiler -Wall --compiler-bindir=g++ -DUSE_GPU_MOVER"
```
Adjust `-arch=sm_75` to match your GPU.

## Run
From `iPIC3D-mini`:
```bash
./bin/miniPIC.out inputfiles/GEM_2D.inp
```
Outputs are written under `data/`. Copy them aside for comparisons:
```bash
cp -r data data_cpu   # after CPU run
cp -r data data_gpu   # after GPU run
```
