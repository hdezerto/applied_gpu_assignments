Makefile provided has functionality to compile code by running 'make' which creates the run file `a.out` which takes an argument of the size of N. Example of runnign with 1024 elements, execute `./a.out 1024`. It will output two files named `convolution_basic.txt` and convolution_tiled.txt` with the output data.

`make clean` removes the executable file and the output files generated.

`run.sh` is made to generate profile file for NVIDIA visual profiler. It takes an argument of N and requires that the source code has been compiled first.
