Makefile creates the necessary data directories and the executable file a.out
a.out needs two parameters, input size and choice of random distribution. Random distribution choices are uniform (0), normal (1), and debug (2) which is only zeroes. To run the histogram application with 1024 inputs and uniform random distribution, the command used is
```
./a.out 1024 0
```

Data is saved under ./data/{random distribution choice} and graphs can be generated with the python files `normalplot.py` and `uniformplot.py` which require the `matplotlib` library available through pip. 