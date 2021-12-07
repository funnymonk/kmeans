Usage directions :
============================

The current folder has 2 implementations of the KMeans clustering algorithm:
1.) 8d-example.c : This implements a sequential KMeans algorithm
2.) 8d-example-simd.c : This implements a SIMD KMeans algorithms, and contains the kernels used in the project. 

To make the various C implementations, run :
make

To clean up:
make clean

The pwd also has a directory called scipy_test, with a file called test.py.
To run this implementation:
cd scipy_test
python3 test.py

NOTE : This requires the scikit-learn python3 module to be available.
 
Run:
=============================
The implementations were run on ece009 machine

./8d-example : This runs the sequential implementation
./8d-example-simd : This runs the SIMD implementation



