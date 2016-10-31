
all: bench_cublas bench_clblas

bench_cublas: bench_cublas.c
	gcc -o $@ -O2 -std=c99 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 $^ -lcublas -lcudart

bench_clblas: bench_clblas.c
	gcc -o $@ -O2 -std=c99 -I/opt/AMDAPPSDK-3.0/include -I/data/hishiko0/hidaka/clBLAS/src -L/data/hishiko0/hidaka/clBLAS/src/build/library $^ -lOpenCL -lclBLAS
