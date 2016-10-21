all: libdettmers_weight_compression.so

libdettmers_weight_compression.so : dettmers_weight_compression.o
	gcc -shared -fopenmp -Wl,-soname,libdettmers_weight_compression.so -o libdettmers_weight_compression.so dettmers_weight_compression.o

dettmers_weight_compression.o : dettmers_weight_compression.c
	gcc -O3 -std=c99 -fopenmp -c -fPIC dettmers_weight_compression.c -o dettmers_weight_compression.o

clean :
	-rm -vf libdettmers_weight_compression.so dettmers_weight_compression.o dettmers_weight_compression.pyc
