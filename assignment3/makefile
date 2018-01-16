flag=-Wno-deprecated-gpu-targets

main: kernels.o main.o
	nvcc $(flag) kernels.o main.o -o main

kernels.o: kernels.cu kernels.h
	nvcc $(flag) -c kernels.cu

main.o: main.cu
	nvcc $(flag) -c main.cu

clean: 
	rm -rf *.o main
