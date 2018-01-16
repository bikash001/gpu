flag=-Wno-deprecated-gpu-targets

main: CS14B039.o main.o
	nvcc $(flag) CS14B039.o main.o -o msort

CS14B039.o: CS14B039.cu kernels.h
	nvcc $(flag) -c CS14B039.cu

main.o: main.cu
	nvcc $(flag) -c main.cu

clean: 
	rm -rf *.o msort
