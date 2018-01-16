flag=-Wno-deprecated-gpu-targets

main: kernel.o main.o
	nvcc $(flag) kernel.o main.o -o msort

kernel.o: kernel.cu kernel.h
	nvcc $(flag) -c kernel.cu

main.o: main.cu
	nvcc $(flag) -c main.cu

clean: 
	rm -rf *.o msort
