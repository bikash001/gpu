#include <stdio.h>
#include "kernels.h"
#include <stdlib.h>
#include <cuda.h>

void printarr(const int *arr, int M, int N);

int compare(const int *a, const int *b, int size)
{
	for (int i=0; i<size; ++i) {
		if (a[i] != b[i]) {
			return 1;
		}
	}
	return 0;
}

void chistogram(const int *in, int *out, int M, int N, int count)
{
	memset(out, 0, sizeof(int)*count);
	for (int i=0; i<M*N; ++i) {
		++out[in[i]%count];
	}
}

void cstencil(const int *in, int *out, int M, int N)
{
	memcpy(out, in, sizeof(int)*M*N);
	int id;
	for (int i=1; i<M-1; ++i) {
		for (int j=1; j<N-1; ++j) {
			id = i*N+j;
			out[id] = 0.2 * (in[id]+in[id+1]+in[id-1]+in[id+N]+in[id-N]);
		}
	}
}

int comp(const int *in, const int *out, int M, int N)
{
	for (int i=1; i<M-1; ++i) {
		for (int j=1; j<N-1; ++j) {
			int id = i*N+j;
			int sum = 0.2 * (in[id]+in[id+1]+in[id-1]+in[id+N]+in[id-N]);
			if (sum != out[j]) {
				return 1;
			}
		}
	}
	for (int i=0; i<N; ++i) {
		if (out[i] != in[i])
			return 1;
	}
	for (int i=(M-1)*N; i<M*N; ++i) {
		if (out[i] != in[i])
			return 1;
	}
	for (int i=0; i<M; ++i) {
		if (out[i*N] != in[i*N])
			return 1;
		if (out[i*N+N-1] != in[i*N+N-1])
			return 1;
	}
	return 0;
}

void cupdate(const int *in, int *out, int M, int N)
{
	memcpy(out, in, sizeof(int)*M*N);
	for (int i=0; i<N; ++i) {
		out[i] = 1;
	}
	for (int i=(M-1)*N; i<M*N; ++i) {
		out[i] = 1;
	}
	for (int i=0; i<M; ++i) {
		out[i*N] = 1;
		out[i*N+N-1] = 1;
	}
}

void printarr(const int *arr, int M, int N)
{
	for (int i=0; i<M; ++i) {
		for (int j=0; j<N; ++j) {
			printf("%d ", arr[i*N+j]);
		}
		puts("");
	}
	puts("");
}

int main()
{
	//histogram
	int *dinput, *dbin, M, N;
	M = 2;
	N = 1024;
	//scanf("%d %d", &M, &N);
	const int bincount = 56;
	int *cinp = (int *)malloc(sizeof(int)*M*N);
	int *cbin = (int*)malloc(sizeof(int)*bincount);
	int *hout = (int*)malloc(sizeof(int)*bincount);
	for (int i=0; i<M*N; ++i) {
		cinp[i] = rand()%100000;
	}
	// printarr(cinp, M, N);
	
	// printf("%s\n", "running");
	cudaMalloc(&dinput, sizeof(int)*M*N);
	cudaMemcpy(dinput, cinp, sizeof(int)*M*N, cudaMemcpyHostToDevice);
	cudaMalloc(&dbin, sizeof(int)*bincount);
	cudaMemset(dbin, 0, sizeof(int)*bincount);

	histogram<<<N, M>>>(dinput, dbin, M, N, bincount);
	chistogram(cinp, cbin, M, N, bincount);
	cudaMemcpy(hout, dbin, sizeof(int)*bincount, cudaMemcpyDeviceToHost);

	if (compare(hout, cbin, bincount)==0) {
		printf("Success histogram\n");
	} else {
		printf("Failure histogram\n");
	}
	free(cbin);
	free(hout);
	cudaFree(dbin);
	
	hout = (int*)malloc(sizeof(int)*M*N);
	cbin = (int*)malloc(sizeof(int)*M*N);
	dim3 p(M/2, N/512);
	dim3 q(2,512);
	stencil<<<p, q>>>(dinput, M, N);
	// cudaDeviceSynchronize();
	cstencil(cinp, cbin, M, N);
	cudaMemcpy(hout, dinput, sizeof(int)*M*N, cudaMemcpyDeviceToHost);
	if (compare(hout, cbin, M*N)==0) {
		printf("Success stencil\n");
	} else {
		printf("Failure stencil\n");
		printarr(cinp, M, N);
		printarr(cbin, M, N);
		printarr(hout, M, N);
	}

	hout = (int*)malloc(sizeof(int)*M*N);
	cbin = (int*)malloc(sizeof(int)*M*N);
	cudaMemcpy(dinput, cinp, sizeof(int)*M*N, cudaMemcpyHostToDevice);
	updateBC<<<p, q>>>(dinput, M, N);
	cupdate(cinp, cbin, M, N);
	cudaMemcpy(hout, dinput, sizeof(int)*M*N, cudaMemcpyDeviceToHost);
	if (compare(hout, cbin, M*N)==0) {
		printf("Success\n");
	} else {
		printf("Failure\n");
	}

	free(hout);
	free(cbin);
	cudaFree(dinput);
	free(cinp);
}
