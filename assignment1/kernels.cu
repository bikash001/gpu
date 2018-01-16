#include <cuda_runtime.h>
#include "kernels.h"
#include <stdio.h>

__global__ void transpose_parallel_per_row(float *in, float *out, int rows_in, int cols_in)
{
	int index = (blockIdx.x)*blockDim.x + threadIdx.x;


	if (index < rows_in) {
		int i = index * cols_in;
		int j = index;
		int k=0;
		for (k=0; k < cols_in; ++k) {
			out[j] = in[i+k];
			j += rows_in;
		}
	}
}

__global__ void 
transpose_parallel_per_element(float *in, float *out, int rows_in, int cols_in, int K1, int K2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < rows_in && j < cols_in) {
    	out[j*rows_in+i] = in[i*cols_in+j];
    }
}
