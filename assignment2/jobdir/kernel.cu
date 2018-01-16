#include <stdio.h>

__device__ volatile unsigned int count = 0;

/* The kernel "msort" should sort the input array using parallel merge-sort. */

__global__ void msort(int *d_input, int* d_temp, int N)
{

	const int id = blockIdx.x * blockDim.x + threadIdx.x;

	const int c = (int)ceilf(log2f(N));
	int top = 0;
	int upper_bound = (int)ceilf((float)N/2);

	for (int i=0; i<c; ++i) {
		if (id < upper_bound) {
			int j = id*(1 << (i+1));
			int diff = (1 << i);
			int j_max = j+diff;
			int k = j_max;
			int k_max = k+diff;
			int t_cnt = j;

			if (j_max < N) {
				if (k_max > N) {
					k_max = N;
				}

				while (j < j_max && k < k_max) {
					if (d_input[j] < d_input[k]) {
						d_temp[t_cnt++] = d_input[j++];
					} else {
						d_temp[t_cnt++] = d_input[k++];
					}
				}

				while (j < j_max) {
					d_temp[t_cnt++] = d_input[j++];
				}

				while (k < k_max) {
					d_temp[t_cnt++] = d_input[k++];
				}

				for (int cnt = id*(1 << (i+1)); cnt < k_max; ++cnt) {
					d_input[cnt] = d_temp[cnt];
				}
			}
			__threadfence();
			atomicInc((unsigned int*)(&count), 1 << 30);
			top += upper_bound;
			while (count < top);
			
			upper_bound = (int)ceilf((float)upper_bound/2);
		}
	}
	if (id == 0) {
		count = 0;
	}
}
