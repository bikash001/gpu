__device__ volatile int uc = 1;
__device__ volatile unsigned int counter = 0;
__device__ volatile unsigned int cnt = 1;


__global__ void histogram(int *d_input, int* d_bin, int M, int N, int BIN_COUNT)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < M*N) {
		int bid = d_input[id] % BIN_COUNT;
		atomicAdd(&(d_bin[bid]), 1);
	}
}

__global__ void updateBC(int* d_input, int M, int N)
{
 	int old = atomicAdd((int *)(&uc), 1);
 	if (N==1 || M==1) {
 		int mx = (N < M)? M:N;
 		if (old <= mx) {
 			d_input[old] = 1;
 		}
 	} else if (old <= 2*(N+M) - 4) {
 		int id;
 		if (old <= N) {
 			id = old;
 		} else if (old <= N+M-2) {
 			id = 1+N*(old-N);
 		} else if (old <= N+2*(M-2)) {
 			id = N+(old-(N+M-2))*N;
 		} else {
 			id = N*(M-1) + old-(N+2*(M-2));
 		}
 		d_input[id-1] = 1;
 	}
 	if (old == blockDim.x*gridDim.x*blockDim.y*gridDim.y) {
 		uc = 1;
 	}
}


__global__ void stencil(int* d_input, const int M, const int N)
{
	int old = atomicInc((unsigned int*)(&counter), 1<<30);
 	const int max = (N-2)*(M-2);
 	if (old < max) {
 		int i = old % (N-2);
 		int j = old / (N-2);
 		int id = N*(j+1)+i+1;
 		int sum = 0.2 * (d_input[id]+d_input[id+1]+d_input[id-1]+d_input[id+N]+d_input[id-N]);
 		atomicInc((unsigned int*)(&cnt), 1<<30);
 		
 		while (cnt < max);
 		d_input[id] = sum;
 		if (old+1 == max) {
 			cnt = 1;
 		}
 	}
 	if (old+1 == blockDim.x*gridDim.x*blockDim.y*gridDim.y) {
 		counter = 0;
 	}
}
