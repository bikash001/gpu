#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/find.h>
#include <iostream>
#include <utility>
using namespace std;

struct saxpy
{
	int *N;
	saxpy(int a) {
		cudaHostAlloc(&N, sizeof(int), 0);
		*N = a;
	}
	__host__ __device__
	bool operator() (const int a) {
		return a%(*N) == 0;
	}
};

__host__ __device__
int ordinal(const int N, const int x, const int y)
{
	return (N*x)+y;
}

void print(int *begin, int *end, int start, int diff, const int N)
{
	int *first = thrust::find(thrust::device, begin, end, 1);
	int *second = thrust::find(thrust::device, first+1, end, 1);
	
	int a = first - begin;
	int b = second - begin;
	int ordA = start + diff * a;
	int ordB = start + diff * b;

	cout << "NO" << endl;
	cout << ordA/N << " " << ordA % N << endl;
	cout << ordB/N << " " << ordB % N << endl;
}


int evaluate(int *A, int *B, int *C, const int N)
{
	int *begin, *end;
	// row
	for (int i=0; i<N; ++i) {
		begin = A+ordinal(N, i, 0);
		end = A+1+ordinal(N,i,N-1);
		if (thrust::count(thrust::device, begin, end, 1) > 1) {
			print(begin, end, begin-A, 1, N);
			return 0;
		}
	}

	// column
	thrust::sequence(thrust::device, B, B+N*N);
	for (int i=0; i<N; ++i) {
		begin = A+ordinal(N,0,i);
		end = A+1+ordinal(N,N-1,i);
		int *r_end = thrust::copy_if(thrust::device, begin, end, B, C, saxpy(N));
		if (thrust::count(thrust::device, C, r_end, 1) > 1) {
			print(C, r_end, begin-A, N, N);
			return 0;
		}
	}
	
	// diagonal 1
	for (int i=1; i<N; ++i) {
		begin = A+ordinal(N,0,i);
		end = A+1+ordinal(N,i,0);
		int *r_end = thrust::copy_if(thrust::device, begin, end, B, C, saxpy(N-1));
		if (thrust::count(thrust::device, C, r_end, 1) > 1) {
			print(C, r_end, begin-A, N-1, N);
			return 0;
		}
	}
	for (int i=1; i<N-1; ++i) {
		begin = A+ordinal(N,i,N-1);
		end = A+1+ordinal(N,N-1,i);
		int *r_end = thrust::copy_if(thrust::device, begin, end, B, C, saxpy(N-1));
		if (thrust::count(thrust::device, C, r_end, 1) > 1) {
			print(C, r_end, begin-A, N-1, N);
			return 0;
		}
	}
	
	//diagonal 2
	for (int i=0; i<N-1; ++i) {
		begin = A+ordinal(N,0,i);
		end = A+1+ordinal(N,N-i-1,N-1);
		int *r_end = thrust::copy_if(thrust::device, begin, end, B, C, saxpy(N+1));
		if (thrust::count(thrust::device, C, r_end, 1) > 1) {
			print(C, r_end, begin-A, N+1, N);
			return 0;
		}
	}
	for (int i=1; i<N-1; ++i) {
		begin = A+ordinal(N,i,0);
		end = A+1+ordinal(N,N-1,N-i-1);
		int *r_end = thrust::copy_if(thrust::device, begin, end, B, C, saxpy(N+1));
		if (thrust::count(thrust::device, C, r_end, 1) > 1) {
			print(C, r_end, begin-A, N+1, N);
			return 0;
		}
	}

	return 1;
}

int main(int argc, char const *argv[])
{
	int N;
	cin >> N;

	int *cA = new int[N*N];

	for (int i=0; i<N*N; ++i) {
		cin >> cA[i];
	}
	
	int *A, *B, *C;
	cudaMalloc(&A, sizeof(int)*N*N);
	cudaMalloc(&B, sizeof(int)*N*N);
	cudaMalloc(&C, sizeof(int)*N);
	cudaMemcpy(A, cA, sizeof(int)*N*N, cudaMemcpyHostToDevice);
	
	if (evaluate(A, B, C, N)) {
		cout << "YES" << endl;
	}

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	delete[] cA;

	return 0;
}