#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <iostream>
using namespace std;

struct saxpy
{
	const int N, i;
	saxpy(int a, int b): N(a), i(b) {}
	__host__ __device__
	bool operator() (const int a) {
		return a%N == i;
	}
};

int index(const int N, const int x, const int y)
{
	return (N*x)+y;
}

int main(int argc, char const *argv[])
{
	cout << "init" << endl;
	int N = 4;
	int A[N*N];//{0,1,0,0,	0
				//0,0,0,1,	4
				//1,0,0,0,	8
				//0,0,1,0};	12
	memset(A, 0, sizeof(int)*N*N);
	int a, b;
	cin >> a >> b;
	A[a] = A[b] = 1;
	int B[N*N];
	int C[N];
	// int D[N*N];

	cout << "row" << endl; 
	// row
	for (int i=0; i<N; ++i) {
		if (thrust::count(A+i*N, A+(i+1)*N, 1) > 1) {
			cout << "NO 1" << endl;
			return 0;
		}
	}

	cout << "column" << endl;
	// column
	thrust::sequence(B, B+N*N);
	for (int i=0; i<N; ++i) {
		int *r_end = thrust::copy_if(A, A+N*N, B, C, saxpy(N,i));
		if (thrust::count(C, r_end, 1) > 1) {
			cout << "NO 2" << endl;
			return 0;
		}
	}

	// diagonal 1
	for (int i=1; i<N; ++i) {
		int *r_end = thrust::copy_if(A+i, A+N*i+1, B, C, saxpy(N-1,0));
		if (thrust::count(C, r_end, 1) > 1) {
			cout << "NO 3" << endl;
			return 0;
		}
	}
	for (int i=1; i<N-1; ++i) {
		int *r_end = thrust::copy_if(A+(N-1)+N*i, A+N*i+(N-1)*(N-i)+1, B, C, saxpy(N-1,0));
		if (thrust::count(C, r_end, 1) > 1) {
			cout << "NO 4" << endl;
			return 0;
		}
	}

	//diagonal 2
	for (int i=0; i<N-1; ++i) {
		int *r_end = thrust::copy_if(A+i, A+i+(N+1)*(N-i-1)+1, B, C, saxpy(N+1,0));
		if (thrust::count(C, r_end, 1) > 1) {
			cout << "NO 5" << endl;
			return 0;
		}
	}
	for (int i=1; i<N-1; ++i) {
		int *r_end = thrust::copy_if(A+N*i, A+N*i+(N+1)*(N-i-1)+1, B, C, saxpy(N+1,0));
		if (thrust::count(C, r_end, 1) > 1) {
			cout << "NO 6" << endl;
			return 0;
		}
	}
	return 0;
}