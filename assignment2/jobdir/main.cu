#include <stdio.h>
#include "kernel.h"
#include <stdlib.h>
#include <math.h>


int comp( const void *a, const void *b)
{
  return *((int *)a) - *((int*)b);
}


int compare(const int *a, const int *b, int N)
{
  int tr = 0;
  for (int i=0; i<N; ++i) {
    if (a[i] != b[i]) {
      printf("%d ", i);
      tr = 1;
    }
  }
  if (tr == 1) {
    puts("");
    return 0;
  }
  return 1;
}


int main(int argc, char** argv)
{
  int *inp, *out, *d_temp, *d_inp;
  int N = 1024*5;
  int mmax = 10, ii=1;
  scanf("%d %d", &ii, &mmax);
  for (; ii<=mmax; ++ii) {
    N=ii;
  int numbytes = sizeof(int)*N;

  inp = (int *) malloc(numbytes);
  out = (int *) malloc(numbytes);

  for (int i=0; i<N; ++i) {
    // inp[i] = rand();
    inp[i] = N-i;
  }

  cudaError_t err;
  cudaMalloc(&d_inp, numbytes);
  cudaMalloc(&d_temp, numbytes);

  cudaMemcpy(d_inp, inp, numbytes, cudaMemcpyHostToDevice);

  msort<<<(int)ceil((float)N/1024), 1024>>>(d_inp, d_temp, N);
  // msort<<<1, N>>>(d_inp, d_temp, N);
  

  cudaThreadSynchronize();
  /* Print the last error encountered -- helpful for debugging */
  err = cudaGetLastError();  
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  
  cudaMemcpy(out, d_inp, numbytes, cudaMemcpyDeviceToHost);
  qsort(inp, N, sizeof(int), comp);
  // printf("%s\n", compare(inp, out, N)? "Success":"Fail");
  if (compare(inp, out, N) == 0) {
    printf("nval: %d\n", N);
    for (int jj=0; jj<N; ++jj) {
      printf("%d ", out[jj]);
    }
    break;
  }

  // for (int i=0; i<N; ++i) {
  //   printf("%d ", out[i]);
  // }
  // puts("");
  free(inp);
  free(out);
  cudaFree(d_inp);
  cudaFree(d_temp);
  }
  return 0; 
}
