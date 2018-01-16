#include <stdio.h>
#include "kernels.h"
#include <stdlib.h>
#include <math.h>


int comp( const void *a, const void *b)
{
  return *((int *)a) - *((int*)b);
}


int compare(const int *a, const int *b, int N)
{
  for (int i=0; i<N; ++i) {
    if (a[i] != b[i]) {
      return 0;
    }
  }
  return 1;
}


int main(int argc, char** argv)
{
  int *inp, *out, *d_temp, *d_inp;
  int N = 4096;
  int M = 32;
for (int ii=0; ii<7; ++ii) {
scanf("%d %d", &N, &M);  
  int numbytes = sizeof(int)*N;

  inp = (int *) malloc(numbytes);
  out = (int *) malloc(numbytes);

  for (int i=0; i<N; ++i) {
    inp[i] = N-i;
  }

  cudaError_t err;
  cudaMalloc(&d_inp, numbytes);
  cudaMalloc(&d_temp, numbytes);

  cudaMemcpy(d_inp, inp, numbytes, cudaMemcpyHostToDevice);

  msort<<<(int)ceil((float)N/M), M>>>(d_inp, d_temp, N);
  
  cudaThreadSynchronize();
  /* Print the last error encountered -- helpful for debugging */
  err = cudaGetLastError();  
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));
  
  cudaMemcpy(out, d_inp, numbytes, cudaMemcpyDeviceToHost);
  qsort(inp, N, sizeof(int), comp);
  printf("%s\n", compare(inp, out, N)? "Success":"Fail");
  
  free(inp);
  free(out);
  cudaFree(d_inp);
  cudaFree(d_temp);
}
  return 0; 
}
