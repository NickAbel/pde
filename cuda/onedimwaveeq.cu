#include <stdio.h>
#include <cuda.h>
#include <math.h>


__global__ void WaveEq(float *d_mm1, float *d_m, float *d_mp1, float s,
		       float T, float dt, float cfl)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  d_m[i]=d_mp1[i];
  float t=0.0;
  while (t < T) {
    t=dt+t;
    d_mm1[i]=d_m[i];
    d_m[i]=d_mp1[i];
    __syncthreads();
    if (i>0 && i<(blockDim.x-1)) {
      d_mp1[i]=2*d_m[i]-d_mm1[i]+s*(d_m[i-1]-2*d_m[i]+d_m[i+1]);
    }
  }
}

int main(int argc, char** argv)
{
  const int n=100;
  const int BYTES=n*sizeof(float);
  float h_mm1[n];
  float h_m[n];
  float h_mp1[n];
  float c=1.0;
  float T=1.0;
  float dx=0.1;
  float dt=dx/c;
  float cfl=c*dt/dx;
  float s=cfl*cfl;
  //initialize arrays
  for (int i=0;i<n;i++)
    {
      h_mm1[i]=0.0;
      h_m[i]=0.0;
      h_mp1[i]=0.0;
    }
  h_mp1[48]=0.1f;
  h_mp1[50]=0.1f;
  h_mp1[49]=0.2f;

  //declare GPU memory pointers
  float* d_mm1;
  float* d_m;
  float* d_mp1;

  //allocate memory on the device
  cudaMalloc((void**)&d_mm1,BYTES);
  cudaMalloc((void**)&d_m,BYTES);
  cudaMalloc((void**)&d_mp1,BYTES);

  //transfer the array to the GPU
  //destination, source, size, method
  cudaMemcpy(d_mm1,h_mm1,BYTES,cudaMemcpyHostToDevice);
  cudaMemcpy(d_m,h_m,BYTES,cudaMemcpyHostToDevice);
  cudaMemcpy(d_mp1,h_mp1,BYTES,cudaMemcpyHostToDevice);

  //launch the kernel
  WaveEq<<<1,n>>>(d_mm1,d_m,d_mp1,s,T,dt,cfl);
  cudaDeviceSynchronize();

  //copy the results back onto the device
  //destination, source, size, method
  cudaMemcpy(h_mm1,d_mm1,BYTES,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_m,d_m,BYTES,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_mp1,d_mp1,BYTES,cudaMemcpyDeviceToHost);

  for (int i=0;i<n;i++) 
    {
      printf("%d \t %f",i,h_mp1[i]);
      printf("\n");
    }

  printf("\n");

  //free memory previously allocated on the device
  cudaFree(d_mm1);
  cudaFree(d_m);
  cudaFree(d_mp1);
}