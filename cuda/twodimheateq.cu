#include <stdio.h>
#include <cuda.h>

__global__ void TwoDimHeatEq(float *d_A, float *d_B, double s)
{
  // 2-dimensional block, 2-dimensional grid
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  int threadId = (blockId * (blockDim.x * blockDim.y) + 
		  (threadIdx.y * blockDim.x) + threadIdx.x);
  int threadAbove = threadId - (blockDim.x * gridDim.x);
  int threadBelow = threadId + (blockDim.x * gridDim.x);
  int N = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

  // Punt if this thread is a boundary point
  // I should make this a function in a header file
  if ((threadAbove <= 0) || (threadBelow >= N-1) ||
      (threadId % (blockDim.x * gridDim.x) == 0) || 
      ((threadId+1) % (blockDim.x * gridDim.x) == 0))
    return;
  else
    { 
      //d_B[threadId] = 33.0f;
      d_B[threadId] = (s*(d_A[threadId+1] + d_A[threadId-1]
			  + d_A[threadAbove] + d_A[threadBelow])
		       + (1 - 4*s) * d_A[threadId]);
    }
}

int main(int argc, char** argv)
{
  const int n = 16;
  const int BYTES = n*n * sizeof(float);

  float* h_A = new float[n*n];
  float* h_B = new float[n*n];

  double s = 0.25;
  
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
	{
	  h_A[n*i + j] = 0;
	  h_B[n*i + j] = 0;
	  if (j==0) h_A[n*i + j] = 1000;
	}
    }
  
  //declare GPU memory pointers
  float *d_A;
  float *d_B;

  //allocate memory on the device
  cudaMalloc((void **) &d_A, BYTES);
  cudaMalloc((void **) &d_B, BYTES);
	
  //transfer the array to the GPU
  //destination, source, size, method
  cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, d_A, BYTES, cudaMemcpyDeviceToDevice);

  //launch the kernel
  for (int i=0; i < 10; i++)
    {
      TwoDimHeatEq<<<dim3(n/4,n/4),dim3(n/4,n/4)>>>(d_A, d_B, s);
      cudaMemcpy(d_A, d_B, BYTES, cudaMemcpyDeviceToDevice);
      cudaDeviceSynchronize();
    }
  //copy the results back onto the device
  //destination, source, size, method
  cudaMemcpy(h_B, d_B, BYTES, cudaMemcpyDeviceToHost);
	
  for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
	{
	  printf("%-10f \t", h_B[i*n + j]);
	}
      printf("\n");
    }
  printf("\n \n");
	
  //free memory previously allocated on the device 
  cudaFree(d_A);
  cudaFree(d_B);
}
