#include <stdio.h>
#include <cuda.h>
#include <math.h>

// Uses Cyclic Reduction (CR) algorithm on GPU to solve the system of equations.

__global__ void FwdReduction(float* d_A, float* d_F)
{
  float alpha,gamma;
  int index1,index2,offset;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int n=blockDim.x;
  // Forward reduction
  for (int j=powf(2,i+1)-1;j<n;j+=powf(2,i+1))
    {
      offset=powf(2.0f,(float)i);
      index1=j-offset;
      index2=j+offset;
      printf("%d %d %d\n",index1,index2,i);
      alpha=d_A[n*j+index1]/d_A[n*index1+index1];
      gamma=d_A[n*j+index2]/d_A[n*index2+index2];
      for (int k=0;k<n;k++) {
	d_A[n*j+k]-=(alpha*d_A[n*index1+k]+gamma*d_A[n*index2+k]);
      }
      d_F[j]-=(alpha*d_F[index1]+gamma*d_F[index2]);
    }
}

__global__ void BackSub(float* d_A, float* d_x, float* d_F)
{
  int index1,index2,offset;
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int n=blockDim.x;
  int index=(n-1)/2;
  d_x[index]=d_F[index]/d_A[n*index+index];
  for (int j=powf(2,i+1)-1;j<n;j+=powf(2,i+1)) {
    offset=powf(2,i);
    index1=j-offset;
    index2=j+offset;
    d_x[index1]=d_F[index1];
    d_x[index2]=d_F[index2];
    __syncthreads();
    for (int k=0;k<n;k++)
      {
	if (k!=index1) d_x[index1]-=d_A[n*index1+k]*d_x[k];
	if (k!=index2) d_x[index2]-=d_A[n*index2+k]*d_x[k];
      }
    __syncthreads();
    d_x[index1]=d_x[index1]/d_A[n*index1+index1];
    d_x[index2]=d_x[index1]/d_A[n*index2+index2];
  }
}

int main(int argc, char** argv)
{
  // Declare variables
  const int p=4;
  const int n=pow(2,p)-1;
  const int BYTES=n*sizeof(float);
  const float s=2.0f;
  const float r=2.0f+s;

  // Declare arrays
  float* h_A = new float[n*n];
  float* h_x = new float[n];
  float* h_F = new float[n];

  // Initialize the arrays
  for (int i=0;i<n;i++)
    {
      h_x[i]=0.0f;
      h_F[i]=0.0f;
      for (int j=0;j<n;j++)
	{
	  if (i-j==0) h_A[i*n+j]=r;
	  else if (abs(i-j)==1) h_A[i*n+j]=-1.0f;
	  else h_A[i*n+j]=0.0f;
	}
    }

  h_A[0]=h_A[n*(n-1)+(n-1)]=r;
  h_A[1]=h_A[n*(n-1)+(n-2)]=-1.0f;
  
  h_F[(int)n/2]=0.2f;
  h_F[(int)n/2-1]=0.1f;
  h_F[(int)n/2+1]=0.1f;

  // Declare GPU memory pointers
  float* d_A;
  float* d_x;
  float* d_F;

  // Allocate memory on device
  cudaMalloc((void**)&d_A,BYTES*BYTES);
  cudaMalloc((void**)&d_x,BYTES);
  cudaMalloc((void**)&d_F,BYTES);

  // Transfer the array to the GPU
  // Destination, source, size, method
  cudaMemcpy(d_A,h_A,BYTES*BYTES,cudaMemcpyHostToDevice);
  cudaMemcpy(d_x,h_x,BYTES,cudaMemcpyHostToDevice);
  cudaMemcpy(d_F,h_F,BYTES,cudaMemcpyHostToDevice);

  // Kernel launch
  FwdReduction<<<1,n>>>(d_A,d_F);
  cudaDeviceSynchronize();
  BackSub<<<1,n>>>(d_A,d_x,d_F);

  // Copy results back to device
  // Destination, source, size, method
  cudaMemcpy(h_A,d_A,BYTES,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_x,d_x,BYTES,cudaMemcpyDeviceToHost);
  cudaMemcpy(h_F,d_F,BYTES,cudaMemcpyDeviceToHost);

  for (int i=0;i<n;i++) 
    {
      printf("%f ",h_x[i]);
      printf("\n");
    }
  printf("\n");

  // Free memory
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_F);
  delete[] h_A;
  delete[] h_x;
  delete[] h_F;
}
