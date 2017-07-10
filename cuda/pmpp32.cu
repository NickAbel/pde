#include<stdio.h>
#include<cuda.h>

__global__ void mtrxVecMult(float* d_a, float* d_b, float* d_x)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int n = blockDim.x;
	float aij = d_a[j*n+i];
	d_x[i] = d_b[j]*aij;
}

int main(int argc, char** argv)
{
	const int n = 8;
	const int A_BYTES = n * n * sizeof(float);
	const int B_BYTES = n * sizeof(float);
	float h_a[n*n];
	float h_b[n];
	float h_x[n];
	for (int i=0; i < n; i++)
	{
		for (int j=0; j < n; j++) h_a[i*n+j] = 3*j;
		h_b[i] = 2;
	}
	
	//declare GPU memory pointers
	float *d_a;
	float *d_b;
	float *d_x;

	//allocate memory on the device
	cudaMalloc((void**)&d_a,A_BYTES);
	cudaMalloc((void**)&d_b,B_BYTES);
	cudaMalloc((void**)&d_x,B_BYTES);
	
	//transfer the array to the GPU
	//destination, source, size, method
	cudaMemcpy(d_x,h_x,B_BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,B_BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(d_a,h_a,A_BYTES,cudaMemcpyHostToDevice);

	//launch the kernel	
	mtrxVecMult<<<n,n>>>(d_a,d_b,d_x);
	cudaDeviceSynchronize();
	cudaGetLastError();

	//copy the results back onto the device
	//destination, source, size, method
	cudaMemcpy(h_x,d_x,B_BYTES,cudaMemcpyDeviceToHost);
	
	for (int i=0; i<n; i++) {
		printf("%.2f",h_x[i]);
		printf("\n");
	}
	
	//free memory previously allocated on the device 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_x);
}
