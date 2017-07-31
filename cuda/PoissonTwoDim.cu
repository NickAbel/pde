#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void TwoDimPoisson(float *d_A, float *d_B, float *d_F, double dx, float* diff)
{
	int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
	int threadAbove = threadId - blockDim.x;
	int threadBelow = threadId + blockDim.x;
	int N = gridDim.x * blockDim.x;

	// Punt if this thread is a boundary point
	if ((threadAbove <= 0) || (threadBelow >= N-1) ||
			(threadId % (blockDim.x) == 0) || 
			((threadId+1) % (blockDim.x) == 0))
		return;
	else
	{ 
		d_B[threadId] = (.25*(d_A[threadId+1] + d_A[threadId-1]
					+ d_A[threadAbove] + d_A[threadBelow])
				+ d_F[threadId]*pow(dx,2.0));
	}
	atomicAdd(diff,abs(d_B[threadId]-d_A[threadId])/(blockDim.x*gridDim.x));
}

int main(int argc, char** argv)
{
	const int n = atoi(argv[1]);
	int steps = 0;
	const int BYTES = n*n * sizeof(float);
	float* h_A = new float[n*n];
	float* h_B = new float[n*n];
	float* h_F = new float[n*n];
	double dx = 0.1;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			h_A[n*i + j] = 0;
			h_B[n*i + j] = 0;
			h_F[n*i + j] = 0;

			if (j==0) h_A[n*i + j] = 1000;

			if (j==n-1) h_F[n*i + j] = 5*i+20;

			h_A[n*i + j]+=h_F[n*i + j];
		}
	}

	//declare GPU memory pointers
	float *d_A;
	float *d_B;
	float *d_F;
	float *diff;

	//allocate memory on the device
	cudaMalloc((void **) &d_A, BYTES);  
	cudaMalloc((void **) &d_B, BYTES);
	cudaMalloc((void **) &d_F, BYTES);
	cudaMallocManaged(&diff, sizeof(float));
	*diff = 0.0;

	//transfer the array to the GPU
	//destination, source, size, method
	cudaMemcpy(d_B, h_B, BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, h_A, BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, d_A, BYTES, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_F, h_F, BYTES, cudaMemcpyHostToDevice);

	//launch the kernel
	while (true)
	{
		steps++;
		*diff = 0.0;
		TwoDimPoisson<<<n,n>>>(d_A, d_B, d_F, dx, diff);
		cudaDeviceSynchronize();
		cudaMemcpy(h_A, d_A, BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_B, d_B, BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(d_A, d_B, BYTES, cudaMemcpyDeviceToDevice);
		if (*diff < 0.0001) break;
	}

	//copy the results back onto the device
	//destination, source, size, method
	cudaMemcpy(h_B, d_B, BYTES, cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			printf("%-10.3f  ", h_B[i*n + j]);
		}
		printf("\n");
	}
	printf("\nSteps: %d \nn: %d\n",steps,n);

	//free memory previously allocated on the device 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_F);
}
