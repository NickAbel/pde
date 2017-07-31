#include <stdio.h>
#include <cuda.h>

__global__ void ThreeDimPoisson(float *d_A, float *d_B, float *d_F, float dx, float* diff)
{
	// 2-dimensional block, 1-dimensional grid
	int threadId = blockDim.x*blockDim.y*blockIdx.x + blockDim.x*threadIdx.y + threadIdx.x; 
	int threadAbove  = threadId + blockDim.x*blockDim.y; // z++
	int threadBelow  = threadId - blockDim.x*blockDim.y; // z--
	int threadAhead  = threadId + blockDim.x;   // y++
	int threadBehind = threadId - blockDim.x;   // y--
	int threadRight  = threadId + 1;   // x++
	int threadLeft   = threadId - 1;   // x--

	// Punt if this thread is a boundary point
	if ((threadIdx.x == 0) || (threadIdx.x == blockDim.x-1) ||
			(threadIdx.y == 0) || (threadIdx.y == blockDim.y-1) ||
			(blockIdx.x == 0) || (blockIdx.x == gridDim.x - 1))
	{
		return;
	}
	else
	{ 
		d_B[threadId] = (((float)(1/6.0))*(d_A[threadRight] + d_A[threadLeft]
					+ d_A[threadAbove] + d_A[threadBelow]
					+ d_A[threadAhead] + d_A[threadBehind])
				+ d_F[threadId]*pow((double)dx,2.0));
	}
	atomicAdd(diff,abs(d_B[threadId]-d_A[threadId])/(blockDim.x*blockDim.y*gridDim.x));
}

int main(int argc, char** argv)
{
	int steps = 0;
	const int n = atoi(argv[1]);
	const int BYTES = n*n*n * sizeof(float);
	float* h_A = new float[n*n*n];
	float* h_B = new float[n*n*n];
	float* h_F = new float[n*n*n];
	float dx = 0.1;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				h_A[i + n*(j + n*k)] = 0;
				h_B[i + n*(j + n*k)] = 0;
				h_F[i + n*(j + n*k)] = 0;
				
				if (k==0 || k==(n-1)) h_A[i + n*(j + n*k)] = 5;

				if (j==0) h_A[i + n*(j + n*k)] = 10;

				if (j==n-1) h_F[i + n*(j + n*k)] = 2*i+2;

				h_A[i + n*(j + n*k)]+=h_F[i + n*(j + n*k)];
			}
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
		ThreeDimPoisson<<<n,dim3(n,n)>>>(d_A, d_B, d_F, dx, diff);
		cudaDeviceSynchronize();
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
			for (int k = 0; k < n; k++)
			{
				printf("%-10.3f  ", h_B[i + n*(j + n*k)]);
			}
			printf("\n");
		}
		printf("\n\n\n");
	}
	printf("\nSteps: %d\nn: %d\n\n",steps,n);

	//free memory previously allocated on the device 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_F);
}
