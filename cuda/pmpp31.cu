#include<stdio.h>
#include<cuda.h>

__global__ void mtrxAdd(float* d_a, float* d_b, float* d_c)
{
	int i = threadIdx.x;
	int j = blockIdx.x;
	int n = blockDim.x;
	float bij = d_b[i*n+j];
	float cij = d_c[i*n+j];
	d_a[i*n+j]=bij+cij;
}

int main(int argc, char** argv)
{
	const int ARRAY_N = 8;
	const int n = ARRAY_N;
	const int ARRAY_BYTES = ARRAY_N * ARRAY_N * sizeof(float);
	float h_b[ARRAY_N*ARRAY_N];
	float h_c[ARRAY_N*ARRAY_N];
	for (int i=0; i < ARRAY_N; i++)
	{
		for (int j=0; j < ARRAY_N; j++) {
			h_b[i*n+j] = float(i);
			h_c[i*n+j] = float(j);
		}
	}
	
	float h_a[ARRAY_N*ARRAY_N];
	
	//declare GPU memory pointers
	float *d_a;
	float *d_b;
	float *d_c;

	//allocate memory on the device
	cudaMalloc((void**)&d_a,ARRAY_BYTES);
	cudaMalloc((void**)&d_b,ARRAY_BYTES);
	cudaMalloc((void**)&d_c,ARRAY_BYTES);
	
	//transfer the array to the GPU
	//destination, source, size, method
	cudaMemcpy(d_c,h_c,ARRAY_BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b,ARRAY_BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(d_a,h_a,ARRAY_BYTES,cudaMemcpyHostToDevice);

	//launch the kernel	
	mtrxAdd<<<ARRAY_N,ARRAY_N>>>(d_a,d_b,d_c);
	cudaDeviceSynchronize();

	//copy the results back onto the device
	//destination, source, size, method
	cudaMemcpy(h_a,d_a,ARRAY_BYTES,cudaMemcpyDeviceToHost);
	
	for (int i=0; i<ARRAY_N; i++) {
		for (int j=0; j<ARRAY_N; j++) {
			if (h_a[i*n+j] < 10) printf(" ");
			printf("%.2f ",h_a[i*n+j],h_b[i*n+j],h_c[i*n+j]);
		}
		printf("\n");
	}
	
	//free memory previously allocated on the device 
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}
