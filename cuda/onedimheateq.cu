#include<stdio.h>
#include<cuda.h>

__global__ void HeatEq(float* d_a, float* d_b, double s)
{
	int i = threadIdx.x;
	d_b[i+1] = d_a[i+1]+s*(d_a[i+2]+d_a[i]-2*d_a[i+1]);
}

int main(int argc, char** argv)
{
	const int n = 16;
	const int BYTES = n * sizeof(float);
	float h_a[n];
	float h_b[n];
	double s = 0.25;
	for (int i=0; i < n; i++)
	{
		h_a[i]=0;
	}
	h_a[5]=h_a[8]=0.1;
	h_a[6]=h_a[7]=0.2;
	
	//declare GPU memory pointers
	float *d_a;
	float *d_b;

	//allocate memory on the device
	cudaMalloc((void**)&d_a,BYTES);
	cudaMalloc((void**)&d_b,BYTES);
	
	//transfer the array to the GPU
	//destination, source, size, method
	cudaMemcpy(d_b,h_b,BYTES,cudaMemcpyHostToDevice);
	cudaMemcpy(d_a,h_a,BYTES,cudaMemcpyHostToDevice);

	//launch the kernel
	for (int i=0; i<25; i++) {
		HeatEq<<<1,(n-2)>>>(d_a,d_b,s);
		cudaMemcpy(d_a,d_b,BYTES,cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
	}
	//copy the results back onto the device
	//destination, source, size, method
	cudaMemcpy(h_b,d_b,BYTES,cudaMemcpyDeviceToHost);
	
	for (int i=0; i<n; i++) {
		printf("%d \t %.5f",i,h_b[i]);
		printf("\n");
	}
	printf("\n \n");
	
	//free memory previously allocated on the device 
	cudaFree(d_a);
	cudaFree(d_b);
}
