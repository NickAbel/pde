#include <stdio.h>
#include <cuda.h>
#include <math.h>

// Uses Cyclic Reduction (CR) algorithm on GPU to solve the system of equations.

__global__ void CyclicReduction(int m, int n, float *d_x, float s)
{
	__shared__  float a[128],c[128],d[128];
	float ai,bi,ci,di,bb;
	int i;
	for (int j=0;j<m;j++) 
	{
		i=threadIdx.x;
		bb=1.0f/(2.0f+s);
		if (i>0) ai=-bb;
		else ai=0.0f;
		if (i<blockDim.x-1) ci=-bb;
		else ci=0.0f;
		if (j==0) di=s*d_x[i]*bb;
		else di=s*di*bb;
		a[i]=ai;
		c[i]=ci;
		d[i]=di;
		//Forward reduction
		for (int k=1; k<n; k=2*k) 
		{
			__syncthreads();
			bi=1.0f;
			if (i-k>=0) {
				di=di-ai*d[i-k];
				bi=bi-ai*c[i-k];
				ai=-ai*a[i-k];
			}
			if (i+k<n) {
				di=di-ci*d[i+k];
				bi=bi-ci*a[i+k];
				ci=-ci*c[i+k];
			}
			bb=1.0f/bi;
			ai=ai*bb;
			ci=ci*bb;
			di=di*bb;
			a[i]=ai;
			c[i]=ci;
			d[i]=di;
		}
	}
	d_x[i]=di;
}

int main(int argc, char** argv)
{
	const int m=5;
	const int n=16;
	const int BYTES=n*sizeof(float);
	float h_x[n];
	float s=2.0f;
	//initialize arrays
	for (int i=0;i<n;i++)
	{
		h_x[i]=0.0f;
	}
	h_x[3]=h_x[6]=0.1f;
	h_x[4]=h_x[5]=0.2f;

	//declare GPU memory pointers
	float* d_x;

	//allocate memory on the device
	cudaMalloc((void**)&d_x,BYTES);

	//transfer the array to the GPU
	//destination, source, size, method
	cudaMemcpy(d_x,h_x,BYTES,cudaMemcpyHostToDevice);

	//launch the kernel
	CyclicReduction<<<1,n>>>(n,m,d_x,s);
	cudaDeviceSynchronize();

	//copy the results back onto the device
	//destination, source, size, method
	cudaMemcpy(h_x,d_x,BYTES,cudaMemcpyDeviceToHost);

	for (int i=0;i<n;i++) 
	{
		printf("%f",h_x[i]);
		printf("\n");
	}
	printf("\n");

	//free memory previously allocated on the device
	cudaFree(d_x);
}
