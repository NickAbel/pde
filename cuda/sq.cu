#include<stdio.h>
#include<cuda.h>

__global__ void sq(float *d_out, float* d_in)
{
	int idx = threadIdx.x;
	float f = d_in[idx];
	d_out[idx] = f*f;
}

int main(int argc, char** argv)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	float h_in[ARRAY_SIZE];
	for(int i=0; i < ARRAY_SIZE; i++)
	{
		h_in[i] = float(i);
	}
	
	float h_out[ARRAY_SIZE];
	
	//declare GPU memory pointers
	float *d_in;
	float *d_out;

	//allocate memory for the two arrays on the device
	cudaMalloc((void**)&d_in,ARRAY_BYTES);
	cudaMalloc((void**)&d_out,ARRAY_BYTES);
	
	//transfer the array to the GPU
	// destination,source,size,method
	cudaMemcpy(d_in,h_in,ARRAY_BYTES,cudaMemcpyHostToDevice);

	//launch the kernel	
	sq<<<ARRAY_SIZE,1>>>(d_out,d_in);
	cudaDeviceSynchronize();

	cudaError_t error = cudaGetLastError();

	if(error!=cudaSuccess)

	{

	   fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );

	   exit(-1);

	}

	//copy the results back onto the device
	cudaMemcpy(h_out,d_out,ARRAY_BYTES,cudaMemcpyDeviceToHost);
	
	for(int i=0; i < ARRAY_SIZE; i++)
	{
		printf("%f:%f \t",h_in[i], h_out[i]);
		if (i%4==0) printf("\n");
	}
	
	printf("\n");
	
	cudaFree(d_in);
	cudaFree(d_out);

}//end of main
