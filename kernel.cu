
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 32
#define THREADS_IN_BLOCK 32

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf("%s\n", "Failed to run stmt ", #stmt);                       \
            printf("%s %s\n", "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__device__ float kernal_func(double sigma, double* vArr, double* wArr, 
							int trSampleInd, int testSampleInd, int dim) {
	float cVal = 0.0;
	float vEl, wEl;
	for (int i = 0; i < dim; ++i) {
		vEl = vArr[trSampleInd * dim + i];
		wEl = wArr[testSampleInd * dim + i];
		cVal += (vEl - wEl) * (vEl - wEl);
	}
	return exp(-0.5 * cVal / (sigma * sigma));
}

__global__ void make_tkern_device(double* train, double* test, double* eigvecs, double* Kt,
								double sigma,
								int trTotal, int testTotal, int dim) {
								
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd < trTotal)
		Kt[rowInd*trTotal + colInd] = kernal_func(sigma, train, test, colInd, rowInd, dim);
}

void fill_matr(double* M, int nrow, int ncol) {
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			M[i * ncol + j] = rand() % 3;
}

void print_matr(double* M, int nrow, int ncol) {
	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++)
			printf("%5.3f ", M[i * ncol + j]);
		printf("%s\n", "");
	}
	printf("%s\n", "");
}

// Helper function for using CUDA to add vectors in parallel.
void make_tkern(double *train, double *test, 
				int trTotal, int testTotal, int dim, double sigma,
				double *tKern)
{
	cudaError_t error;

	double * dTrain;
    double * dTest;
    double * dtKern;
    
    
    printf("%s\n", "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    
    error = cudaMalloc((void**) &dTrain, trTotal * dim * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
	error = cudaMalloc((void**) &dTest, testTotal * dim * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc((void**) &dtKern, testTotal * trTotal * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	
	printf("%s\n", "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	error = cudaMemcpy(dTrain, train, trTotal * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (train, dTrain) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMemcpy(dTest, test, testTotal * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (test,dTest) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    //@@ Initialize the grid and block dimensions here
	dim3 DimGrid((trTotal - 1)/THREADS_IN_BLOCK + 1, (testTotal - 1)/THREADS_IN_BLOCK + 1, 1);
	dim3 DimBlock(THREADS_IN_BLOCK, THREADS_IN_BLOCK, 1);

	printf("%s\n", "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	// Allocate CUDA events that we'll use for timing
	
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

	// Record the start event
    error = cudaEventRecord(start, NULL);
    
    make_tkern_device<<<DimGrid, DimBlock>>>(dTrain, dTest, NULL, dtKern,
											sigma, trTotal, testTotal, dim);
											
	cudaThreadSynchronize();
	
	// Record the stop event
    error = cudaEventRecord(stop, NULL);
    
	error = cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	printf("Performance: Time= %.3f msec\n", msecTotal);
	
	printf("%s\n", "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	error = cudaMemcpy(tKern, dtKern, testTotal * trTotal * sizeof(double), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (test,dTest) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	print_matr(tKern, testTotal, trTotal);
	
	printf("%s\n", "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(dTrain);
	cudaFree(dTest);
	cudaFree(dtKern);
}


int main()
{
    const int dim = 13;
    int trTotal = 10;
    int testTotal = 7;
    int tKernRows;
    int tKernCols;
    double * train; 
    double * test; 
    double * tKern; 
    
    printf("%s\n", "Importing data and creating memory on host");
    train = (double *) malloc(trTotal * dim * sizeof(double));
    test = (double *) malloc(testTotal * dim * sizeof(double));
	fill_matr(train, trTotal, dim);
	print_matr(train, trTotal, dim);
	fill_matr(test, testTotal, dim);
	print_matr(test, testTotal, dim);
	
	tKernRows = testTotal;
    tKernCols = trTotal;
    //@@ Allocate the hostC matrix
    printf("%s\n", "Importing data and creating memory on host");
	tKern = (double *) malloc(tKernRows * tKernCols * sizeof(double));

    printf("%s %d %s %d\n", "The dimensions of train are ", trTotal, " x ", dim);
    printf("%s %d %s %d\n", "The dimensions of test are ", testTotal, " x ", dim);
    
    free(train);
    free(test);
    free(tKern);
    
    return 0;
}

