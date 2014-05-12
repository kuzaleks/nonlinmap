
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 16
#define THREADS_IN_BLOCK 16

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf("%s\n", "Failed to run stmt ", #stmt);                       \
            printf("%s %s\n", "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


// Compute C = A * B
__device__ void matrixMultiply(double * A, double * B, double * C,
                   int numARows, int numAColumns,
                   int numBRows, int numBColumns,
                   int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (row < numCRows && col < numCColumns) {
		double cvalue = 0.0;
		for (int k = 0; k < numAColumns; ++k)
			cvalue += A[row * numAColumns + k] * B[k * numBColumns + col];
		C[row * numCColumns + col] = cvalue;
	}
}
 

__device__ void matrixMultiplyShared(double * A, double * B, double * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	__shared__ double tileForA[TILE_WIDTH][TILE_WIDTH];
	__shared__ double tileForB[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	double cVal = 0.0;
	for (int tInd = 0; tInd < (numAColumns-1) / TILE_WIDTH + 1; ++tInd) {
		if (rowInd < numARows && (tInd*TILE_WIDTH + tx) < numAColumns)
			tileForA[ty][tx] = A[rowInd * numAColumns + tInd*TILE_WIDTH + tx];
		else
			tileForA[ty][tx] = 0.0;
		if ((tInd*TILE_WIDTH + ty) < numBRows && colInd < numBColumns)
			tileForB[ty][tx] = B[(tInd*TILE_WIDTH + ty)*numBColumns + colInd];
		else
			tileForB[ty][tx] = 0.0;
		__syncthreads();
		
		for (int elInd = 0; elInd < TILE_WIDTH; ++elInd)
			cVal += tileForA[ty][elInd] * tileForB[elInd][tx];
		__syncthreads();
	}
	
	if (rowInd < numCRows && colInd < numCColumns)
		C[rowInd*numCColumns + colInd] = cVal;
		
}

__device__ double kernal_func(double sigma, double* vArr, double* wArr, 
							int trSampleInd, int testSampleInd, int dim) {
	double cVal = 0.0;
	double vEl, wEl;
	for (int i = 0; i < dim; ++i) {
		vEl = vArr[trSampleInd * dim + i];
		wEl = wArr[testSampleInd * dim + i];
		cVal += (vEl - wEl) * (vEl - wEl);
	}

	return exp(-0.5 * cVal / (sigma * sigma));
}

__device__ void make_tkern_device(double* train, double* test, double* Kt,
								double sigma,
								int trTotal, int testTotal, int dim) {
								
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd < trTotal) {
		Kt[rowInd*trTotal + colInd] = kernal_func(sigma, train, test, colInd, rowInd, dim);
		//dprintf("(%d %d): %.2f\n", rowInd, colInd, Kt[rowInd*trTotal + colInd]);
	}
}

__global__ void kern_transform(double* train, double* test, 
							   double* eigvecs, double* Kt, double* transTest, 
							   double sigma, int trTotal, int testTotal, int dim, int transDim) {
	make_tkern_device(train, test, Kt, sigma, trTotal, testTotal, dim);
	matrixMultiply(Kt, eigvecs, transTest,
			             testTotal, trTotal,
			             trTotal, transDim,
			             testTotal, transDim);
			             
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

void save_to_file(double* arr, int n, const char* fname) {
	FILE* fp = fopen(fname, "wb");
	
	fwrite((void *) arr, sizeof(double), n, fp);
	
	fclose(fp);
}

void read_file(double* arr, int n, const char* fname) {
	FILE* fp = fopen(fname, "rb");
	
	fread((void *) arr, sizeof(double), n, fp);
	
	fclose(fp);
}

// Helper function for using CUDA to add vectors in parallel.
void transform(double *train, double *test, double *eigvecs, double *transTest,
				int trTotal, int testTotal, int dim, int transDim, double sigma,
				double *tKern, bool verbose)
{
	cudaError_t error;

	double * dTrain;
    double * dTest;
    double * dtKern;
    double * deigvecs;
    double * dTransTest;
    
    
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
        printf("cudaMalloc dTest returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
	error = cudaMalloc((void**) &dtKern, testTotal * trTotal * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc dtKern returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void**) &deigvecs, trTotal * transDim * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc deigvecs returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void**) &dTransTest, testTotal * transDim * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc dTransTest returned error code %d, line(%d)\n", error, __LINE__);
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
    error = cudaMemcpy(deigvecs, eigvecs, trTotal * transDim * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (eigvecs, deigvecs) returned error code %d, line(%d)\n", error, __LINE__);
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
    
    kern_transform<<<DimGrid, DimBlock>>>(dTrain, dTest, deigvecs, dtKern, dTransTest,
											sigma, trTotal, testTotal, dim, transDim);
											
	cudaThreadSynchronize();
	
	// Record the stop event
    error = cudaEventRecord(stop, NULL);
    
	error = cudaEventSynchronize(stop);
	
	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);
	printf("Performance: Time= %.3f msec\n", msecTotal);
	
	
	printf("%s\n", "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	error = cudaMemcpy(tKern, dtKern, testTotal * trTotal * sizeof(double), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (dTest to test) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    if (verbose)
		print_matr(tKern, testTotal, trTotal);
		
	save_to_file(tKern, testTotal * trTotal, "tkern.bin");
	//int i = 0, j = 0;
	//printf("tKern(%d, %d) = %.5f\n", i, j, tKern[i * dim + j]);
	
	error = cudaMemcpy(transTest, dTransTest, testTotal * transDim * sizeof(double), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (dTransTest to transTest) returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    if (verbose)
		print_matr(transTest, testTotal, transDim);
	save_to_file(transTest, testTotal * transDim, "trans_test.bin");

	printf("%s\n", "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(dTrain);
	cudaFree(dTest);
	cudaFree(dtKern);
	cudaFree(deigvecs);
	cudaFree(transTest);
}


int main()
{
	double sigma = 4.0;
    const int dim = 13;
    const int transDim = dim;
    bool verbose = false;
    int trTotal = 2000;
    int testTotal = 10000;
    int tKernRows;
    int tKernCols;
    double * train; 
    double * test; 
    double * tKern; 
    double * eigvecs;
    double * transTest;
    
    printf("%s\n", "Importing data and creating memory on host");
    train = (double *) malloc(trTotal * dim * sizeof(double));
    test = (double *) malloc(testTotal * dim * sizeof(double));
    
	fill_matr(train, trTotal, dim);
	if (verbose)
		print_matr(train, trTotal, dim);
	save_to_file(train, trTotal * dim, "train.bin");
	fill_matr(test, testTotal, dim);
	if (verbose)
		print_matr(test, testTotal, dim);
	save_to_file(test, testTotal * dim, "test.bin");
	
	eigvecs = (double *) malloc(trTotal * transDim * sizeof(double));
	fill_matr(eigvecs, trTotal, transDim);
	if (verbose)
		print_matr(eigvecs, trTotal, transDim);
	save_to_file(eigvecs, trTotal * transDim, "eigvecs.bin");
	
	transTest = (double *) malloc(testTotal * transDim * sizeof(double));
	
	tKernRows = testTotal;
    tKernCols = trTotal;
    //@@ Allocate the hostC matrix
    printf("%s\n", "Importing data and creating memory on host");
	tKern = (double *) malloc(tKernRows * tKernCols * sizeof(double));
	transTest = (double *) malloc(testTotal * dim * sizeof(double));

    printf("%s %d %s %d\n", "The dimensions of train are ", trTotal, " x ", dim);
    printf("%s %d %s %d\n", "The dimensions of test are ", testTotal, " x ", dim);

	transform(train, test, eigvecs, transTest, 
			  trTotal, testTotal, dim, transDim, sigma, tKern, verbose);
    
    free(train);
    free(test);
    free(tKern);
    free(eigvecs);
    free(transTest);
    
    return 0;
}

