#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 16
#define THREADS_IN_BLOCK 16

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf("%s\n", "Failed to run stmt ");                       \
            printf("%s %s\n", "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__device__ float kernal_func(float sigma, float* vArr, float* wArr, 
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

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	__shared__ float tileForA[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tileForB[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	float cVal = 0.0;
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

__global__ void make_tkern_matr(float* train, float* test, float* eigvecs, float* Kt
								float sigma,
								int trTotal, int testTotal, int dim) {
								
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd < trTotal)
		Kt[rowInd*trTotal + colInd] = kernal_func(sigma, train, test, colInd, rowInd, dim);
}

void fill_matr(float* M, int nrow, int ncol) {
	for (int i = 0; i < nrow; i++)
		for (int j = 0; j < ncol; j++)
			M[i * ncol + j] = rand() % 3;
}

void print_matr(float* M, int nrow, int ncol) {
	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++)
			printf("%5.3f ", M[i * ncol + j]);
		printf("%s\n", "");
	}
	printf("%s\n", "");
}

int main(int argc, char ** argv) {
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows = 5; // number of rows in the matrix A
    int numAColumns = 5; // number of columns in the matrix A
    int numBRows = 5; // number of rows in the matrix B
    int numBColumns = 5; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    // args = wbArg_read(argc, argv);

	printf("%s\n", "Importing data and creating memory on host");
    hostA = (float *) malloc(numARows * numAColumns * sizeof(float));
    hostB = (float *) malloc(numBRows * numBColumns * sizeof(float));
	fill_matr(hostA, numARows, numAColumns);
	print_matr(hostA, numARows, numAColumns);
	fill_matr(hostB, numBRows, numBColumns);
	print_matr(hostB, numBRows, numBColumns);
	//@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    printf("%s\n", "Importing data and creating memory on host");
	hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));

    printf("%s %d %s %d\n", "The dimensions of A are ", numARows, " x ", numAColumns);
    printf("%s %d %s %d\n", "The dimensions of B are ", numBRows, " x ", numBColumns);

    printf("%s\n", "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	wbCheck(cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(float)));
	wbCheck(cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(float)));
	wbCheck(cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(float)));

    printf("%s\n", "Allocating GPU memory.");

    printf("%s\n", "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

    printf("%s\n", "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
	dim3 DimGrid((numCColumns-1)/THREADS_IN_BLOCK + 1, (numCRows-1)/THREADS_IN_BLOCK + 1, 1);
	dim3 DimBlock(THREADS_IN_BLOCK, THREADS_IN_BLOCK, 1);
    
    printf("%s\n", "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	// Allocate CUDA events that we'll use for timing
	
    cudaEvent_t start;
    wbCheck(cudaEventCreate(&start));

    cudaEvent_t stop;
    wbCheck(cudaEventCreate(&stop));

	// Record the start event
    wbCheck(cudaEventRecord(start, NULL));
    
	matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
			             						numARows, numAColumns,
			             						numBRows, numBColumns,
			             						numCRows, numCColumns);

	cudaThreadSynchronize();
	
	// Record the stop event
    wbCheck(cudaEventRecord(stop, NULL));
    
	wbCheck(cudaEventSynchronize(stop));
	float msecTotal = 0.0f;
    wbCheck(cudaEventElapsedTime(&msecTotal, start, stop));
    double flopsPerMatrixMul = 2.0 * (double)numAColumns * (double)numARows * (double)numBColumns;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f);
    
    //printf("Performance: Time= %.3f msec\n", msecTotal);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
        gigaFlops,
        msecTotal,
        flopsPerMatrixMul,
        DimBlock.x * DimBlock.y);
    
    printf("%s\n", "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));
	print_matr(hostC, numCRows, numCColumns);
	/*printf(
		"(0, 0): %.2f (0, 1): %.2f\n(1, 0): %.2f, (1, 1): %.2f\n", 
		hostC[0], hostC[1], 
		hostC[numCColumns], hostC[numCColumns + 1]
	);
	*/
    printf("%s\n", "Copying output memory to the CPU");
	
    printf("%s\n", "Freeing GPU Memory");
    //@@ Free the GPU memory here
	wbCheck(cudaFree(deviceA));
	wbCheck(cudaFree(deviceB));
	wbCheck(cudaFree(deviceC));

    printf("%s\n", "Freeing GPU Memory");

    //wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

