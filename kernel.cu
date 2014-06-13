
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

#define MAX(a,b) (((a)>(b))?(a):(b))

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

__device__ void center(double* Kt, double* K, double* KtCent, int trTotal, int testTotal, double sumsumK) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd < trTotal) {
		double KtOne = 0.0;
		for (int j = 0; j < trTotal; j++)
			KtOne += Kt[rowInd * trTotal + j];
		double oneK = 0.0;
		for (int i = 0; i < trTotal; i++)
			oneK += K[i * trTotal + colInd];
		KtCent[rowInd * trTotal + colInd] = Kt[rowInd * trTotal + colInd] - (1.0 / trTotal) * KtOne - \
											(1.0 / trTotal) * oneK + (1.0 / (trTotal * trTotal)) * sumsumK;
	}
}

__global__ void kern_transform(double* train, double* test, double* Kx,
							   double* eigvecs, double* Kt, double* transTest, 
							   double sigma, int trTotal, int trTotalExt, 
							   int testTotal, int dim, int transDim) {
	make_tkern_device(train, test, Kt, sigma, trTotal, testTotal, dim);
	__syncthreads();
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

short big_to_little_endian(short inBig) {
	unsigned short b0,b1;
	short res;

	b0 = (inBig & 0x00ff) << 8u;
	b1 = (inBig & 0xff00) >> 8u;
	
	res = b0 | b1;
	
	return res;
}

int big_to_little_endian(int inBig) {
	unsigned int b0,b1,b2,b3;
	int res;

	b0 = (inBig & 0x000000ff) << 24u;
	b1 = (inBig & 0x0000ff00) << 8u;
	b2 = (inBig & 0x00ff0000) >> 8u;
	b3 = (inBig & 0xff000000) >> 24u;

	res = b0 | b1 | b2 | b3;
	return res;
}

float big_to_little_endian(float inBig) {
	union {
		int i;
		float f;
	} res;

	res.f = inBig;
	unsigned int b0,b1,b2,b3;

	b0 = (res.i & 0x000000ff) << 24u;
	b1 = (res.i & 0x0000ff00) << 8u;
	b2 = (res.i & 0x00ff0000) >> 8u;
	b3 = (res.i & 0xff000000) >> 24u;

	res.i = b0 | b1 | b2 | b3;
	return res.f;
}

void read_htk_header(int& nSamples, int& sampPeriod, short& sampSize, 
					short& parmKind, char* fname) {
	FILE* fp = fopen(fname, "rb");
	
	fread((void *)& nSamples, sizeof(int), 1, fp);
	nSamples = big_to_little_endian(nSamples);
	fread((void *)& sampPeriod, sizeof(int), 1, fp);
	sampPeriod = big_to_little_endian(sampPeriod);
	fread((void *)& sampSize, sizeof(short), 1, fp);
	sampSize = big_to_little_endian(sampSize);
	fread((void *)& parmKind, sizeof(short), 1, fp);
	parmKind = big_to_little_endian(parmKind);
	//fread((void *) params, sizeof(float), nSamples * sampSize / sizeof(float), fp);
	
	fclose(fp);
}

void read_htk_params(double* params, int testTotal, int dim, char* testfn) {
	FILE* fp = fopen(testfn, "rb");
	int buf;
	for (int i = 0; i < 3; i++) // just to skip 12 bytes of the header
		fread((void *) &buf, sizeof(int), 1, fp);
	
	for (int i = 0; i < testTotal * dim; i++) {
		float fbuf;
		fread((void *) &fbuf, sizeof(float), 1, fp);
		params[i] = (double) big_to_little_endian(fbuf);
		//printf("%d:, %.4f   %.4f   %.4f\n", i, fbuf, big_to_little_endian(fbuf), params[i]);
	}
	
	fclose(fp);
}

// Helper function for using CUDA to add vectors in parallel.
void transform(double *train, double *test, double* Kx, double *eigvecs, double *transTest,
				int trTotal, int trTotalExt, int testTotal, int dim, int transDim, double sigma,
				double *tKern, bool verbose, bool saveToFile)
{
	cudaError_t error;

	double * dTrain;
    double * dTest;
    double * dKx;
    double * dtKern;
    double * deigvecs;
    double * dTransTest;
    
    
    printf("%s\n", "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    
    error = cudaMalloc((void**) &dTrain, trTotal * dim * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("cudaMalloc dTrain returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
	error = cudaMalloc((void**) &dTest, testTotal * dim * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc dTest returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void**) &dKx, trTotal * trTotalExt * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc dKx returned error code %d, line(%d)\n", error, __LINE__);
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
    error = cudaMemcpy(dKx, Kx, trTotal * trTotalExt * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (Kx to dKx) returned error code %d, line(%d)\n", error, __LINE__);
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
    
    kern_transform<<<DimGrid, DimBlock>>>(dTrain, dTest, dKx, deigvecs, dtKern, dTransTest,
											sigma, trTotal, trTotalExt, testTotal, dim, transDim);
											
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
	if (saveToFile)	
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
	if (saveToFile)
		save_to_file(transTest, testTotal * transDim, "trans_test.bin");

	printf("%s\n", "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(dTrain);
	cudaFree(dTest);
	cudaFree(dKx);
	cudaFree(dtKern);
	cudaFree(deigvecs);
	cudaFree(transTest);
}


int main()
{
	double sigma = 19.63;
    int dim = 13;
    int transDim = dim;
    
    bool verbose = false;
    bool saveToFile = true;
    
    int trTotal = 1917;
    int trTotalExt = 3834;
    int testTotal = 10000;
    int tKernRows;
    int tKernCols;
    double * train; 
    double * test; 
    double * tKern; 
    double * eigvecs;
    double * transTest;
    double * Kx;
    int nSamples;
    int sampPeriod;
    short sampSize;
    short parmKind;
    
    printf("%s\n", "Importing data and creating memory on host");
    
    train = (double *) malloc(trTotal * dim * sizeof(double));
    read_file(train, trTotal * dim, "tr_data_subset.bin");
    /*fill_matr(train, trTotal, dim);
	if (verbose)
		print_matr(train, trTotal, dim);
	*/	
	if (saveToFile)
		save_to_file(train, trTotal * dim, "train.bin");
	
	Kx = (double *) malloc(trTotal * trTotalExt * sizeof(double));
	read_file(Kx, trTotal * trTotalExt, "Kx.bin");
	
	char testfn[] = "Word_44.mfc";
    read_htk_header(nSamples, sampPeriod, sampSize, parmKind, testfn);
    printf("%d %d %d %d\n", nSamples, sampPeriod, sampSize, parmKind);
    testTotal = nSamples;
    dim = sampSize / sizeof(float);
    test = (double *) malloc(testTotal * dim * sizeof(double));
    read_htk_params(test, testTotal, dim, testfn);
    
	/*fill_matr(test, testTotal, dim);
	if (verbose)
		print_matr(test, testTotal, dim);
	*/
	if (saveToFile)	
		save_to_file(test, testTotal * dim, "test.bin");
	
	eigvecs = (double *) malloc(trTotal * transDim * sizeof(double));
	read_file(eigvecs, trTotal * transDim, "eigvecs.bin");
	// fill_matr(eigvecs, trTotal, transDim);
	if (verbose)
		print_matr(eigvecs, trTotal, transDim);
	// save_to_file(eigvecs, trTotal * transDim, "eigvecs.bin");
	transTest = (double *) malloc(testTotal * transDim * sizeof(double));
	
	tKernRows = testTotal;
    tKernCols = trTotal;
    //@@ Allocate the hostC matrix
    printf("%s\n", "Importing data and creating memory on host");
	tKern = (double *) malloc(tKernRows * tKernCols * sizeof(double));
	transTest = (double *) malloc(testTotal * dim * sizeof(double));

    printf("%s %d %s %d\n", "The dimensions of train are ", trTotal, " x ", dim);
    printf("%s %d %s %d\n", "The dimensions of test are ", testTotal, " x ", dim);

	transform(train, test, Kx, eigvecs, transTest, 
			  trTotal, trTotal, testTotal, dim, transDim, 
			  sigma, tKern, verbose, saveToFile);
    
    free(train);
    free(Kx);
    free(test);
    free(tKern);
    free(eigvecs);
    free(transTest);
	    
    return 0;
}

