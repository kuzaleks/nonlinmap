
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "htk_utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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
#define STR_MAX_LEN 64

// Compute C = A * B
__global__ void matrixMultiply(double * A, double * B, double * C,
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

__device__ double kernel_rbf(double sigma, double* vArr, double* wArr, 
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

__device__ double kernel_lin(double sigma, double* vArr, double* wArr, 
							int trSampleInd, int testSampleInd, int dim) {
	double cVal = 0.0;
	double vEl, wEl;
	for (int i = 0; i < dim; ++i) {
		vEl = vArr[trSampleInd * dim + i];
		wEl = wArr[testSampleInd * dim + i];
		cVal += vEl * wEl;
	}

	return cVal;
}

__global__ void make_tkern_device(double* train, double* test, double* Kt,
								double sigma,
								int trTotal, int testTotal, int dim) {
								
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd < trTotal) {
		Kt[rowInd*trTotal + colInd] = kernel_rbf(sigma, train, test, colInd, rowInd, dim);
	}
}

__global__ void estim_row_sums(double* Kt, double* K, double* KtRowsSums, double* KRowsSums,
								int trTotal, int trTotalExt, int testTotal) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd == 0) {
		double KtOne = 0.0;
		for (int j = 0; j < trTotal; j++)
			KtOne += Kt[rowInd * trTotal + j];
		KtRowsSums[rowInd] = KtOne;
	}
	if (rowInd == 0 && colInd < trTotal) {
		double oneK = 0.0;
		for (int i = 0; i < trTotalExt; i++)
			oneK += K[colInd * trTotalExt + i];
		KRowsSums[colInd] = oneK;
	}
}

__global__ void center(double* Kt, double* K, double* KtCent, double* KtRowsSums, double* KRowsSums, 
						int trTotal, int trTotalExt, int testTotal, double sumsumK) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	
	int rowInd = by * blockDim.y + ty;
	int colInd = bx * blockDim.x + tx;
	
	if (rowInd < testTotal && colInd < trTotal) {
		double KtOne = 0.0;
		//for (int j = 0; j < trTotal; j++)
		//	KtOne += Kt[rowInd * trTotal + j];
		double oneK = 0.0;
		//for (int i = 0; i < trTotalExt; i++)
		//	oneK += K[colInd * trTotalExt + i]; 
		KtOne = KtRowsSums[rowInd];
		oneK = KRowsSums[colInd];
		KtCent[rowInd * trTotal + colInd] = Kt[rowInd * trTotal + colInd] - (1.0 / trTotal) * KtOne - \
											(1.0 / trTotalExt) * oneK + (1.0 / (trTotal * trTotalExt)) * sumsumK;
	}
}

__global__ void kern_transform(double* train, double* test, double* Kx,
							   double* eigvecs, double* Kt, double * KtCent, double* transTest, 
							   double* KtRowsSums, double* KRowsSums,
							   double sigma, int trTotal, int trTotalExt,
							   int testTotal, int dim, int transDim, double sumsumK) {
	make_tkern_device(train, test, Kt, sigma, trTotal, testTotal, dim);
	__syncthreads();
	estim_row_sums(Kt, Kx, KtRowsSums, KRowsSums, trTotal, trTotalExt, testTotal);
	__syncthreads();
	center(Kt, Kx, KtCent, KtRowsSums, KRowsSums, trTotal, trTotalExt, testTotal, sumsumK);
	__syncthreads();
	matrixMultiply(KtCent, eigvecs, transTest,
			             testTotal, trTotal,
			             trTotal, transDim,
			             testTotal, transDim);
	//__syncthreads();		             
}

// Helper function for using CUDA to add vectors in parallel.
void transform(double *train, char *datafn, char codetrfn[], double* Kx, double *eigvecs, 
				int trTotal, int trTotalExt, int dim, int transDim, double sigma,
				bool verbose, bool saveToFile)
{
	cudaError_t error;

	double * data;

	double * dTrain;
    double * dData;
    double * dKx;
    double * dtKern;
    double * tKernCentr;
    double * dtKernCentr;
    double * deigvecs;
    
    double * transData;
    double * dTransData;
    
    double * dKtRowsSums;
    double * dKRowsSums;
    double sumsumKx;
    
    if (verbose)
		printf("%s\n", "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    
    error = cudaMalloc((void**) &dTrain, trTotal * dim * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("cudaMalloc dTrain returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    sumsumKx = 0.0;
    for (int i = 0; i < trTotal; i++)
		for (int j = 0; j < trTotalExt; j++)
			sumsumKx += Kx[i*trTotalExt + j];
	if (verbose)
		printf("sumsumKx = %3.8f\n", sumsumKx);
    error = cudaMalloc((void**) &dKx, trTotal * trTotalExt * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc dKx returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void**) &deigvecs, trTotal * transDim * sizeof(double));
	if (error != cudaSuccess)
    {
        printf("cudaMalloc deigvecs returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
    
	if (verbose)    
		printf("%s\n", "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	error = cudaMemcpy(dTrain, train, trTotal * dim * sizeof(double), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
    {
        printf("cudaMemcpy (train, dTrain) returned error code %d, line(%d)\n", error, __LINE__);
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
    error = cudaMalloc((void**) &dKtRowsSums, trTotal * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("cudaMalloc dKtRowsSums returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void**) &dKRowsSums, trTotal * sizeof(double));
    if (error != cudaSuccess)
    {
        printf("cudaMalloc dKRowsSums returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }
    
	if (verbose)
		printf("%s\n", "Read data to be transformed");
    
	FILE *infile = fopen(codetrfn, "r");       
	if (infile == NULL) {
		printf("Error! Unable to open list of files %s.\n", codetrfn); 
		return;
	}

	char* line = (char*) malloc((STR_MAX_LEN + 1) * sizeof(char));
	while (fscanf(infile, "%s", line) != EOF) {

		printf("%s\n", line);

		int nSamples, sampPeriod;
		short sampSize, parmKind;
		read_htk_header(nSamples, sampPeriod, sampSize, parmKind, line);
		if (verbose)
			printf("%d %d %d %d\n", nSamples, sampPeriod, sampSize, parmKind);
		int dataTotal = nSamples;
		if (dim != sampSize / sizeof(float)) {
			printf("Error! train data dim %d not equal to input data dim %d\n", dim, sampSize / sizeof(float));
			//@@ Free the GPU memory here

			cudaFree(dTrain);	
			cudaFree(dKx);
			cudaFree(deigvecs);
			cudaFree(dKtRowsSums);
			cudaFree(dKRowsSums);
			return;
		}
		
		data = (double *) malloc(dataTotal * dim * sizeof(double));
		read_htk_params(data, dataTotal, dim, line);
    
		if (saveToFile)	
			save_to_file(data, dataTotal * dim, "data.bin");

		error = cudaMalloc((void**) &dData, dataTotal * dim * sizeof(double));
		if (error != cudaSuccess)
		{
			printf("cudaMalloc dData returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
		error = cudaMemcpy(dData, data, dataTotal * dim * sizeof(double), cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			printf("cudaMemcpy (data to dData) returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
    
		error = cudaMalloc((void**) &dTransData, dataTotal * transDim * sizeof(double));
		if (error != cudaSuccess)
		{
			printf("cudaMalloc dTransData returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
    
		error = cudaMalloc((void**) &dtKern, dataTotal * trTotal * sizeof(double));
		if (error != cudaSuccess)
		{
			printf("cudaMalloc dtKern returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
		error = cudaMalloc((void**) &dtKernCentr, dataTotal * trTotal * sizeof(double));
		if (error != cudaSuccess)
		{
			printf("cudaMalloc dtKernCentr returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
    
		//@@ Initialize the grid and block dimensions here
		dim3 DimGrid((trTotal - 1)/THREADS_IN_BLOCK + 1, (dataTotal - 1)/THREADS_IN_BLOCK + 1, 1);
		dim3 DimBlock(THREADS_IN_BLOCK, THREADS_IN_BLOCK, 1);

		if (verbose)
			printf("%s\n", "Performing CUDA computation");
		//@@ Launch the GPU Kernel here
		// Allocate CUDA events that we'll use for timing
	
		cudaEvent_t start;
		error = cudaEventCreate(&start);

		cudaEvent_t stop;
		error = cudaEventCreate(&stop);

		// Record the start event
		error = cudaEventRecord(start, NULL);
    
		make_tkern_device<<<DimGrid, DimBlock>>>(dTrain, dData, dtKern, sigma, trTotal, dataTotal, dim);
		estim_row_sums<<<DimGrid, DimBlock>>>(dtKern, dKx, dKtRowsSums, dKRowsSums, trTotal, trTotalExt, dataTotal);
		center<<<DimGrid, DimBlock>>>(dtKern, dKx, dtKernCentr, dKtRowsSums, dKRowsSums, trTotal, trTotalExt, dataTotal, sumsumKx);
		matrixMultiply<<<DimGrid, DimBlock>>>(dtKernCentr, deigvecs, dTransData, dataTotal, trTotal,
												trTotal, transDim, dataTotal, transDim);
		/*kern_transform<<<DimGrid, DimBlock>>>(dTrain, dData, dKx, deigvecs, dtKern, dtKernCentr, dTransData,
											  dKtRowsSums, dKRowsSums,
											  sigma, trTotal, trTotalExt, dataTotal, dim, transDim, sumsumKx);
			*/								
		//cudaThreadSynchronize();
		cudaDeviceSynchronize();
		// Record the stop event
		error = cudaEventRecord(stop, NULL);
    
		error = cudaEventSynchronize(stop);
	
		float msecTotal = 0.0f;
		error = cudaEventElapsedTime(&msecTotal, start, stop);
		printf("Performance: Time= %.3f msec\n", msecTotal);
	
	
		if (verbose)
			printf("%s\n", "Copying output memory to the CPU");
		//@@ Copy the GPU memory back to the CPU here
    
		if (saveToFile) {
			int tKernRows = dataTotal;
			int tKernCols = trTotal;
			double * tKern = (double *) malloc(tKernRows * tKernCols * sizeof(double));
			error = cudaMemcpy(tKern, dtKern, dataTotal * trTotal * sizeof(double), cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				printf("cudaMemcpy (dtKern to tKern) returned error code %d, line(%d)\n", error, __LINE__);
				exit(EXIT_FAILURE);
			}
			
			save_to_file(tKern, dataTotal * trTotal, "tkern.bin");
			free(tKern);
		}
		transData = (double *) malloc(dataTotal * transDim * sizeof(double));
		error = cudaMemcpy(transData, dTransData, dataTotal * transDim * sizeof(double), cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			printf("cudaMemcpy (dTransData to transData) returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
		if (saveToFile) {
			save_to_file(transData, dataTotal * transDim, "trans_test.bin");
		}

		char path[STR_MAX_LEN + 1] = "data/transgpu/";
		char fn[STR_MAX_LEN + 1];
		base_name(line, fn);
		strcat(path, fn);
		save_to_file(transData, dataTotal * transDim, path);

		if (saveToFile) {
			tKernCentr = (double *) malloc(dataTotal * trTotal * sizeof(double));
			error = cudaMemcpy(tKernCentr, dtKernCentr, dataTotal * trTotal * sizeof(double), cudaMemcpyDeviceToHost);
			if (error != cudaSuccess)
			{
				printf("cudaMemcpy (dtKernCentr to tKernCentr) returned error code %d, line(%d)\n", error, __LINE__);
				exit(EXIT_FAILURE);
			}
		
			save_to_file(tKernCentr, dataTotal * trTotal, "t_kern_centr.bin");
			free(tKernCentr);
		}
		if (verbose)
			printf("%s\n", "Freeing GPU Memory");
	
		//@@ Free the GPU memory here
		cudaFree(dData);
		cudaFree(dTransData);
		cudaFree(dtKern);
		cudaFree(dtKernCentr);

		free(data);
		free(transData);
	}
	cudaFree(dTrain);
	
	cudaFree(dKx);
	
	cudaFree(deigvecs);
	
	cudaFree(dKtRowsSums);
	cudaFree(dKRowsSums);
	
	if (verbose)
		printf("%s\n", "Freeing RAM Memory");
	free(line);
	
}


int main()
{
	char codetrfn[] = "codetr.scp";

	double sigma = 19.63;
    int dim = 13;
    int transDim = dim;
    
    bool verbose = true;
    bool saveToFile = true;
    
    int trTotal = 1917;
    int trTotalExt = 3834;
   // int dataTotal = 10000;
    double * train; 
    double * eigvecs;
    double * Kx;
    

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
	
    
	eigvecs = (double *) malloc(trTotal * transDim * sizeof(double));
	read_file(eigvecs, trTotal * transDim, "eigvecs.bin");
	
    printf("%s\n", "Importing data and creating memory on host");
	
    printf("%s %d %s %d\n", "The dimensions of train are ", trTotal, " x ", dim);
    
	char testfn[] = "Word_44.mfc";

	transform(train, testfn, codetrfn, Kx, eigvecs,
			  trTotal, trTotalExt, dim, transDim, 
			  sigma, verbose, saveToFile);
    
    free(train);
    free(Kx);
    
    
    free(eigvecs);
	    
    return 0;
}

