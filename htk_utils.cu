#include "htk_utils.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>

#define FLOAT_SIZE 4

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

void write_htk_params(double* params, int nSamples, int sampPeriod, short sampSize, short parmKind, char* trDatafn) {
	FILE* fp = fopen(trDatafn, "wb");
	
//	fwrite((void *) arr, sizeof(double), n, fp);
	nSamples = reversed_bytes_order(nSamples);
	fwrite((void *)& nSamples, sizeof(int), 1, fp);

	sampPeriod = reversed_bytes_order(sampPeriod);
	fwrite((void *)& sampPeriod, sizeof(int), 1, fp);

	sampSize = reversed_bytes_order(sampSize);
	fwrite((void *)& sampSize, sizeof(short), 1, fp);

	parmKind = reversed_bytes_order(parmKind);
	fwrite((void *)& parmKind, sizeof(short), 1, fp);
	
	// Reverse it back to be utilized in the following loop
	nSamples = reversed_bytes_order(nSamples);
	sampSize = reversed_bytes_order(sampSize);
	for (int i = 0; i < nSamples * (int)(sampSize / FLOAT_SIZE); i++) {
		float fbuf = reversed_bytes_order((float) params[i]);
		fwrite((void *) &fbuf, sizeof(float), 1, fp);
		//printf("%d:, %.4f   %.4f   %.4f\n", i, fbuf, big_to_little_endian(fbuf), params[i]);
	}
	
	fclose(fp);
}