#include "htk_utils.h"
#include "utils.h"

#include <stdlib.h>
#include <stdio.h>

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