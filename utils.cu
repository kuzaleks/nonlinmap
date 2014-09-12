
#include "utils.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

void base_name(char* path, char* dest) {
	char* from = strrchr(path, '/');
	if (from == NULL)
		from = strrchr(path, '\\') + 1;
	strcpy(dest, from);
}

int reversed_bytes_order(int genuin) {
	return big_to_little_endian(genuin);
}

float reversed_bytes_order(float genuin) {
	return big_to_little_endian(genuin);
}

short reversed_bytes_order(short genuin) {
	return big_to_little_endian(genuin);
}