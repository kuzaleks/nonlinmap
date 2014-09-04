
void save_to_file(double* arr, int n, const char* fname);
void read_file(double* arr, int n, const char* fname);
void fill_matr(double* M, int nrow, int ncol);

void print_matr(double* M, int nrow, int ncol);
short big_to_little_endian(short inBig);
int big_to_little_endian(int inBig);
float big_to_little_endian(float inBig);
void base_name(char* path, char* dest);

int reversed_bytes_order(int genuin);
float reversed_bytes_order(float genuin);
short reversed_bytes_order(short genuin);