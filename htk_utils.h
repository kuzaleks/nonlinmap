
void read_htk_header(int& nSamples, int& sampPeriod, short& sampSize, 
		     short& parmKind, char* fname); 
void read_htk_params(double* params, int testTotal, int dim, char* testfn);
void write_htk_params(double* params, int nSamples, int sampPeriod, short sampSize, short parmKind, char* trDatafn);
