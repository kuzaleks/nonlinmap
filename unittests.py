# !c:\Anaconda\python

import sys, os
import numpy as np

def vsum(v, w): return [elv + elw for (elv, elw) in zip(v, w)]

def scal(alpha, v): return [alpha * el for el in v]

def dot(v, w): return sum([elv * elw for (elv, elw) in zip(v, w)])

def read_config(fn=""):
    config = {}
    config["sigma"] = 4.0;
    config["dim"] = 13;
    config["transDim"] = 13;
    config["dt"] = np.dtype('<d')
    return config

def file_reader(dt, workdir):
    def read_bin(fname, ncol):
        f = open(os.path.join(workdir, fname), 'rb')
        train = np.fromfile(f, dtype=dt)
        f.close()
        return np.reshape(train, (len(train) / ncol, ncol))
    return read_bin

def test_tkernel():
        
    workdir = os.path.join("tkernel", "tkernel")
    config = read_config()
    dim = config["dim"]
    dt = config["dt"]
    sigma = config["sigma"]

    freader = file_reader(dt, workdir)

    trainfn = "train.bin"
    testfn = "test.bin"
    tkernfn = "tkern.bin"
    
    train = freader(trainfn, dim)
    nTrain = train.shape[0]

    test = freader(testfn, dim)
    nTest = test.shape[0]


    tKernEstimed = [[np.exp(-0.5 * np.dot(trEl-testEl, trEl-testEl)/sigma**2) for trEl in train] for testEl in test]

    tkern = freader(tkernfn, nTrain)

    assert np.allclose(tKernEstimed, tkern)

def test_trans_data():
    workdir = os.path.join("tkernel", "tkernel")
    config = read_config()
    dim = config["dim"]
    transDim = config["transDim"]
    dt = config["dt"]
    sigma = config["sigma"]

    freader = file_reader(dt, workdir)

    trainfn = "train.bin"
    testfn = "test.bin"
    tkernfn = "tkern.bin"
    eigvecsfn = "eigvecs.bin"
    transfn = "trans_test.bin"
    
    train = freader(trainfn, dim)
    nTrain = train.shape[0]

    test = freader(testfn, dim)
    nTest = test.shape[0]

    tKernEstimed = [[np.exp(-0.5 * np.dot(trEl-testEl, trEl-testEl)/sigma**2) for trEl in train] for testEl in test]
    
    eigvecs = freader(eigvecsfn, transDim)
    transTest = freader(transfn, transDim)
        
    transTestEstimed = np.dot(tKernEstimed, eigvecs)
    assert np.allclose(transTest, transTestEstimed)

def main():
    args = sys.argv[1:]
    
    promptMessage = "--test" 
    if not args:
        print promptMessage
        return
    
    if args[0] == '--test':
        test_tkernel()
        test_trans_data()


if __name__ == '__main__':
    main()
