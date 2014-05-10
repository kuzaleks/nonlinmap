# !c:\Anaconda\python

import sys, os
import numpy as np

def vsum(v, w): return [elv + elw for (elv, elw) in zip(v, w)]

def scal(alpha, v): return [alpha * el for el in v]

def dot(v, w): return sum([elv * elw for (elv, elw) in zip(v, w)])



def test_tkernel():
        
    workdir = os.path.join("tkernel", "tkernel")
    dim = 13
    dt = np.dtype('<d')

    def read_bin(fname, ncol):
        f = open(os.path.join(workdir, fname))
        train = np.fromfile(f, dtype=dt)
        f.close()
        return np.reshape(train, (len(train) / ncol, ncol))

    trainfn = "train.bin"
    testfn = "test.bin"
    tkernfn = "tkern.bin"
    
    train = read_bin(trainfn, dim)
    nTrain = train.shape[0]

    test = read_bin(testfn, dim)
    nTest = test.shape[0]

    sigma = 4.0
    tKernEstimed = [[np.exp(-0.5 * np.dot(trEl-testEl, trEl-testEl)/sigma**2) for trEl in train] for testEl in test]

    tkern = read_bin(tkernfn, nTrain)

    assert np.allclose(tKernEstimed, tkern)


def main():
    args = sys.argv[1:]
    
    promptMessage = "--test" 
    if not args:
        print promptMessage
        return
    
    if args[0] == '--test':
        test_tkernel()


if __name__ == '__main__':
    main()
