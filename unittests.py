# !c:\Anaconda\python

import sys, os, json, pickle
import numpy as np
import mlpy

def vsum(v, w): return [elv + elw for (elv, elw) in zip(v, w)]

def scal(alpha, v): return [alpha * el for el in v]

def dot(v, w): return sum([elv * elw for (elv, elw) in zip(v, w)])

def centered_kernel_matrix(Kt, KSub):
    """
    Center samples in the Reproduced Hilbert Space in Kt() on training kernel
    matrix KSub
    Kt = np.random.rand(testTot, trTot)
    KSub = np.random.rand(trTot, trTot)
    """
    assert Kt.shape[1] == KSub.shape[0]
    # Kt.shape = (10, 2), KSub.shape = (2, 4)
    trTotSub = KSub.shape[0]
    trTot = KSub.shape[1]
    KtMan = np.zeros(Kt.shape)
    for i in range(KtMan.shape[0]):
        for j in range(KtMan.shape[1]):
            KtOne = sum(Kt[i, :])
            oneK = sum(KSub[j, :])
            sumsumK = sum(sum(KSub))
            KtMan[i, j] = Kt[i, j] - (1.0/trTotSub) * KtOne - (1.0/trTot) * oneK + (1.0 / (trTotSub * trTot)) * sumsumK
    return KtMan

def read_config(fn=""):
    config = {}
    config["sigma"] = 19.63;
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

    print "test x train = ", str(nTest), " x ", str(nTrain)
    tKernEstimed = [[np.exp(-0.5 * np.dot(trEl-testEl, trEl-testEl)/sigma**2) for trEl in train] for testEl in test]

    tkern = freader(tkernfn, nTrain)

    assert np.allclose(tKernEstimed, tkern)


class NLTransformer(object):
    def __init__(self, trainfn, Kxfn, trTotalExt, eigvecsfn, workdir, config):
        dim = config["dim"]
        transDim = config["transDim"]
        dt = config["dt"]
        self.sigma = config["sigma"]
        freader = file_reader(dt, workdir)
        self.Kx = freader(Kxfn, trTotalExt)
        self.eigvecs = freader(eigvecsfn, transDim)
        self.train = freader(trainfn, dim)
        self.nTrain = self.train.shape[0]
        
    def transform(self, data):
        tKernEstimed = np.array([[np.exp(-0.5 * np.dot(trEl-el, trEl-el)/self.sigma**2) for trEl in self.train] for el in data])
        tKernEstimed = mlpy.kernel_center(tKernEstimed, self.Kx.T)
        return np.dot(tKernEstimed, self.eigvecs)
    
        

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
    Kxfn = "Kx.bin"

    transTest = freader(transfn, transDim)
    nTrTest = transTest.shape[0]
    
    f = open("trans_test", "w")
    f.write(str(transTest))
    f.close()
    f = open("trans_test.pkl", 'w')
    pickle.dump(transTest, f)
    f.close()

    test = freader(testfn, dim)
    nTest = test.shape[0]
    trTotalExt = 3834
    tformer = NLTransformer(trainfn, Kxfn, trTotalExt, eigvecsfn, \
                                workdir, config)
    transTestEstimed = tformer.transform(test)

    f = open("trans_test_estimed", "w")
    f.write(str(transTestEstimed))
    f.close()
    f = open("trans_test_estimed.pkl", 'w')
    pickle.dump(transTestEstimed, f)
    f.close()

    assert np.allclose(transTest, transTestEstimed, atol=1e-5)

def test_centering():
    workdir = os.path.join("tkernel", "tkernel")
    config = read_config()

    freader = file_reader(config["dt"], workdir)

    testfn = "test.bin"
    trainfn = "train.bin"
    tkernfn = "tkern.bin"
    Kxfn = "Kx.bin"
    
    train = freader(trainfn, config["dim"])
    nTrain = train.shape[0]

    test = freader(testfn, config["dim"])
    nTest = test.shape[0]

    sigma = config["sigma"]
    tKernEstimed = np.array([[np.exp(-0.5 * np.dot(trEl-testEl, trEl-testEl)/sigma**2) for trEl in train] for testEl in test])

    trTotalExt = 3834
    Kx = freader(Kxfn, trTotalExt)
    tKernEstimedCent = mlpy.kernel_center(tKernEstimed, Kx.T)

    gpuTKernCent = freader("t_kern_centr.bin", nTrain)

    assert np.allclose(tKernEstimedCent, gpuTKernCent, atol=1e-5)


def main():
    args = sys.argv[1:]
    
    promptMessage = "--test" 
    if not args:
        print promptMessage
    #    return
    
    if True: #args[0] == '--test':
        test_tkernel()
        test_trans_data()
        test_centering()


if __name__ == '__main__':
    main()
