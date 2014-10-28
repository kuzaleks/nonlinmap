
import sys, os

from struct import unpack

import numpy as np

class ParamCollector(object):
    """
    Extracts features from parameter file and stores it in matrix
    """
    def __init__(self, paramFileName):
        self.paramFileName = paramFileName
        self.header = {} 
        self.sampleMatrix = []
        
    def htk_param_header(self, fileObj):
        """Read and returns header of the htk parameter file (e.g. *.mfc)"""
        
        block = fileObj.read(4)
        self.header['nSamples'] = unpack('>l', block )[0]
        block = fileObj.read(4)
        self.header['nPeriod'] = unpack('>l', block )[0]
        block = fileObj.read(2)
        self.header['sampSize'] = unpack('>h', block)[0]
        block = fileObj.read(2)
        self.header['sampKind'] = unpack('>h', block )[0]

    def next_sample_from_file(self, fileObject, sampleSize):
        """Reads from htk parameter file and return one sample"""
        FLOAT_SIZE = 4

        sample = []
        for smplCompIdx in range(sampleSize):
            compData = fileObject.read(FLOAT_SIZE)
            sample.append(unpack('>f', compData)[0])    # [0] becouse it returns tuple
        return sample

    def store_sample_matrix(self):
        """
        Reads from htk parameter file samples, 
        which are used to form the sample matrix.
        """

        fileObj = open(self.paramFileName, 'rb')
        try:
            self.htk_param_header(fileObj)
            FLOAT_SIZE = 4

            for smplInd in range(self.header['nSamples']):
                sample = self.next_sample_from_file(fileObj, 
                                                    self.header['sampSize'] // FLOAT_SIZE)
                self.sampleMatrix.append(sample)
        finally:
            fileObj.close()
