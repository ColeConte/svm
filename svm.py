'''
Cole Conte CSE 512 HW 2
svm.py
'''

import argparse
import pandas as pd
import numpy as np
import struct

def readIdxFile(x,y):
	'''https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format'''
	with open(x,'rb') as f:
		magic, size = struct.unpack(">II", f.read(8))
		nrows, ncols = struct.unpack(">II", f.read(8))
		xData = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
		xData = xData.reshape((size, nrows, ncols))
		f.close()
	with open(y,'rb') as f:
		magic, size = struct.unpack(">II", f.read(8))
		nrows, ncols = struct.unpack(">II", f.read(8))
		yData = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
		f.close()
	return xData, yData

def svm():
	pass


parser = argparse.ArgumentParser()
parser.add_argument("--kernel")
parser.add_argument("--dataset")
parser.add_argument("--cost")
parser.add_argument("--train")
parser.add_argument("--test")
parser.add_argument("--output")
args = parser.parse_args()


if(args.dataset == "mnist"):
	trainImgs, trainLbls = readIdxFile("train-images-idx3-ubyte","train-labels-idx1-ubyte")
	testImgs, testLbls = readIdxFile("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")
