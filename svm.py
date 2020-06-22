'''
Cole Conte CSE 512 HW 2
svm.py
'''

import argparse
import pandas as pd
import numpy as np
import struct
import random

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

def trainTestSplit(df):
	random.seed(13579)
	print("Splitting data into 80% training, 20% testing.\n")
	df["trainTest"] = [random.random() for i in range(len(df))]
	test = df[df["trainTest"]>=0.8]
	test = test.drop(labels="trainTest",axis=1)
	train = df[df["trainTest"]<0.8]
	train = train.drop(labels="trainTest",axis=1)
	return train,test

def svmTrain(train,cost):
	'''Train SVM algorithm'''
	m = len(train)
	theta = np.zeros(len(train.columns)-1)
	ws = np.empty([1,len(train.columns)-1])
	T = 1000
	for t in range(1,T+1):
		w = np.array([(1.0/(cost*t))*theta])
		ws = np.append(ws,w,axis=0)
		i = random.randint(0,m-1)
		if train.iloc[i,-1]*np.dot(w,train.iloc[i,:-1]) < 1:
			theta = theta + train.iloc[i,-1]*train.iloc[i,:-1]
	return ws.mean(0)

def svmTest(test,w):
	'''Test SVM algorithm'''
	#test["predicted"] = test.map(lambda x: -1 if x.iloc[:,-1]*np.dot(w,x.iloc[:,:-1]) < 0 else 1)


parser = argparse.ArgumentParser()
parser.add_argument("--kernel")
parser.add_argument("--dataset")
parser.add_argument("--cost")
parser.add_argument("--output")
args = parser.parse_args()


#TODO: Implement CL interface asking for train/test file paths
if(args.dataset == "mnist"):
	trainImgs, trainLbls = readIdxFile("train-images-idx3-ubyte","train-labels-idx1-ubyte")
	testImgs, testLbls = readIdxFile("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")
	#Convert to pandas DFs
else:
	data = pd.read_csv('Breast_cancer_data.csv')
	train,test = trainTestSplit(data)
	weightVector = svmTrain(train,float(args.cost))
	svmTest(test,weightVector)
