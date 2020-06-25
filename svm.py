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
	'''Referenced:
	https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format
	'''
	with open(x,'rb') as f:
		magic, size = struct.unpack(">II", f.read(8))
		nrows, ncols = struct.unpack(">II", f.read(8))
		xData = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
		xData = xData.reshape((size, nrows, ncols))
		f.close()
	with open(y,'rb') as f:
		magic, size = struct.unpack(">II", f.read(8))
		yData = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
		f.close()
	return xData, yData

def svmTrain(train,lam,T):
	'''Train SVM algorithm for given training set and cost hyperparameter, and compute
	training error.'''
	#random.seed(123)
	m = len(train)
	theta = pd.DataFrame(0,index=np.arange(1),columns=train.columns[:-1])
	ws = pd.DataFrame(columns=train.columns[:-1])
	for t in range(1,T+1):
		w = pd.DataFrame((1.0/(lam*t))*theta)
		ws = ws.append(w)
		i = random.randint(0,m-1)
		if (train.iloc[i,-1] * w.dot(train.iloc[i,:-1]) < 1).bool():
			theta = theta + train.iloc[i,-1]*train.iloc[i,:-1]
	w = ws.mean(0)
	train["hypothesis"] = train.iloc[:,-1]*train.iloc[:,:-1].dot(w)
	print(train)
	train["hypothesis"] = train["hypothesis"].map(lambda x: -1.0 if x>0 else 1.0)
	errors = float(len(train[train["hypothesis"] != train.iloc[:,-2]]))
	print("Training error is " + str(errors/len(train)*100) +"%")
	return w

def svmTest(test,w):
	'''Compute testing error for given test set and weight vector.'''
	test["hypothesis"] = test.iloc[:,-1]*test.iloc[:,:-1].dot(w)
	test["hypothesis"] = test["hypothesis"].map(lambda x: -1.0 if x>0 else 1.0)
	errors = float(len(test[test["hypothesis"] != test.iloc[:,-2]]))
	print("Testing error is " + str(errors/len(test)*100) +"%")

parser = argparse.ArgumentParser()
parser.add_argument("--kernel")
parser.add_argument("--dataset")
parser.add_argument("--cost")
parser.add_argument("--lam")
parser.add_argument("--T")
parser.add_argument("--output")
args = parser.parse_args()


#TODO: Implement CL interface asking for train/test file paths
#Default parameters if not included in CL
if(args.lam == None):
	lam = 1.0
else:
	lam = float(args.lam)
if(args.T == None):
	T = 1000
else:
	T = int(args.T)
if (args.output == None):
	output = "output.txt"
else:
	output = args.output

f = open(output, "a")
if(args.dataset == "mnist"):
	trainImgs, trainLbls = readIdxFile("train-images-idx3-ubyte","train-labels-idx1-ubyte")
	testImgs, testLbls = readIdxFile("t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")
	train = trainImgs.reshape(trainImgs.shape[0],-1)
	test = testImgs.reshape(testImgs.shape[0],-1)
	train = pd.DataFrame(train)
	test = pd.DataFrame(test)
	train["y"] = trainLbls
	test["y"] = testLbls
	for i in range(10):
		trainCopy = train.copy()
		testCopy = test.copy()
		trainCopy.iloc[:,-1] = trainCopy.iloc[:,-1].map(lambda y: 1 if y == i  else -1)
		testCopy.iloc[:,-1] = testCopy.iloc[:,-1].map(lambda y: 1 if y == i else -1)
		weightVector = svmTrain(trainCopy,float(lam),T)
		svmTest(testCopy,weightVector)
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			f.write(str(weightVector))
		

elif(args.dataset == "bcd"):
	train = pd.read_csv('Breast_cancer_data.csv')
	#Represent success and failure as -1 and 1
	train.iloc[:,-1] = train.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
	weightVector = svmTrain(train,float(lam),T)
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		f.write(str(weightVector))

f.close()
