'''
Cole Conte CSE 512 HW 2
svm.py
'''

import argparse
import pandas as pd
import numpy as np
import struct
import random
import math

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

def svmTrain(train,lam,T,sigma):
	'''Train SVM algorithm for given training set and cost hyperparameter, and compute
	training error.'''
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
	train["hypothesis"] = train["hypothesis"].map(lambda x: -1.0 if x>0 else 1.0)
	errors = float(len(train[train["hypothesis"] != train.iloc[:,-2]]))
	print("Training error is " + str(errors/len(train)*100) +"%")
	return w,(errors/len(train)*100)

def svmTest(test,w):
	'''Compute testing error for given test set and weight vector.'''
	test["hypothesis"] = test.iloc[:,-1]*test.iloc[:,:-1].dot(w)
	test["hypothesis"] = test["hypothesis"].map(lambda x: -1.0 if x>0 else 1.0)
	errors = float(len(test[test["hypothesis"] != test.iloc[:,-2]]))
	print("Testing error is " + str(errors/len(test)*100) +"%")
	return (errors/len(test)*100)

def rbfTrain(train,lam,T,sigma):
	'''Train SVM algorithm with radial basis function kernel for given training set
	and cost hyperparameter, and compute training error.'''
	m = len(train)
	beta = pd.DataFrame(0,index=np.arange(m),columns=["val"])
	alphas = pd.DataFrame(index=np.arange(m))
	psi = train.apply(lambda x: gaussianFunction(x,sigma),axis=1)
	for t in range(1,T+1):
		alpha = pd.DataFrame((1.0/(lam*t))*beta)
		i = random.randint(0,m-1)
		ifSum = rbfKernel(psi,i,alpha,sigma)
		if(train.iloc[i,-1]*ifSum < 1):
			beta.iloc[i,0] = float(beta.iloc[i,0]) + train.iloc[i,-1] #this right?
		alphas[str(t)] = alpha
	alphas["mean"] = alphas.mean(axis=1)
	wbar = []
	for j in range(len(alphas)):
		wbar += [alphas["mean"].loc[j] * psi.loc[j]]

	wOutput = dimReduce(wbar,train.iloc[:,:-1],len(train.columns)-1,sigma)
	print(wOutput)
	train["hypothesis"] = train.iloc[:,-1]*train.iloc[:,:-1].dot(wOutput.loc[0])
	print(train)
	train["hypothesis"] = train["hypothesis"].map(lambda x: -1.0 if x<0 else 1.0)
	errors = float(len(train[train["hypothesis"] != train.iloc[:,-2]]))
	print("Training error is " + str(errors/len(train)*100) +"%")
	print(wOutput)
	return wOutput, (errors/len(train)*100)

def rbfTest(test,w):
	'''Compute testing error for given test set and weight vector.'''
	test["hypothesis"] = test.iloc[:,-1]*test.iloc[:,:-1].dot(w.loc[0])
	test["hypothesis"] = test["hypothesis"].map(lambda x: -1.0 if x<0 else 1.0)
	errors = float(len(test[test["hypothesis"] != test.iloc[:,-2]]))
	print("Testing error is " + str(errors/len(test)*100) +"%")
	return (errors/len(test)*100)

def dimReduce(wbar,train,newDim,sigma):
	'''reduces dimension of weight vector'''
	ws = []
	for i in range(newDim):
		psis = []
		xi = train.iloc[i,:]
		for j in range(len(train)):
			gf = gaussianFunction(xi-train.iloc[j,:],sigma)
			psis+=[gf]
		psis = pd.DataFrame(psis)
		result = psis.T.dot(wbar)
		ws+=[float(result)]
	retVal = pd.DataFrame(columns=train.columns)
	retVal.loc[0] = ws
	return retVal

def rbfKernel(psi,i,alpha,sigma):
	'''computes rbf kernel condition'''
	ifSum = 0.0
	psii = psi.loc[i]
	for j in range(len(train)):
		ifSum += alpha.iloc[j,0] * (psii*psi.iloc[j])
	return ifSum

def gaussianFunction(x,sigma):
	'''computes gaussian function'''
	x = x.values.flatten()
	return math.exp((-np.linalg.norm(x)**2.0)/(2.0*sigma))


parser = argparse.ArgumentParser()
parser.add_argument("--kernel")
parser.add_argument("--dataset")
parser.add_argument("--sigma")
parser.add_argument("--lam")
parser.add_argument("--T")
parser.add_argument("--output")
args = parser.parse_args()


#Default parameters if not included in CL
if(args.lam == None):
	lam = 0.1
else:
	lam = float(args.lam)
if(args.T == None):
	T = 100
else:
	T = int(args.T)
if(args.sigma == None):
	sigma = 100.0
else:
	sigma = float(args.sigma)
if (args.output == None):
	output = "output.txt"
else:
	output = args.output
if (args.kernel=="linear"):
	trainfcn = svmTrain
	testfcn = svmTest
else:
	trainfcn = rbfTrain
	testfcn = rbfTest
f = open(output, "a")
if(args.dataset == "mnist"):
	trainImageFile = input("train image file path: ")
	trainLabelFile = input("train label file path: ")
	testImageFile = input("test image file path: ")
	testLabelFile = input("test label file path: ")
	trainImgs, trainLbls = readIdxFile(trainImageFile,trainLabelFile)
	testImgs, testLbls = readIdxFile(testImageFile,testLabelFile)
	train = trainImgs.reshape(trainImgs.shape[0],-1)
	test = testImgs.reshape(testImgs.shape[0],-1)
	train = pd.DataFrame(train)
	test = pd.DataFrame(test)
	train["y"] = trainLbls
	test["y"] = testLbls
	trainErrors = 0
	testErrors = 0
	for i in range(10):
		print("Digit is "+str(i))
		trainCopy = train.copy()
		testCopy = test.copy()
		trainCopy.iloc[:,-1] = trainCopy.iloc[:,-1].map(lambda y: 1 if y == i  else -1)
		testCopy.iloc[:,-1] = testCopy.iloc[:,-1].map(lambda y: 1 if y == i else -1)
		weightVector,err = trainfcn(trainCopy,float(lam),T,sigma)
		trainErrors += err
		testErrors += testfcn(testCopy,weightVector)
		with pd.option_context('display.max_rows', None, 'display.max_columns', None):
			f.write(str(weightVector))
	print("Average training error was " + str(trainErrors/10.0)+"%")
	print("Average testing error was " + str(testErrors/10.0)+"%")
		

elif(args.dataset == "bcd"):
	trainFile = input("train file path: ")
	train = pd.read_csv(trainFile)
	#Represent success and failure as -1 and 1
	train.iloc[:,-1] = train.iloc[:,-1].map(lambda y: -1 if y == 0 else 1)
	weightVector,_ = trainfcn(train,float(lam),T,sigma)
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print("Printing weight vector to: " + output)
		f.write(str(weightVector))

f.close()
