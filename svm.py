'''
Cole Conte CSE 512 HW 2
svm.py
'''

import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--kernel")
parser.add_argument("--dataset")
parser.add_argument("--train")
parser.add_argument("--test")
args = parser.parse_args()

train = pd.read_csv(args.train)
test = pd.read_csv(args.test)