# svm
Implementing the soft-SVM learning algorithm using stochastic gradient
descent.

# Usage:
```bash
python svm.py --kernel [linear|rbf] \
--dataset [mnist|bcd] \
--cost [Cvalue] \
--output /path/to/weightvector
```

Kernel: Linear (no kernel function) or RBF (Radial Basis Function)

Dataset: mnist (handwritten digits) or bcd (breast cancer dataset)

Cost: cost hyperparameter C

Based on your selection of dataset, you'll be prompted by the command line interface to enter the path for your training and test files.

# Output:
The *d*-dimensional weight vector that SVM learns, available as a plain text file with *d* lines, where the *i<sup>th</sup>* line contains the value of the *i<sup>th</sup>* coordinate of **w**.
