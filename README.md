# svm
Implementing the soft-SVM learning algorithm using stochastic gradient
descent.

# Usage:
```bash
python svm.py --kernel [linear|rbf] \
--dataset [mnist|bcd] \
--cost [Cvalue] \
--train /path/to/training/data \
--test /path/to/test/data --output /path/to/weightvector
```

Kernel: Linear (no kernel function) or RBF (Radial Basis Function)

Dataset: mnist (handwritten digits) or bcd (breast cancer dataset)

Cost: cost hyperparameter C

# Output:
The *d*-dimensional weight vector that SVM learns, available as a plain text file with *d* lines, where the *i<sup>th</sup>* line contains the value of the *i<sup>th</sup>* coordinate of **w**.
