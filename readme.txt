readme.txt

Code Usage:
python svm.py --kernel [linear/rbf] --dataset [bcd/mnist] --lam 0.1 --sigma 1.0 --T 100 --output output.txt

The parameters lambda (lam), sigma, and T are all optional, as is the output file path.

Once you choose a data set, you will be prompted to enter the paths for either 1 (bcd) or 4 (mnist) files. For example, if the breast cancer data set is in your working directory, when prompted, enter:
train file path: "Breast_cancer_data.csv"

These are standard Python inputs, so don't forget to include the quotation marks to make your input a string.

Here are the default file names for easy copy and paste:
"Breast_cancer_data.csv"
"train-images-idx3-ubyte"
"train-labels-idx1-ubyte"
"t10k-images-idx3-ubyte"
"t10k-labels-idx1-ubyte"


