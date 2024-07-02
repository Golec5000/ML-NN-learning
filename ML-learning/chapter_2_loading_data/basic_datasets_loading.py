from sklearn import datasets

# Load the numpy array of the digits dataset
digits = datasets.load_digits()

# Create feature matrix
features = digits.data

# Create target vector
target = digits.target

# View first observation
print(features[0])

# print(digits.DESCR)