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

from sklearn.datasets import make_regression

# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples=100,
                                                 n_features=3,
                                                 n_informative=3,
                                                 n_targets=1,
                                                 noise=0.0,
                                                 coef=True,
                                                 random_state=1)

# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

from sklearn.datasets import make_classification

# Generate features matrix and target vector

features, target = make_classification(n_samples=100,
                                       n_features=3,
                                       n_informative=3,
                                       n_redundant=0,
                                       n_classes=2,
                                       weights=[.25, .75],
                                       random_state=1)

# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

from sklearn.datasets import make_blobs

# Generate feature matrix and target vector

features, target = make_blobs(n_samples=100,
                              n_features=2,
                              centers=3,
                              cluster_std=0.5,
                              shuffle=True,
                              random_state=1)

# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])

import matplotlib.pyplot as plt

# View scatterplot

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()
