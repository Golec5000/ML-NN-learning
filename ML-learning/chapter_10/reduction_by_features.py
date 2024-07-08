import numpy as np
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# Load data

iris = datasets.load_iris()

# Create features and target

features = iris.data
target = iris.target

# Create thresholder

thresholder = VarianceThreshold(threshold=.5)

# Create high variance feature matrix

features_high_variance = thresholder.fit_transform(features)

# View high variance feature matrix

print(features_high_variance[0:3])

print(thresholder.fit(features).variances_)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

selector = VarianceThreshold()

print(selector.fit(features_std).variances_)

features = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0]
])

thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))  # 75% of observations contain a 1 or 0

print(thresholder.fit_transform(features))

# High variance feature selection

import pandas as pd

features = np.array(
    [
        [1, 1, 1],
        [2, 2, 0],
        [3, 3, 1],
        [4, 4, 0],
        [5, 5, 1],
        [6, 6, 0],
        [7, 7, 1],
        [8, 7, 0],
        [9, 7, 1]
    ]
)

dataframe = pd.DataFrame(features)

print(dataframe.head())

corr_matrix = dataframe.corr().abs()

print(corr_matrix)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(dataframe.drop(dataframe.columns[to_drop], axis=1).head(3))

print(dataframe.corr())

print(upper)

from sklearn.feature_selection import SelectKBest, chi2, f_classif

iris = datasets.load_iris()

features = iris.data
target = iris.target

features = features.astype(int)

chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
