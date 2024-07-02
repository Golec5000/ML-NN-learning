import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

# scale feature to a range

data = np.array([
    [-500.5],
    [-100.1],
    [0],
    [100.1],
    [900.9]
])

# create scaler
min_max_scaler = preprocessing.MinMaxScaler(
    feature_range=(0, 1))  # create MinMaxScaler object with feature range 0 to 1

# scale feature
data_scaled = min_max_scaler.fit_transform(
    data)  # fit and transform the data -> equlavent to min_max_scaler.fit(data) and min_max_scaler.transform(data)

print(data_scaled)

# standardize feature

data = np.array([
    [-1000.1],
    [-200.2],
    [500.5],
    [600.6],
    [9000.9]
])

# create scaler
scaler = preprocessing.StandardScaler()  # create StandardScaler object

# transform the feature
data_standardized = scaler.fit_transform(
    data)  # fit and transform the data -> equlavent to scaler.fit(data) and scaler.transform(data)

print(data_standardized)

print('Mean:', round(data_standardized.mean()))
print('Standard deviation:', data_standardized.std())

# normalize feature

data = np.array([
    [1.1, 2.2],
    [2.2, 3.3],
    [3.3, 4.4],
    [4.4, 5.5],
    [5.5, 6.6]
])

# transform feature
data_l2_norm = Normalizer(norm='l2').transform(data)  # normalize the data using l2 norm

print(data_l2_norm)

data_l1_norm = Normalizer(norm='l1').transform(data)  # normalize the data using l1 norm

print(data_l1_norm)

# generate polynomial and interaction features

data = np.array([
    [2, 3],
    [2, 3],
    [2, 3]
])

# create PolynomialFeatures object
poly = PolynomialFeatures(degree=2,
                          include_bias=False)  # create PolynomialFeatures object with degree 2 and exclude bias

# transform the feature

print(poly.fit_transform(data))  # fit and transform the data -> equlavent to poly.fit(data) and poly.transform(data)

# transform the feature
data = np.array([
    [2, 3],
    [2, 3],
    [2, 3]
])


# create a function
def add_ten(x: int) -> int:
    return x + 10


# create transformer
ten_transformer = FunctionTransformer(add_ten)  # create FunctionTransformer object

# transform the feature
print(ten_transformer.transform(data))  # transform the data using the custom function

df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])

# apply a function to a pandas column
# apply the function to the dataframe
print(df.apply(add_ten).head())
