import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer, PolynomialFeatures, FunctionTransformer, StandardScaler, Binarizer
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer, SimpleImputer
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

# detect outliers

data, _ = make_blobs(n_samples=10,
                     n_features=2,
                     centers=1,
                     random_state=1)

# replace the first data point with an outlier
data[0, 0] = 10000
data[0, 1] = 10000

outlier_detector = EllipticEnvelope(contamination=.1)  # create EllipticEnvelope object with 10% contamination

# fit detector
outlier_detector.fit(data)  # fit the data

# predict outliers
print(outlier_detector.predict(data))  # predict the data

# outlier handling

hauses = pd.DataFrame()
hauses['Price'] = [534433, 392333, 293222, 4322032]
hauses['Bathrooms'] = [2, 3.5, 2, 116]
hauses['Square_Feet'] = [1500, 2500, 1500, 48000]

# filter observations
print(hauses[hauses['Bathrooms'] < 20])  # filter the data

# filter and mark the outliers

hauses['Outlier'] = np.where(hauses['Bathrooms'] < 20, 0, 1)  # mark the outliers

print(hauses.head())

# calc log of the feature
hauses['Log_of_Square_Feet'] = [np.log(x) for x in hauses['Square_Feet']]  # calculate the log of the Square_Feet column

print(hauses.head())

# discretize feature

age = np.array([
    [6],
    [12],
    [20],
    [36],
    [65]
])

# bin feature
binarizer = Binarizer(threshold=18)  # create Binarizer object with threshold 18

print(binarizer.fit_transform(
    age))  # fit and transform the data -> equlavent to binarizer.fit(data) and binarizer.transform(data)

# group observations using k-means clustering

data, _ = make_blobs(n_samples=50,
                     n_features=2,
                     centers=3,
                     random_state=1)

dataframe = pd.DataFrame(data, columns=['feature_1', 'feature_2'])

# create k-means clustering
clusterer = KMeans(3, random_state=0)  # create KMeans object with 3 clusters

# fit the model
clusterer.fit(data)  # fit the data

# predict the cluster
dataframe['group'] = clusterer.predict(data)  # predict the data

print(dataframe.head())

# remove observations with missing values

data = np.array([
    [1.1, 11.1],
    [2.2, 22.2],
    [3.3, 33.3],
    [4.4, 44.4],
    [np.nan, 55]
])

# remove observations with missing values

print(data[~np.isnan(data).any(axis=1)])  # remove the missing values

# remove observations with missing values
dataframe = pd.DataFrame(data, columns=['feature_1', 'feature_2'])

print(dataframe.dropna())  # remove the missing values

# impute missing values

data, _ = make_blobs(n_samples=1000,
                     n_features=2,
                     random_state=1)

scaler = StandardScaler()  # create StandardScaler object

data_standardized = scaler.fit_transform(data)  # fit and transform the data

# introduce a nan

true_value = data_standardized[0, 0]  # store the true value

data_standardized[0, 0] = np.nan  # set the first value to missing

# predict the missing values

imputer = KNNImputer(n_neighbors=5)  # create KNNImputer object

data_standardized_imputed = imputer.fit_transform(data_standardized)  # fit and transform the data

print('True Value:', true_value)
print('Imputed Value:', data_standardized_imputed[0, 0])

data, _ = make_blobs(n_samples=1000,
                     n_features=2,
                     random_state=1)

scaler = StandardScaler()  # create StandardScaler object

data_standardized = scaler.fit_transform(data)  # fit and transform the data

# introduce a nan

true_value = data_standardized[0, 0]  # store the true value
data_standardized[0, 0] = np.nan  # set the first value to missing

mean_imputer = SimpleImputer(strategy='mean')  # create SimpleImputer object

data_standardized_imputed = mean_imputer.fit_transform(data_standardized)  # fit and transform the data

print('True Value:', true_value)
print('Imputed Value:', data_standardized_imputed[0, 0])
