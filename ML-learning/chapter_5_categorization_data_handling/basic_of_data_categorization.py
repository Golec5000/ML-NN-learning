import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Create a list of example data
data = np.array([
    ['Teksas'],
    ['California'],
    ['Texas'],
    ['Washington'],
    ['Texas'],
    ['California'],
    ['Washington'],
    ['California'],
    ['Texas'],
    ['Washington']
])

one_hot = LabelBinarizer()

transform_data = one_hot.fit_transform(data)  # Transform data to one-hot encoding

print(transform_data)
print(one_hot.classes_)  # Show the classes of the one-hot encoding
print(one_hot.inverse_transform(transform_data))  # Inverse transform the one-hot encoding

# Create multi-label data
multi_data = np.array([
    ('Texas', 'Florida'),
    ('California', 'Washington'),
    ('Texas', 'Washington'),
    ('Dalwera', 'California'),
    ('Texas', 'Washington')
])

multi_one_hot = MultiLabelBinarizer()

transform_multi_data = multi_one_hot.fit_transform(multi_data)  # Transform data to multi-label one-hot encoding

print(transform_multi_data)
print(multi_one_hot.classes_)  # Show the classes of the multi-label one-hot encoding

import pandas as pd

dataframe = pd.DataFrame({'Score': ['Low', 'Low', 'Medium', 'Medium', 'High']})

scale_mapper = {
    'Low': 1,
    'Medium': 2,
    'High': 3
}

dataframe['Score'] = dataframe['Score'].replace(scale_mapper)

print(dataframe.head())

# coding dictionary of the scale

from sklearn.feature_extraction import DictVectorizer

data_dict = [
    {'Red': 2, 'Blue': 4, 'Green': 3, 'Yellow': 1},
    {'Red': 4, 'Blue': 3, 'Green': 2},
    {'Red': 1, 'Blue': 2, 'Yellow': 2},
    {'Red': 2, 'Yellow': 2},
    {'Red': 4, 'Blue': 3},
    {'Red': 1, 'Yellow': 2},
    {'Red': 2, 'Yellow': 2}
]

dict_vectorizer = DictVectorizer(sparse=False)

features = dict_vectorizer.fit_transform(data_dict)

print(features)

doc_1_count = {'Red': 2, 'Blue': 4, 'Green': 3, 'Yellow': 1}
doc_2_count = {'Red': 4, 'Blue': 3, 'Green': 2}
doc_3_count = {'Red': 1, 'Blue': 2, 'Yellow': 2}
doc_4_count = {'Red': 2, 'Yellow': 2}

doc_counts = [doc_1_count, doc_2_count, doc_3_count, doc_4_count]

features = dict_vectorizer.transform(doc_counts)

print(features)

# insert lost classes value

from sklearn.neighbors import KNeighborsClassifier

data = np.array([
    [0, 2.10, 1.45],
    [1, 1.18, 1.33],
    [0, 1.22, 1.27],
    [1, 1.35, 1.13],
    [1, -1.75, -1.23]
])

data_with_nan = np.array([
    [np.nan, 1.87, 1.31],
    [np.nan, 1.31, 1.11]
])

clf = KNeighborsClassifier(3, weights='distance')

trained_model = clf.fit(data[:, 1:], data[:, 0])  # Train the model with the data without NaN

imputed_values = trained_model.predict(data_with_nan[:, 1:])  # Predict the class of the data with NaN

data_with_imputed = np.hstack((imputed_values.reshape(-1, 1), data_with_nan[:, 1:]))

data_with_imputed = np.vstack((data_with_imputed, data))

print(data_with_imputed)



