import numpy as np
import pandas as pd

# Correct URL for the raw CSV file
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
dataframe = pd.read_csv(url)

# View the first two rows
# print(dataframe.head(2))

# creat a dataframe
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Location': ['New York', 'Paris', 'Berlin', 'London'],
    'Age': [24, 13, 53, 33]
}

dataframe = pd.DataFrame(data)

# print(dataframe)

dataframe['eyes color'] = ['brown', 'blue', 'green', 'brown']

print(dataframe)


dataframe = pd.read_csv(url)
print(dataframe.shape) # dimensions of the dataframe
print(dataframe.describe()) # summary statistics
print(dataframe.info()) # summary of the data

print(dataframe.iloc[0]) # first row of the dataframe
print(dataframe.iloc[:4]) # first four rows of the dataframe

print(dataframe.sort_values('Age').head(2)) # sort the dataframe by age and print the first two rows
print(dataframe['Cabin'].replace('NaN' , '0').head(2))# replace NaN values with 0

dataframe = pd.read_csv(url)

# Find min max count sum mean median mode std var of the age column
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())
print('Median:', dataframe['Age'].median())
print('Mode:', dataframe['Age'].mode())

# unique values in the sex column

print(dataframe['Sex'].unique())

# number of unique values

print(dataframe['Sex'].value_counts())
print(dataframe['Age'].value_counts())

print(dataframe[dataframe['Cabin'].isnull()].head())

dataframe['Cabin'] = dataframe['Cabin'].replace(np.nan, 0)
print(dataframe['Cabin'].head())