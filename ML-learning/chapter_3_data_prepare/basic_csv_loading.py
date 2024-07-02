import numpy as np
import pandas as pd

url = 'titanic.csv'
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
print(dataframe.shape)  # dimensions of the dataframe
print(dataframe.describe())  # summary statistics
print(dataframe.info())  # summary of the data

print(dataframe.iloc[0])  # first row of the dataframe
print(dataframe.iloc[:4])  # first four rows of the dataframe

print(dataframe.sort_values('Age').head(2))  # sort the dataframe by age and print the first two rows
print(dataframe['Cabin'].replace('NaN', '0').head(2))  # replace NaN values with 0

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

print(dataframe.drop_duplicates().head())

# data grouping
dataframe = pd.read_csv(url)

print(dataframe.groupby('Sex').mean(
    numeric_only=True).head())  # group by Sex and calculate the mean of the numeric columns

print(dataframe.groupby('Survived')['Name'].count())  # group by Survived and count the number of names

print(dataframe.groupby(['Sex', 'Survived'])[
          'Age'].mean())  # group multiple columns and calculate the mean of the Age column

# data grouping by time

time_index = pd.date_range('06/06/2017', periods=100000, freq='30s')

dataframe = pd.DataFrame(index=time_index)

dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

print(dataframe.resample('W').sum())  # resample the data by week and sum the values

print(dataframe.head(3))

print(dataframe.resample('2W').sum())  # resample the data by 2 weeks and sum the values

print(dataframe.resample('ME').count())  # resample the data by month and count the values

print(dataframe.resample('ME', label='left').count())  # resample the data by month and count the values

# data aggregation

dataframe = pd.read_csv(url)

print(dataframe.agg({'Age': ['min', 'max', 'mean', 'median'], 'Sex': ['min', 'max']}))  # aggregate the Age column

# dataframes marge

data_a = {
    'id': ['1', '2', '3'],
    'first': ['Alex', 'Amy', 'Allen'],
    'last': ['Anderson', 'Ackerman', 'Ali']
}

dataframe_a = pd.DataFrame(data_a, columns=['id', 'first', 'last'])

data_a_money = {
    'id': ['2', '3', '4'],
    'money': [100, 200, 300]
}

dataframe_b = pd.DataFrame(data_a_money, columns=['id', 'money'])

marge_dataframe = pd.merge(dataframe_a, dataframe_b, on='id',
                           how='outer')  # merge the dataframes on the id column and use an outer join

print(marge_dataframe.head())
