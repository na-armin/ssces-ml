from pandas import read_csv

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names,)

# view first 20 rows
peek = data.head(20)
print(peek)

# Dimensions of your data
shape = data.shape
print(shape)

# data type for each attribute
print('data type for each attribute:')
types = data.dtypes
print(types)

# Statistical Summary
print('Statistical Summary:')
from pandas import set_option

set_option('precision', 3)
description = data.describe()
print(description)

# Class Distribution
print('Class Distribution:')
class_count = data.groupby('class').size()
print(class_count)

# Pairwise pearson correlations
correlation = data.corr(method='pearson')
print(correlation)

# Skew for each attribute
skew = data.skew()
print(skew)