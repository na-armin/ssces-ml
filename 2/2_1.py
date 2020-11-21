#Load CSV Using Python Standarad Library

import csv
import numpy
filename='pima-indians-diabetes.csv'
raw_data=open(filename,'rt')
reader=csv.reader(raw_data,delimiter=',',quoting=csv.QUOTE_NONE)
x=list(reader)

data=numpy.array(x).astype('float')
print(data)
print (data.shape)

#Load CSV Using Numpy
from numpy import loadtxt
filename='pima-indians-diabetes.csv'
raw_data=open(filename,'rt')
data=loadtxt(raw_data,delimiter=',')
print(data)
print (data.shape)

#Load CSV Using Pandas
from pandas import read_csv
filename='pima-indians-diabetes.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=read_csv(filename,names=names)
print(data)
print (data.shape)