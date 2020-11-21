#Rescale data (between 0 and 1)
from  pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array=dataframe.values
#seprate array into input and output component
X=array[:,0:8]
Y=array[:,8]
scaler=MinMaxScaler(feature_range=(0,1)) # Transformed data into 0,1
rescaledX=scaler.fit_transform(X)
#summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
#seprate array into input and output component
X=array[:,0:8]
Y=array[:,8]
scaler=StandardScaler().fit(X)
rescaledX=scaler.fit_transform(X)
#summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#Normalize data (length 1)
from sklearn.preprocessing import Normalizer
#seprate array into input and output component
X=array[:,0:8]
Y=array[:,8]
scaler=Normalizer().fit(X)
rescaledX=scaler.fit_transform(X)
#summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])

#Binarization data (length 1)
from sklearn.preprocessing import Binarizer
#seprate array into input and output component
X=array[:,0:8]
Y=array[:,8]
scaler=Binarizer(threshold=0.0).fit(X)
rescaledX=scaler.fit_transform(X)
#summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:5,:])