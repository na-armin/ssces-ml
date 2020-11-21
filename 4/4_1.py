# Evaluate using a train and a test set
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

filename = '/content/drive/My Drive/Colab Notebooks/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
seed = 7
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size ,random_state=seed)
model =LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
result = model.score(X_test,Y_test)
print("Accuracy: %.3f" %(result*100.0))

# اعتبار سنجی متقابل K تایی
# Evaluate using Cross Validation
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]
num_folds =10
seed=7
Kfold=KFold(n_splits=num_folds,random_state=seed ,shuffle=True)
model = LogisticRegression(max_iter=1000 )
result=cross_val_score(model, X ,Y ,cv=Kfold)
print("Accuracy: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))

# اعتبار سنجی متقابل Leave One Out
# Evaluate using Leave One Out Cross Validation
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]
loocv=LeaveOneOut()
model=LogisticRegression(max_iter=1000)
result=cross_val_score(model,X,Y,cv=loocv)
print("Accuracy: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))

#تقسیم بندی آزمایشی_آموزشی تصادفی تکراری
# Evaluate using Leave One Out Cross Validation
from pandas import read_csv
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
# filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]
n_splits=10
test_size=0.33
kfold=ShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=7)
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model , X,Y,cv=Kfold)
print("Accuracy: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))