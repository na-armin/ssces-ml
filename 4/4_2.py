#معیارهای دسته بندی

# دقت دسته بندی
# Cross Validation Classification Accuaracy
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = '/content/drive/My Drive/Colab Notebooks/pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]
num_folds =10
seed=7
Kfold=KFold(n_splits=num_folds,random_state=seed ,shuffle=True)
model = LogisticRegression(max_iter=1000)
scorring='accuracy'
result=cross_val_score(model, X ,Y ,cv=Kfold, scoring=scorring)
print("Accuracy: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))

#اتلاف لگاریتمی
# Cross Validation Classification LogLoss

scorring='neg_log_loss'
result=cross_val_score(model, X ,Y ,cv=Kfold, scoring=scorring)
print("Logloss: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))

#مساحت زیر منحنی ROC
# Cross Validation Classification ROC AUC

scorring='roc_auc'
result=cross_val_score(model, X ,Y ,cv=Kfold, scoring=scorring)
print("AUC: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))

#ماتریس درهم ریختگی
# Cross Validation Classification Confusion Matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test,predicted)
print(matrix)

# گزارش دسته بندی
# Cross Validation Classification Report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
predicted = model.predict(X_test)
report = classification_report(Y_test,predicted)
print(report)

# معیارهای رگرسیون

#میانگین قدر مطلق خطا
# Cross Validation Regression MAE
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

filename = '/content/sample_data/california_housing_train.csv'
dataframe = read_csv(filename,header=0)
print(dataframe)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]

Kfold=KFold(n_splits=10,random_state=7,shuffle=True)
model = LinearRegression()
scorring='neg_mean_absolute_error'
result=cross_val_score(model, X ,Y ,cv=Kfold, scoring=scorring)
print("MAE: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))


# #میانگین مربعات خطا
# # Cross Validation Regression MSE

scorring='neg_mean_squared_error'
result=cross_val_score(model, X ,Y ,cv=Kfold, scoring=scorring)
print("MSE: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))


# #معیار R^2
# # Cross Validation Regression R^2
scorring='r2'
result=cross_val_score(model, X ,Y ,cv=Kfold, scoring=scorring)
print("R^2: %.3f (%.3f)" %(result.mean()*100.0 ,result.std()*100.0))