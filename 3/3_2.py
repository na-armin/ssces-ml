#انتخاب یک متغییره
# Feature selection with Univariate Statistical Tests (Chi-squared for classification)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

# load data
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] #نام ستون ها
dataframe = read_csv(filename, names=names)
array = dataframe.values
# seprate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]
# feature selection
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print (fit.scores_)
features = fit.transform(X)
# summarize selected features
print (features[0:5, :])

#حإف ویژگی بازگشتی
# Feature extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print ("Num Features:")
print (fit.n_features_)
print ("Selected Features:")
print (fit.support_)
print ("Num Features: ")
print(fit.ranking_)

#تحلیل مولف اصلی
# feature Extraction with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize componenta
print ("Explanied Variance: %s" % fit.explained_variance_ratio_)
print (fit.components_)