from matplotlib import pyplot
from  pandas import read_csv
import numpy

filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Univariate Histograms
data.hist()
pyplot.show()

# Univariate Density Plots
data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
pyplot.show()

# Box and Whisker Plots
data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False)
pyplot.show()

# Corelation Matrix Plot
correlation=data.corr()
#plot correlation matrix
fig=pyplot.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks=numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()

# Corelation Matrix Plot (generic)
correlation=data.corr()
#plot correlation matrix
fig=pyplot.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(correlation,vmin=-1,vmax=1)
fig.colorbar(cax)
pyplot.show()

#Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data)
pyplot.show()