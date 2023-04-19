import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Python version
# print('Python: {}'.format(sys.version))
# scipy
# print('scipy: {}'.format(scipy.__version__))
# numpy
# print('numpy: {}'.format(numpy.__version__))
# matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
# print('pandas: {}'.format(pandas.__version__))
# scikit-learn
# print('sklearn: {}'.format(sklearn.__version__))
# dataframe

myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

# Load CSV using Pandas from URL

url = "https://goo.gl/bDdBiA"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(url, names=names)
print(data.shape)

array = data.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8] 

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
# print(rescaledX[0:5,:])

# Scatter Plot Matrix
scatter_matrix(data)
plt.show()



kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
