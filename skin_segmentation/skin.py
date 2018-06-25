# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
import seaborn as sb
import os

cwd = os.getcwd()
path = cwd+"\\skin_segmentation\Skin_NonSkin.data"
names = ['B', 'G', 'R', 'Class']
dataset = pandas.read_csv(path, names=names)


""" 
https://archive.ics.uci.edu/ml/datasets/skin+segmentation

This dataset is of the dimension 245057 * 4 where first three columns are B,G,R (x1,x2, and x3 features) 
values and fourth column is of the class labels (decision variable y).

"""

print(dataset.shape)
print(dataset.columns)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('Class').count())
print(dataset.sample(5))
print(dataset.isna().any())
print(dataset.dtypes)

# Shuffle DataFrame rows, since it is ordened by class
dataset = dataset.sample(frac=1).reset_index(drop=True) 

new_size_index = int(len(dataset.values)/2)
df1 = dataset.iloc[:new_size_index]
df2 = dataset.iloc[new_size_index: ]

# df1.hist()
# plt.show()

# df2.hist()
# plt.show()

# scatter_matrix(df1)
# plt.show()

# scatter_matrix(df2)
# plt.show()

size = len(dataset.columns) - 1

array = df1.values
X = array[:,:size]
Y = array[:,size]

validation_size = 0.30
seed = 18
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
test_size=validation_size, random_state=seed)

seed = 18
scoring = 'accuracy'

# algorithms = []
# algorithms.append(('LR', LogisticRegression()))
# algorithms.append(('LDA', LinearDiscriminantAnalysis()))
# algorithms.append(('KNN', KNeighborsClassifier()))
# algorithms.append(('CART', DecisionTreeClassifier()))
# algorithms.append(('NB', GaussianNB()))
# # algorithms.append(('SVM', SVC())) to much time to execute 
# algorithms.append(('RFC', RandomForestClassifier()))
# algorithms.append(('GBC', GradientBoostingClassifier()))
# algorithms.append(('ETC', ExtraTreesClassifier()))
# algorithms.append(('BC', BaggingClassifier()))


# results = []
# names = []
# for name, algorithm in algorithms:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)


alg = KNeighborsClassifier()
alg.fit(X_train, Y_train)
predictions = alg.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


data = df2.values[:,:size]
test = df2.values[:,size]
prediction = alg.predict(data)
print(prediction)
print(accuracy_score(test, prediction))
print(confusion_matrix(test, prediction))
print(classification_report(test, prediction))