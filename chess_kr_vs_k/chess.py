# Load libraries
import pandas as pd
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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
import os

cwd = os.getcwd()
path = cwd+"\\chess_kr_vs_k\krkopt.data"
names = ['White King column', 'White King row', 'White Rook column', 'White Rook row',
         'Black King column', 'Black King row', 'Moves']
dataset = pd.read_csv(path, names=names)


""" 
https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29

Attribute Information:
   1. White King file (column)
   2. White King rank (row)
   3. White Rook file
   4. White Rook rank
   5. Black King file
   6. Black King rank
   7. optimal depth-of-win for White in 0 to 16 moves, otherwise drawn
	{draw, zero, one, two, ..., sixteen}.
"""

# print(dataset.shape)
# print(dataset.columns)
# print(dataset.head(10))
# print(dataset.describe())
# # print(dataset.groupby('Moves').count())
# print(dataset.sample(5))
# print(dataset.isna().any())
# print(dataset.dtypes)

# Shuffle DataFrame rows, since it is ordened by class
dataset = dataset.sample(frac=1).reset_index(drop=True) 

lb_enc = LabelEncoder()
# df = dataset.apply(lb_enc.fit_transform)
# print(df.head(10))


dataset['White King column'] = lb_enc.fit_transform(dataset['White King column'])
dataset['White Rook column'] = lb_enc.fit_transform(dataset['White Rook column'])
dataset['Black King column'] = lb_enc.fit_transform(dataset['Black King column'])
dataset['Moves'] = lb_enc.fit_transform(dataset['Moves'])

oh_positions = OneHotEncoder()
X = oh_positions.fit_transform(dataset['White King column'].values.reshape(-1,1)).toarray()
Y = oh_positions.fit_transform(dataset['White Rook column'].values.reshape(-1,1)).toarray()
Z = oh_positions.fit_transform(dataset['Black King column'].values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["wkc_"+str(int(i)) for i in range(X.shape[1])])
dataset = pd.concat([dataset, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(Y, columns = ["wrc_"+str(int(i)) for i in range(Y.shape[1])])
dataset = pd.concat([dataset, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(Z, columns = ["bkc_"+str(int(i)) for i in range(Z.shape[1])])
dataset = pd.concat([dataset, dfOneHot], axis=1)



# print(dataset.shape)

dataset.drop('White King column', axis=1, inplace=True)
dataset.drop('White Rook column', axis=1, inplace=True)
dataset.drop('Black King column', axis=1, inplace=True)

dataset = dataset[['White King row', 'White Rook row', 'Black King row', 'wkc_0',
       'wkc_1', 'wkc_2', 'wkc_3', 'wrc_0', 'wrc_1', 'wrc_2', 'wrc_3', 'wrc_4',
       'wrc_5', 'wrc_6', 'wrc_7', 'bkc_0', 'bkc_1', 'bkc_2', 'bkc_3', 'bkc_4',
       'bkc_5', 'bkc_6', 'bkc_7', 'Moves']]

# print(dataset.columns)
# print(dataset.head(10))
# print(dataset.describe())
# print(dataset.groupby('Moves').count())
# print(dataset.sample(5))
# print(dataset.isna().any())
# print(dataset.dtypes)

# # dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
# # plt.show()

# dataset.hist()
# plt.show()

# scatter_matrix(dataset)
# plt.show()

size = len(dataset.columns) - 1

array = dataset.values
X = array[:,:size]
Y = array[:,size]

# print(X)
# print(Y)

validation_size = 0.30
seed = 18
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
test_size=validation_size, random_state=seed)

seed = 18
scoring = 'accuracy'

# algorithms = []
# algorithms.append(('LR', LogisticRegression()))
# # algorithms.append(('LDA', LinearDiscriminantAnalysis()))
# algorithms.append(('KNN', KNeighborsClassifier()))
# algorithms.append(('CART', DecisionTreeClassifier()))
# algorithms.append(('NB', GaussianNB()))
# algorithms.append(('SVM', SVC()))
# algorithms.append(('RFC', RandomForestClassifier()))
# algorithms.append(('GBC', GradientBoostingClassifier()))
# algorithms.append(('ETC', ExtraTreesClassifier()))
# algorithms.append(('BC', BaggingClassifier()))
# algorithms.append(('MLP', MLPClassifier()))


# results = []
# names = []
# for name, algorithm in algorithms:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)


alg = BaggingClassifier()
alg.fit(X_train, Y_train)
predictions = alg.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))