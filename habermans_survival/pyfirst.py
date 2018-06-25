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
import seaborn as sb
import os

cwd = os.getcwd()
path = cwd+"\\habermans_survival\haberman.data"
names = ['age-operation', 'year-operation', 'nodes-number', 'survival']
dataset = pandas.read_csv(path, names=names)


""" 
https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival

1. Age of patient at time of operation (numerical)
2. Patient's year of operation (year - 1900, numerical)
3. Number of positive axillary nodes detected (numerical)
4. Survival status (class attribute)
    1 = the patient survived 5 years or longer
    2 = the patient died within 5 year """

print(dataset.shape)
print(dataset.columns)
# print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('survival').size())
print(dataset.sample(5))

dataset.replace({ 'sex': {'Male':0 , 'Female':1} , 'smoker' : {'No': 0 , 'Yes': 1}} ,inplace=True)


colors = ['blue','green']
target = [1, 2]
target_names = [1, 2]
x_index = 0
y_index = 0

# x = ['age-operation', 'year-operation', 'nodes-number', 'survival']
# plt.scatter(dataset['nodes-number'], dataset['age-operation'])
# plt.legend(loc='upper left')
# plt.show()

# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# dataset.hist()
# plt.show()

# scatter_matrix(dataset)
# plt.show()

array = dataset.values
X = array[:,:3]
Y = array[:,3]
validation_size = 0.30
seed = 23
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
test_size=validation_size, random_state=seed)

seed = 23
scoring = 'accuracy'

# algorithms = []
# algorithms.append(('LR', LogisticRegression()))
# algorithms.append(('LDA', LinearDiscriminantAnalysis()))
# algorithms.append(('KNN', KNeighborsClassifier()))
# algorithms.append(('CART', DecisionTreeClassifier()))
# algorithms.append(('NB', GaussianNB()))
# algorithms.append(('SVM', SVC()))
# algorithms.append(('NN', MLPClassifier()))
# algorithms.append(('RFC', RandomForestClassifier()))

# results = []
# names = []
# for name, algorithm in algorithms:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)


nb = GaussianNB()
nb.fit(X_train, Y_train)
predictions = nb.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

prediction = nb.predict([[38,49,21]])
print(prediction)