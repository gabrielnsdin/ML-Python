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
path = cwd+"\\breast_cancer_coimbra\dataR2.csv"
names = ['Age', 'BMI kg/m2', 'Glucose mg/dL', 'Insulin µU/mL', 'HOMA', 'Leptin ng/mL', 'Adiponectin µg/mL', 'Resistin ng/mL', 'MCP-1 pg/dL', 'Health']
dataset = pandas.read_csv(path, names=names)


""" 
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

Data Set Information:

There are 10 predictors, all quantitative, and a binary dependent variable, indicating the presence or absence of breast cancer. 
The predictors are anthropometric data and parameters which can be gathered in routine blood analysis. 
Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer.


Attribute Information:

Quantitative Attributes: 
Age (years) 
BMI (kg/m2) 
Glucose (mg/dL) 
Insulin (µU/mL) 
HOMA 
Leptin (ng/mL) 
Adiponectin (µg/mL) 
Resistin (ng/mL) 
MCP-1(pg/dL) 

Labels: 
1=Healthy controls 
2=Patients

 """

print(dataset.shape)
print(dataset.columns)
print(dataset.head(10))
print(dataset.describe())
print(dataset.groupby('Health').count())
print(dataset.sample(5))
print(dataset.isna().any())

# plt.scatter(dataset['Insulin µU/mL'], dataset['Resistin ng/mL'])
# plt.legend(loc='upper left')
# plt.show()

# dataset.hist()
# plt.show()

# scatter_matrix(dataset)
# plt.show()

dataset.drop(['MCP-1 pg/dL'], axis = 1, inplace = True)
dataset.drop(['Insulin µU/mL'], axis = 1, inplace = True)
dataset.drop(['HOMA'], axis = 1, inplace = True)
dataset.drop(['Leptin ng/mL'], axis = 1, inplace = True)
dataset.drop(['Adiponectin µg/mL'], axis = 1, inplace = True)

size = len(dataset.columns) - 1

array = dataset.values
X = array[:,:size]
Y = array[:,size]

validation_size = 0.30
seed = 18
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


gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
predictions = gbc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))