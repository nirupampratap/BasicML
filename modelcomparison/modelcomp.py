## Data and Visual Analytics - Homework 4
## Georgia Institute of Technology
## Applying ML algorithms to detect seizure

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize

######################################### Reading and Splitting the Data ###############################################
# XXX
# TODO: Read in all the data. Replace the 'xxx' with the path to the data set.
# XXX
data = pd.read_csv('seizure_dataset.csv')

# Separate out the x_data and y_data.
x_data = data.loc[:, data.columns != "y"]
y_data = data.loc[:, "y"]

# The random state to use while splitting the data.
random_state = 100

# XXX
# TODO: Split 70% of the data into training and 30% into test sets. Call them x_train, x_test, y_train and y_test.
# Use the train_test_split method in sklearn with the paramater 'shuffle' set to true and the 'random_state' set to 100.
# XXX
#train, test = train_test_split(data, test_size=0.3, random_state=100, shuffle=True)
#x_train = train.loc[:, train.columns != "y"]
#x_test = test.loc[:, test.columns != "y"]
#y_train = train.loc[:, "y"]
#y_test = test.loc[:, "y"]
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=100, shuffle=True)

# ############################################### Linear Regression ###################################################
# XXX
# TODO: Create a LinearRegression classifier and train it.
# XXX
lmod = LinearRegression()
lmod.fit(x_train, y_train)

y_trainfit = lmod.predict(x_train)
#y_trainpred = map(lambda x: 0 if x <=0.5 else 1, y_trainfit)
y_trainpred = y_trainfit.round()

y_testfit = lmod.predict(x_test)
#y_testpred = map(lambda x: 0 if x <=0.5 else 1, y_testfit)
y_testpred = y_testfit.round()

# XXX
# TODO: Test its accuracy (on the training set) using the accuracy_score method.
# TODO: Test its accuracy (on the testing set) using the accuracy_score method.
# Note: Use y_predict.round() to get 1 or 0 as the output.
# XXX
print("Linear Regression - Train Set: ", accuracy_score(y_train, y_trainpred))
print("Linear Regression - Test Set: ", accuracy_score(y_test, y_testpred))

# ############################################### Multi Layer Perceptron #################################################
# XXX
# TODO: Create an MLPClassifier and train it.
# XXX
mlpmod = MLPClassifier(random_state = 100)
mlpmod.fit(x_train, y_train)

y_mtrainfit = mlpmod.predict(x_train)
y_mtestfit = mlpmod.predict(x_test)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print("MLP Classifier - Train Set: ", accuracy_score(y_train, y_mtrainfit))
print("MLP Classifier - Test Set: ", accuracy_score(y_test, y_mtestfit))


# ############################################### Random Forest Classifier ##############################################
# XXX
# TODO: Create a RandomForestClassifier and train it.
# XXX
rfmod = RandomForestClassifier(random_state = 100)
rfmod.fit(x_train, y_train)

y_rtrainfit = rfmod.predict(x_train)
y_rtestfit = rfmod.predict(x_test)
# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print("Random Forest Classifier - Train set: ",accuracy_score(y_train, y_rtrainfit))
print("Random Forest Classifier - Test set: ", accuracy_score(y_test, y_rtestfit))

# XXX
# TODO: Tune the hyper-parameters 'n_estimators' and 'max_depth'.
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX   

#params = [{'n_estimators': [10, 15, 20, 25, 30, 50, 100, 200], 'max_depth': [2,5,10,25,50,100] }]
params = [{'n_estimators': [10, 20, 50], 'max_depth': [2,10,20] }]
clf = GridSearchCV(rfmod, params, cv=10)
clf.fit(x_train, y_train)
print("Random forest classifier - Grid Search - Best params: ", clf.best_params_)
print("Random forest classifier - Grid search - Best Score: ", clf.best_score_)

y_rgstestfit = clf.predict(x_test)
print("Random forest Classifier - Grid Search : ",accuracy_score(y_test, y_rgstestfit))

# ############################################ Support Vector Machine ###################################################
# XXX
# TODO: Pre-process the data to standardize or normalize it, otherwise the grid search will take much longer
# TODO: Create a SVC classifier and train it.
# XXX
scaler = StandardScaler()
scaler.fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)

svcmod = SVC(random_state = 100)
svcmod.fit(x_train_sc, y_train)

y_svctrainfit = svcmod.predict(x_train_sc)
y_svctestfit = svcmod.predict(x_test_sc)

# XXX
# TODO: Test its accuracy on the training set using the accuracy_score method.
# TODO: Test its accuracy on the test set using the accuracy_score method.
# XXX
print("SVC - Train set: ", accuracy_score(y_train, y_svctrainfit))
print("SVC - Test set: ", accuracy_score(y_test, y_svctestfit))

# XXX
# TODO: Tune the hyper-parameters 'C' and 'kernel' (use rbf and linear).
#       Print the best params, using .best_params_, and print the best score, using .best_score_.
# XXX

#x_train_sc = normalize(x_train,axis=0,norm='l2')

params = [{'C': [0.1,1,10], 'kernel': ['rbf', 'linear'] }]
clf = GridSearchCV(svcmod, params, cv=10)
clf.fit(x_train_sc, y_train)
print("SVC - Grid search - Best params: ", clf.best_params_)
print("SVC - Grid Search - Best score: ", clf.best_score_)

y_svcgtestfit = clf.predict(scaler.transform(x_test))
print("SVC - Grid Search - Test set : ",accuracy_score(y_test, y_svcgtestfit))

#print("Grid Search - Grid", clf.cv_results_)
