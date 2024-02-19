# Execution samples:

from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

data = load_iris()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = piecewise_regressor(regression_type="multi")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))


# ###########################
#############################

from sklearn.datasets import load_wine

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

data = load_wine()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = piecewise_regressor(regression_type="multi")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))


# ###########################
#############################

from sklearn.datasets import load_breast_cancer

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    
data = load_breast_cancer()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = piecewise_regressor(regression_type="logistic")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))


# ###########################
#############################

from sklearn.datasets import load_diabetes

from sklearn.metrics import accuracy_score, classification_report

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

import warnings

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

data = load_diabetes()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

model = piecewise_regressor(regression_type="linear")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))
