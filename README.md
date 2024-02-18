# Execution samples:

from sklearn.datasets import load_iris from sklearn.model_selection import train_test_split

iris = load_iris() X = iris.data y = iris.target

model = piecewise_regressor(regression_type="multi") model.fit(X, y) y_pred = model.predict(X)

# ###########################

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer() X, y = data.data, data.target

model = piecewise_regressor(regression_type="logistic") model.fit(X, y) y_pred = model.predict(X)

# ###########################

from sklearn.datasets import load_diabetes

data = load_diabetes() X, y = data.data, data.target

model = piecewise_regressor(regression_type="linear") model.fit(X, y) y_pred = model.predict(X)
