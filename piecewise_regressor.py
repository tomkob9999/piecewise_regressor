### Name: piecewise_regressor
# Author: tomio kobayashi
# Version: 1.0.1
# Date: 2024/02/18


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import Counter

class piecewise_regressor:
    
    def __init__(self, regression_type="linear", max_clusters=10):
    # regression_type = {"linear", "multinomial", "logistic"}
        self.gmm = None
        self.regression_type = regression_type
        self.max_clusters = max_clusters
        self.models = {}
        self.direct_val = {}
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):

        n_components = np.arange(1, self.max_clusters)
        models = [GaussianMixture(n, covariance_type='full').fit(X) for n in n_components]
        aic = [model.aic(X) for model in models]
        bic = [model.bic(X) for model in models]

        num_clusters = 1
        for j in range(len(models)-1):
            if j == 0 and (aic[0] < aic[1] or bic[0] < bic[1]):
                break
            if j > 0 and (aic[j] < aic[j+1] or bic[j] < bic[j+1]):
                num_clusters = j+1
                break

        X_clust = {}
        y_clust = {}
        if num_clusters > 1:
            self.gmm = GaussianMixture(n_components=num_clusters)
            self.gmm.fit(X)
            cluster_labels = self.gmm.predict(X)
            for i in range(max(cluster_labels)+1):
                X_clust[i] = np.array([X[j] for j, x in enumerate(cluster_labels) if x == i])
                y_clust[i] = np.array([y[j] for j, x in enumerate(cluster_labels) if x == i])
            for k, v in X_clust.items():
                model = None
                if self.regression_type == "multinomial" or self.regression_type == "multi":
                    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                elif self.regression_type == "logistic":
                    model = LogisticRegression(solver='lbfgs', max_iter=1000)
                else:
                    model = LinearRegression()

                try:
                    model.fit(v, y_clust[k])
                    coefs = model.coef_[0] if self.regression_type == "multinomial" or self.regression_type == "multi"  or self.regression_type == "logistic" else model.coef_
                    if not all([cf == 0 for cf in coefs]):
                        self.models[k] = model
                except Exception as e:
#                     print("cluster", i, "could not be regressed", e)
                    if self.regression_type == "multinomial" or self.regression_type == "multi" or self.regression_type == "logistic":

                        count_dict = Counter(y_clust[k])
                        if len(count_dict) == 1:
                            self.direct_val[k] = y_clust[k][0]
        else:
            model = None
            if self.regression_type == "multinomial" or self.regression_type == "multi":
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            elif self.regression_type == "logistic":
                model = LogisticRegression(solver='lbfgs', max_iter=1000)
            else:
                model = LinearRegression()

            model.fit(X, y)
            self.models[0] = model

        for k, v in self.models.items():
            self.coef_ = v.coef_
            self.intercept_ = v.intercept_
        
#         print(self.models)
#         print(self.direct_val)
        
    def predict(self, X):
        if len(self.models) == 1 and len(self.direct_val) == 0:
            return self.models[0].predict(X)
        else:
            res = []
            for x in X:
                clust = self.gmm.predict([x])[0]
                if clust in self.direct_val:
                    res.append(self.direct_val[clust])
                else:
                    if clust in self.models:
                        res.append(self.models[clust].predict([x])[0])
                    else:
                        res.append(0)
        return np.array(res)