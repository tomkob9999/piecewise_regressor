### Name: piecewise_regressor
# Author: tomio kobayashi
# Version: 1.0.4
# Date: 2024/02/18


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import Counter

class piecewise_regressor:
    
    def __init__(self, regression_type="linear", max_clusters=6):
    # regression_type = {"linear", "multinomial", "logistic"}
        self.gmm = None
        self.regression_type = regression_type
        self.max_clusters = max_clusters
        self.models = {}
        self.direct_val = {}
        self.coef_ = None
        self.intercept_ = None
        self.all_model = None
        
    def fit(self, X, y):

        if self.regression_type == "multinomial" or self.regression_type == "multi":
            self.all_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        elif self.regression_type == "logistic":
            self.all_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        else:
            self.all_model = LinearRegression()

        self.all_model.fit(X, y)
            
        n_components = np.arange(1, self.max_clusters)
        
        models = []
        try:
            models = [GaussianMixture(n, covariance_type='full').fit(X) for n in n_components]
        except Exception as e:
            return
        aic = [model.aic(X) for model in models]
        bic = [model.bic(X) for model in models]
#         print("aic", aic)
#         print("bic", bic)
        
        num_clusters = 1
        min_aic = float("inf")
        min_bic = float("inf")
        aic_up = False
        bic_up = False
        for j in range(len(models)):
            if aic[j] < min_aic:
                min_aic = aic[j]
            else:
                aic_up = True
            if bic[j] < min_bic:
                min_bic = bic[j] 
            else:
                bic_up = True
            if aic_up or bic_up:
                num_clusters = j
                break
        
#         print("num_clusters", num_clusters)
        X_clust = {}
        y_clust = {}
        if num_clusters > 1:
#             self.gmm = GaussianMixture(n_components=num_clusters)
#             self.gmm.fit(X)
            self.gmm = models[num_clusters-1]
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
                    if self.regression_type == "multinomial" or self.regression_type == "multi" or self.regression_type == "logistic":

                        count_dict = Counter(y_clust[k])
                        if len(count_dict) == 1:
                            self.direct_val[k] = y_clust[k][0]

        if len(self.models) > 0:
            for k, v in self.models.items():
                self.coef_ = v.coef_
                self.intercept_ = v.intercept_
                break
        else:
            self.coef_ = self.all_model.coef_
            self.intercept_ = self.all_model.intercept_
            
        
    def predict(self, X):
        if len(self.models) == 0 and len(self.direct_val) == 0:
            return self.all_model.predict(X)
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
                        res.append(self.all_model.predict([x])[0])
        return np.array(res)