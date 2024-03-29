### Name: piecewise_regressor
# Author: tomio kobayashi
# Version: 1.1.2
# Date: 2024/03/15


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import Counter
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

class piecewise_regressor:
    
    def __init__(self, regression_type="lasso", max_clusters=20):
    # regression_type = {"linear", "multinomial", "logistic", "lasso", "elastic"}
        self.gmm = None
        self.regression_type = regression_type
        self.max_clusters = max_clusters
        self.models = {}
        self.direct_val = {}
        self.coef_ = None
        self.intercept_ = None
        self.all_model = None
        
    def fit(self, X, y, use_aic=True):

        if self.regression_type == "multinomial" or self.regression_type == "multi":
#             self.all_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
            self.all_model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
        elif self.regression_type == "logistic":
#             self.all_model = LogisticRegression(solver='lbfgs', max_iter=1000)
            self.all_model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
        elif self.regression_type == "lasso":
            self.all_model = Lasso()
        elif self.regression_type == "elastic":
            self.all_model = ElasticNet()
        else:
            self.all_model = LinearRegression()
                    
        self.all_model.fit(X, y)
        
#         Testing
#         n_components = np.arange(1, 5)  # Number of clusters to try
#         models = [GaussianMixture(n, covariance_type='full').fit(X) for n in n_components]
#         aic = [m.aic(X) for m in models]
#         bic = [m.bic(X) for m in models]
#         plt.plot(n_components, aic, label='AIC')
#         plt.plot(n_components, bic, label='BIC')
#         plt.legend(loc='best')
#         plt.xlabel('n_components')
#         plt.ylabel('Information Criterion')
#         plt.show()

        num_clusters = 1
        gmm = None
        prev_gmm = None
        prev_aic = float("inf")
        prev_bic = float("inf")
        for n in range(1, self.max_clusters+1, 1):
            try:
                gmm = GaussianMixture(n, covariance_type='full').fit(X)
                aic = gmm.aic(X)
                bic = gmm.bic(X)
#                 print("n", n, "prev_aic", prev_aic, "aic", aic)
#                 print("n", n, "prev_bic", prev_bic, "bic", bic)
                num_clusters = n
                if (use_aic and aic > prev_aic) or (not use_aic and bic > prev_bic):
                    num_clusters = n-1
                    gmm = prev_gmm
                    break
                prev_aic = aic
                prev_bic = bic
                prev_gmm = gmm
            except Exception as e:
                print(e)
                if n > 1:
                    gmm = prev_gmm
                    num_clusters = n-1
                else:
                    return
                
        X_clust = {}
        y_clust = {}
        if num_clusters > 1:
            self.gmm = gmm
            cluster_labels = self.gmm.predict(X)
            for i in range(max(cluster_labels)+1):
                X_clust[i] = np.array([X[j] for j, x in enumerate(cluster_labels) if x == i])
                y_clust[i] = np.array([y[j] for j, x in enumerate(cluster_labels) if x == i])
            for k, v in X_clust.items():
                model = None
                if self.regression_type == "multinomial" or self.regression_type == "multi":
#                     model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
                    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
                elif self.regression_type == "logistic":
#                     model = LogisticRegression(solver='lbfgs', max_iter=1000)
                    model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
                elif self.regression_type == "lasso":
                    model = Lasso()
                elif self.regression_type == "elastic":
                    model = ElasticNet()
                else:
                    model = LinearRegression()

                try:
                    model.fit(v, y_clust[k])
                    coefs = model.coef_[0] if self.regression_type == "multinomial" or self.regression_type == "multi"  or self.regression_type == "logistic" else model.coef_
#                     print(k, "coefs", coefs)
                    if not all([cf == 0 for cf in coefs]):
                        self.models[k] = model
                    else:
#                         print("zero coef")
                        self.models[k] = self.all_model
                except Exception as e:
                    if isinstance(e, ValueError) and self.regression_type == "multinomial" or self.regression_type == "multi" or self.regression_type == "logistic":
                        count_dict = Counter(y_clust[k])
                        if len(count_dict) == 1:
                            self.direct_val[k] = y_clust[k][0]
                    else:
                        print(e)
                        print(k)
                        return

        if len(self.models) > 0:
            for k, v in self.models.items():
                self.coef_ = v.coef_
                self.intercept_ = v.intercept_
                break
        else:
            self.coef_ = self.all_model.coef_
            self.intercept_ = self.all_model.intercept_
            
#         print("self.models", self.models)
        
    def predict(self, X):
        if len(self.models) == 0 and len(self.direct_val) == 0:
            return self.all_model.predict(X)
        else:
            clusts = self.gmm.predict(X)
#             return np.array([self.direct_val[clusts[i]] if clusts[i] in self.direct_val else (self.models[clusts[i]].predict([x])[0] if clusts[i] in self.models else res.append(self.all_model.predict([x])[0])) for i, x in enumerate(X)])
            return np.array([self.direct_val[clusts[i]] if clusts[i] in self.direct_val else self.models[clusts[i]].predict([x])[0] for i, x in enumerate(X)])        
#             return np.array([self.direct_val[clusts[i]] if clusts[i] in self.direct_val else self.models[clusts[i]].predict([x])[0] if clusts[i] in self.models else self.all_model.predict([x])[0] for i, x in enumerate(X)])   
