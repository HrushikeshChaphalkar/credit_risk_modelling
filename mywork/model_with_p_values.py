import numpy as np
from sklearn import linear_model
from scipy import stats
import statsmodels.api as sm

class LogisticRegressionWithPValues:
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        
        # Calculating p-values for coefficients
        denom = 2.0 * (1.0 + np.cosh(self.model.decision_function(X)))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores]
        
        # Store model attributes
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values


class LinearRegressionWithPValues:
    def __init__(self):
        self.model = None
        self.p_values = None

    def fit(self, X, y):
        X = sm.add_constant(X)  # Add a constant term for the intercept
        self.model = sm.OLS(y, X).fit()

        # Calculate p-values for coefficients
        self.p_values = self.model.pvalues[1:]  # Exclude the intercept term

    def predict(self, X):
        X = sm.add_constant(X)
        return self.model.predict(X)