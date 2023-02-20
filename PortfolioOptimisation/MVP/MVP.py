import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin

###############################################
###      Mean Variance Portfolio class      ###
###############################################

class MVP(BaseEstimator, TransformerMixin):

    # Instantiation method.
    def __init__(self, opt, target_return, A_eq=None, A_ineq=None, b_eq=None, b_ineq=None, **kwargs):
        self.target_return = target_return
        self.A_eq = A_eq
        self.A_ineq = A_ineq
        self.b_eq = b_eq
        self.b_ineq = b_ineq
        self.opt = opt
        self.kwargs = kwargs
        
    # Fit method.
    def fit(self, X, y=None):

        # Get first and second moments.
        cov_mat = np.cov(X.T)
        mu = np.mean(X, axis=0)

        # Add target return constraint and summation constraint to equality constraints.
        n_assets = cov_mat.shape[0]
        A_eq = np.concatenate((np.atleast_2d(mu), np.ones((1, n_assets))), axis=0)
        b_eq = np.array([self.target_return, 1])

        if self.A_eq is not None:
            A_eq = np.concatenate((self.A_eq, A_eq), axis=1)
            b_eq = np.concatenate((self.b_eq, b_eq), axis=0)

        # Instantiate optimiser class.
        optim = self.opt(cov_mat=cov_mat, 
                         mu=mu, 
                         A_eq=A_eq, 
                         A_ineq=self.A_ineq, 
                         b_eq=b_eq, 
                         b_ineq=self.b_ineq,
                         **self.kwargs)

        # Perform minimisation.
        res = optim.minimise()

        self.cov_mat = cov_mat
        self.mu = mu
        self.weights = np.ravel(res['x'])

    # Predict method.
    def predict(self, X):

        return self.weights @ X.T
