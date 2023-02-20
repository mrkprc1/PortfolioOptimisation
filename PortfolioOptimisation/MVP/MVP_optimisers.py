import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import minimize
import cvxpy as cp
from numbers import Number
from inspect import signature

class optimiser(ABC):

    @abstractmethod
    def __init__(self, cov_mat, mu, A_eq, A_ineq, b_eq, b_ineq, **kwargs):
        pass

    @abstractmethod
    def minimise(self):
        pass

class sp_optimiser(optimiser):

    def __init__(self, cov_mat, mu, A_eq, A_ineq, b_eq, b_ineq, **kwargs):

        # Get method if it is passed.
        if 'method' in kwargs.keys():
            method = kwargs['method']
        else:
            method = "SLSQP"

        # Turn linear constraints into constraints for scipy.
        constraints_eq = [{'type': 'eq', 'fun' : lambda x: A_eq[i,:] @ x - b_eq[i]} for i in range(A_eq.shape[0])]

        if A_ineq is not None:
            constraints_ineq = [{'type': 'ineq', 'fun' : lambda x: A_ineq[i,:] @ x - b_ineq[i]} for i in range(A_ineq.shape[0])]
            constraints = constraints_eq + constraints_ineq
        else:
            constraints = constraints_eq

        # Perform minimisation.
        n_assets = cov_mat.shape[0]

        constraints = [{'type': 'eq','fun': lambda x: np.sum(x) - 1},
                       {'type': 'eq','fun': lambda x: (x.T @ mu) - b_eq[0]}] 

        self.method = method
        self.constraints = constraints
        self.n_assets = n_assets
        self.cov_mat = cov_mat
        self.mu = mu
        self.target_return = b_eq[0]

    def minimise(self):

        res = minimize(fun = sp_optimiser.objective_func, 
                       jac = sp_optimiser.jacobian,
                       x0 = np.ones(self.n_assets) / self.n_assets, 
                       method = self.method, 
                       args = (self.cov_mat),
                       bounds = ((0, 1),) * self.n_assets,
                       constraints = self.constraints)

        return res

    # Define objective function.
    @staticmethod
    def objective_func(x, cov_mat):
        return x.T @ cov_mat @ x 

    # Define Jacobian of objective.
    @staticmethod
    def jacobian(x, cov_mat):
        return 2 * cov_mat @ x


class cvx_optimiser(optimiser):

    def __init__(self, cov_mat, mu, A_eq, A_ineq, b_eq, b_ineq, **kwargs):

        # Get method if it is passed.
        if 'delta' in kwargs.keys():
            delta = kwargs['delta']
        else:
            delta = 1

        # # Turn linear constraints into constraints for scipy.
        # constraints_eq = [{'type': 'eq', 'fun' : lambda x: A_eq[i,:] @ x - b_eq[i]} for i in range(A_eq.shape[0])]

        # if A_ineq is not None:
        #     constraints_ineq = [{'type': 'ineq', 'fun' : lambda x: A_ineq[i,:] @ x - b_ineq[i]} for i in range(A_ineq.shape[0])]
        #     constraints = constraints_eq + constraints_ineq
        # else:
        #     constraints = constraints_eq

        # Perform minimisation.
        n_assets = cov_mat.shape[0]

        self.delta = delta
        self.target_return = b_eq[0]
        self.n_assets = n_assets
        self.cov_mat = cov_mat
        self.mu = mu

    def minimise(self):

        # Determine vector of deltas.
        if self.delta == 1:
            x = cp.Variable((self.n_assets,1))
            delta = np.ones((self.n_assets, 1))
        elif isinstance(self.delta, Number):
            x = cp.Variable((self.n_assets,1), integer=True)
            delta = np.atleast_2d(np.array([self.delta] * self.n_assets)).T
        elif isinstance(self.delta, list):
            x = cp.Variable((self.n_assets,1), integer=True)
            delta = np.atleast_2d(np.array(self.delta)).T

        # Calculate matrix for objective function
        A = np.multiply(delta @ delta.T, self.cov_mat)

        # Define objective function.
        objective = cp.Minimize(cp.quad_form(x, A))
        constraints = [x >= 0, delta.T @ x == 1, np.multiply(np.atleast_2d(self.mu).T, delta).T @ x == self.target_return]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        res = {'x' : np.multiply(delta, x.value).ravel()}

        return res

