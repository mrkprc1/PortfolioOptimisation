import numpy as np
from qubovert import boolean_var
from qubovert.sim import anneal_qubo

###############################################
###  Quantum Hierachical Risk Parity Class  ###
###############################################


class QHRP():

    # Instantiation method.
    def __init__(self, P, num_anneals=20):
        self.P = P
        self.num_anneals = num_anneals

    # Method for finding portfolio weights.
    def fit(self, X, y=None):
        
        # Find covariance matrix.
        cov_mat = np.cov(X.T)
        self.cov_mat = cov_mat

        # Determine the number of assets.
        n_assets = X.shape[1]
        self.n_assets = n_assets

        # Create QUBO.
        qubo = self.create_qubo(n_assets=n_assets, cov_mat=cov_mat, P=self.P)

        # Minimse QUBO.
        num_bits = qubo.shape[0]
        x = {i : boolean_var(f'x{i}') for i in range(num_bits)}
        y = np.array([x[i] for i in range(num_bits)])
        model = y @ qubo @ y

        # Run simulated annealing.
        res = anneal_qubo(model, num_anneals=self.num_anneals)

        sol = [x for x in res.best.state.items()]
        P_mat = np.array(sol.configuration).reshape((n_assets,n_assets))

        # Check solution is a permutation matrix.
        is_orth = np.array_equal(P_mat @ P_mat.T , np.identity(n_assets))
        col_sums = np.array_equal(P_mat.T @ np.ones((n_assets,1)), np.ones((n_assets,1)))
        row_sums =  np.array_equal(P_mat @ np.ones((n_assets,1)), np.ones((n_assets,1)))

        self.check_orth = {'orthogonal' : is_orth, 
                           'row_sums' : row_sums,
                           'col_sums' : col_sums}

        # Get re-ordered covariance matrix.
        sol_mat = P_mat @ cov_mat @ P_mat.T

        # Find weights.
        weights = P_mat @ self.split(cov_mat=sol_mat)

        self.sol_mat = sol_mat
        self.weights = weights

    
    def predict(self, X):
        return self.weights @ X.T


    # Static method for creating qubo.
    @staticmethod
    def create_qubo(n_assets, cov_mat, P=1):

        # Create matrix of square index distances.
        idx_row = np.kron(np.array(range(n_assets)).reshape((n_assets,1)), np.ones((1, n_assets)))
        D = (idx_row - idx_row.T)**2

        # Create QUBO for quasi-diagonal term.
        Q1 = np.kron(D, np.abs(cov_mat))

        # Create QUBO for row constraints.
        A1 = np.kron(np.ones((1, n_assets)), np.identity(n_assets))
        Q2 = P * A1.T @ A1 - 2 * P * np.diag((A1.T @ np.ones((n_assets, 1))).T[0])

        # Create QUBO for column constraints.
        A2 = np.kron(np.identity(n_assets), np.ones((1, n_assets)))
        Q3 = P * A2.T @ A2 - 2 * P * np.diag((A2.T @ np.ones((n_assets, 1))).T[0])

        qubo = Q1 + Q2 + Q3
        return qubo 

    # Static method for splitting budget.
    @staticmethod
    def split(cov_mat, budget=1):
    
        if cov_mat.shape[0] == 1:
            return budget

        # Find block diagonal splitting of covariance matrix that minimises the mean absolute
        # value of the off-diagonal block.
        n = cov_mat.shape[0]
        metric_scores = np.array([float("inf")] * (n-1))
        for i in range(1, n):
            metric_scores[i-1] = np.mean(np.abs(cov_mat[0:i, i:]))
            min_i = np.argmin(metric_scores) + 1

        # Select block diagonal sub-matrices.
        C1 = cov_mat[0:min_i, 0:min_i]
        C2 = cov_mat[min_i:, min_i:]

        # Calculate the weightings for diagonal blocks.
        def calculate_V(sub_cov_mat):
            sub_diag_inv = 1/np.diag(sub_cov_mat)
            w = sub_diag_inv / np.sum(sub_diag_inv)
            V = w @ sub_cov_mat @ w
            return V

        V1 = calculate_V(C1)
        V2 = calculate_V(C2)
        s = [V2 / (V1 + V2), V1 / (V1 + V2)]

        # Split submatrices.
        alpha1 = QHRP.split(C1, s[0] * budget)
        alpha2 = QHRP.split(C2, s[1] * budget)
        
        return np.append(alpha1, alpha2)


