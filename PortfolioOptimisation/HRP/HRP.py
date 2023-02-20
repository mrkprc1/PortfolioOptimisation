from pypfopt.hierarchical_risk_parity import HRPOpt
import pandas as pd
import numpy as np

###############################################
###      Hierachical Risk Parity Class      ###
###############################################


class HRP():

    def __init__(self, assets):
        self.assets = assets

    def fit(self, X, y=None):
        returns = pd.DataFrame(X, columns=self.assets)
        hrp = HRPOpt(returns)

        # Calculate HRP weights.
        self.portfolio = hrp.hrp_portfolio()
        self.weights = np.array([self.portfolio[asset] for asset in self.assets])

    def predict(self, X):
        return self.weights @ X.T