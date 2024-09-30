import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import ridge_regression
from scipy.linalg import cho_factor, cho_solve

class SR3(BaseEstimator):
    def __init__(self, 
                 reg_weight=0.005,
                 relax_coeff=1.0,
                 regularizer='L0',
                 max_iter=20):
    
        self.reg_weight = reg_weight
        self.relax_coeff = relax_coeff
        self.regularizer = regularizer
        assert self.regularizer in ['L0', 'L1', 'L2']
        self.max_iter = max_iter

    def fit(self, X, y):
        """ Perform the sparse relaxed regularized regression algorithm 
            See https://arxiv.org/pdf/1906.10612 for more details
        """
        self.input_features_ = X.shape[1]
        self.output_features_ = y.shape[1]
        self.n_samples_ = X.shape[0]

        assert X.shape[0] == y.shape[0]

        # Initial guess is the least squares solution
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]

        # Optimization is relaxed to minimize the objective
        # 0.5 || y - X w ||^2 + reg_weight R(u) + 0.5 || w - u ||^2 / relax_coeff
        coef_w = np.copy(self.coef_)
        coef_u = np.copy(self.coef_)

        for ii in range(self.max_iter):
            # Update the full variables
            # w = argmin 0.5 || y - X w ||^2 + 0.5 || w - u ||^2 / relax_coeff
            coef_w = cho_solve(
                cho_factor(np.dot(X.T, X) + np.diag(np.full(X.shape[1], 1. / self.relax_coeff))),
                np.dot(X.T, y) + coef_u / self.relax_coeff
            )

            # Update the regularized variables
            # u = prox_{reg_weight x relax_coef} (w)
            reg_relax = self.reg_weight * self.relax_coeff
            if self.regularizer == 'L0':
                # prox is a hard threshold
                threshold = np.sqrt(2 * reg_relax)
                coef_u = np.where(np.abs(coef_w) >= threshold, coef_w, 0)
            elif self.regularizer == 'L1':
                    # prox is a clipped deviation
                    coef_u = np.sign(coef_w) * np.maximum(np.abs(coef_w) - reg_relax, 0)
            elif self.regularizer == 'L2':
                    # prox is a shrinking operator
                    coef_u = coef_w / (1 + 2 * reg_relax)

        self.coef_ = coef_u
        self.coef_full_ = coef_w

        return self

    def predict(self, X):
        return X @ self.coef_