import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import ridge_regression

class STLSQ(BaseEstimator):
    def __init__(self, 
                 threshold=0.1,
                 alpha=0.1,
                 max_iter=20):
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        """ Perform the sequentially thresholded least squares algorithm """
        self.input_features_ = X.shape[1]
        self.output_features_ = y.shape[1]
        self.n_samples_ = X.shape[0]

        assert X.shape[0] == y.shape[0]

        # Initial guess is the least squares solution
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        self.indices_ = np.ones(self.coef_.shape, dtype=bool)

        for ii in range(self.max_iter):
            for jj in range(self.output_features_):
                # Perform ridge regression on the current target variable using the valid coefficients
                valid_indices = self.indices_[:, jj]
                X_subset = X[:, valid_indices]
                
                if np.sum(valid_indices) == 0:
                    continue

                coef_ij = ridge_regression(X_subset, y[:, jj], alpha=self.alpha)

                # Threshold the coefficients following the regression
                coef_ij = np.where(np.abs(coef_ij) >= self.threshold, coef_ij, 0)

                # Update the coefficients and indices
                self.coef_[valid_indices, jj] = coef_ij
                self.indices_[:, jj] = np.abs(self.coef_[:, jj]) >= self.threshold


        return self

    def predict(self, X):
        return X @ self.coef_