import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from scipy.integrate import solve_ivp

class MySINDy(BaseEstimator):
    def __init__(self,
                 library : TransformerMixin,
                 differentiator: TransformerMixin,
                 optimizer: RegressorMixin):
        self.library = library
        self.optimizer = optimizer
        self.differentiator = differentiator
    
    def fit(self, X, X_dot=None, t=None):
        if X_dot is None and t is None:
            raise ValueError("Either X_dot or t must be provided")
        
        # Get time derivatives
        if X_dot is None:
            self.differentiator.fit(X)
            X_dot = self.differentiator.transform(X, t)
        
        # Generate sklearn pipeline
        self.pipeline = Pipeline([
            ("library", self.library),
            ("optimizer", self.optimizer)
        ])
        self.pipeline.fit(X, X_dot)

        self.input_features_ = self.library.input_features_
        self.output_features_ = self.library.output_features_
        self.predict_features_ = self.optimizer.output_features_

        return self
    
    def predict(self, x):
        return self.pipeline.predict(x)

    def simulate(self, x0, t):
        """ Simulate the inferred model forward in time """
        
        def rhs(t, x):
            return self.predict(x[None, :])[0]
        
        return solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, method='LSODA', rtol=1e-12, atol=1e-12).y.T

    def print(self, lhs=None, precision=3, input_features=None, threshold=1e-8):
        """ Print the inferred equations """
        feature_names = self.library.get_feature_names(input_features)
        coefs = self.optimizer.coef_
        
        for i in range(self.predict_features_):
            eqn_string = lhs[i] if lhs is not None else f"x{i}"
            lhs_i = lhs[i] if lhs is not None else f"x{i}"
            eqn_string = f'({lhs_i})\' = '

            for coef, feature in zip(coefs[:, i], feature_names):
                if hasattr(self.optimizer, 'threshold') and np.abs(coef) >= self.optimizer.threshold:
                    eqn_string += f"{coef:.{precision}f} {feature} + "
                elif np.abs(coef) >= threshold:
                    eqn_string += f"{coef:.{precision}f} {feature} + "

            eqn_string = eqn_string[:-3]

            print(eqn_string)