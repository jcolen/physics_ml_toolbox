import numpy as np
from sklearn.base import TransformerMixin

class FiniteDifference(TransformerMixin):
    """ Apply 2nd order central differences to compute a derivative """
    def __init__(self, axis=1):
        self.axis = axis

    def fit(self, X, y0=None):
        self.input_features_ = X.shape[self.axis]
        return self
    
    def transform(self, X, t):
        x_dot = np.zeros_like(X)

        # Right now, only allow a uniform time grid
        if np.isscalar(t):
            dt = t
        else:
            dt = t[1] - t[0]

        # Central differences in center of domain
        x_dot[1:-1] = (X[2:] - X[:-2]) / (2 * dt)

        # Forward difference on boundary
        x_dot[0] = (X[1] - X[0]) / dt
        x_dot[-1] = (X[-1] - X[-2]) / dt

        return x_dot
