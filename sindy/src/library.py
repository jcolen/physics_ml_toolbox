import numpy as np

from sklearn.base import TransformerMixin
from itertools import chain, combinations, combinations_with_replacement

class PolynomialLibrary(TransformerMixin):
    def __init__(self, 
                 degree=3,
                 include_interaction=True,
                 interaction_only=False,
                 include_bias=True):
        
        # Input validation
        if degree < 0 or not isinstance(degree, int):
            raise ValueError("degree must be a positive integer")

        self.degree = degree
        self.include_interaction = include_interaction
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def _combinations(self):
        """ Get iterator yielding combinations as in sklearn.preprocessing.PolynomialFeatures """
        if self.include_interaction:
            if self.interaction_only:
                # Combinations without replacement gives interaction terms only
                _combinations = chain.from_iterable(
                    combinations(range(self.input_features_), i) for i in range(1, self.degree + 1)
                )
            else:
                # Allow replacement to enable e.g. x0^2 terms
                _combinations = chain.from_iterable(
                    combinations_with_replacement(range(self.input_features_), i) for i in range(1, self.degree + 1)
                )
        else:
            # Only allow non-interacting powers up to self.degree
            _combinations = chain((
                exp * (feat_idx,)
                for exp in range(1, self.degree + 1)
                for feat_idx in range(self.input_features_)
            ))

        return _combinations
    
    def get_feature_names(self, input_features=None):
        """ Returns the polynomial feature name for each combination """
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.input_features_)]
        assert len(input_features) == self.input_features_

        feature_names = []
        powers = np.vstack(
            [np.bincount(c, minlength=self.input_features_) for c in self._combinations()]
        )
        for row in powers:
            inds = np.where(row)[0]
            name = " ".join(
                f"{input_features[ind]}^{exp}" if exp != 1 else input_features[ind]
                for ind, exp in zip(inds, row[inds])
            )
            feature_names.append(name)

        if self.include_bias:
            feature_names.append("1")

        return feature_names

    def fit(self, X, y=None):
        """ Just get the correct shape information """
        self.input_features_ = X.shape[-1]
        
        self.output_features_ = sum(1 for _ in self._combinations())
        if self.include_bias:
            self.output_features_ += 1
        
        return self

    def transform(self, X):
        """ Compute the polynomial features """
        n_samples, n_features = X.shape
        assert n_features == self.input_features_

        library = np.zeros([n_samples, self.output_features_], dtype=X.dtype)
        for i, comb in enumerate(self._combinations()):
            library[:, i] = np.prod(X[:, comb], axis=1)

        if self.include_bias:
            library[:, -1] = 1.

        return library
        
