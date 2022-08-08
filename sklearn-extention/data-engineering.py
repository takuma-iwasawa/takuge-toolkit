# -*- coding: utf-8 -*-
import re
from functools import reduce as ftreduce
from typing import Union, Tuple, List, Optional, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, _transform_one, _fit_transform_one


class FeatureList(FeatureUnion):
    def __init__(self, transformer_list, transformer_weights=None):
        super().__init__(transformer_list=transformer_list, transformer_weights=transformer_weights)

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = (
            _fit_transform_one(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return Xs

    def transform(self, X):
        print('transform')
        Xs = (_transform_one(trans, X, None, weight)
              for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        return list(Xs)


class FeatureDict(FeatureUnion):
    def __init__(self, transformer_list, transformer_weights=None):
        super().__init__(transformer_list=transformer_list, transformer_weights=transformer_weights)

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = (
            _fit_transform_one(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        names, _, _ = zip(*self._iter())
        return dict(zip(names, Xs))

    def transform(self, X):
        print('transform')
        Xs = (_transform_one(trans, X, None, weight)
              for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        names, _, _ = zip(*self._iter())
        return dict(zip(names, Xs))


class FeatureJoin(FeatureUnion):
    def __init__(self, transformer_list, transformer_weights=None, how: str = 'left'):
        self.how = how
        super().__init__(transformer_list=transformer_list, transformer_weights=transformer_weights)

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = (
            _fit_transform_one(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter()
        )

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        return ftreduce(
            lambda i, j: i.to_frame().join(j, how=self.how) if type(i) == pd.Series else i.join(j, how=self.how),
            list(Xs)
        )

    def transform(self, X):
        Xs = (
            _transform_one(trans, X, None, weight)
            for name, trans, weight in self._iter()
        )

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return ftreduce(
            lambda i, j: i.to_frame().join(j, how=self.how) if type(i) == pd.Series else i.join(j, how=self.how),
            list(Xs)
        )

    def predict(self, X):
        return self.transform(X)

    def fit_predict(self, X, y=None, **fit_params):
        return self.fit_transform(X, y=None, **fit_params)


class ReplaceSeriesIndexTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None):
        return pd.Series(X[0].values, index=X[1], name=X[0].name)


class DataSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, data_names):
        self.data_names = data_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.data_names]


class ConvertDictTransformer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, data_names: Union[list, str]):
        self.data_names = data_names if type(data_names) == list else [data_names]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return ftreduce(lambda x, y: {y: x}, [X] + self.data_names)
        # return {self.data_names: X}


