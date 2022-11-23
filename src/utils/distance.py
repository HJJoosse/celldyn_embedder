import os
import sys
import numpy
from scipy.spatial import distance
from scipy.sparse import coo_matrix
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from numba import jit, njit, float32
from numpy import linalg as la


@njit(fastmath=True)
def poincarre_dist(x, y):
    return numpy.arccosh(
        1
        + 2
        * (
            la.norm(x - y, ord=2) ** 2
            / ((1 - la.norm(x, ord=2) ** 2) * (1 - la.norm(y, ord=2) ** 2))
        )
    )


@njit(float32(float32[:], float32[:], float32), fastmath=True)
def fractional_distance(x, y, f=0.5):
    return numpy.power(numpy.abs(numpy.sum(numpy.power(x - y, f))), 1 / f)


class Distance(BaseEstimator, TransformerMixin):
    def __init__(self, metric="seuclidean", n_jobs=1, batch_size=1000, threshold=0.75):
        self.distance = distance
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        if callable(metric):
            self.dfun = metric
        elif metric in ["poincarre", "poincare"]:
            self.dfun = poincarre_dist
        elif metric in ["fractional", "fractional_distance"]:
            self.dfun = fractional_distance
        else:
            self.dfun = distance._METRIC_ALIAS[metric].dist_func

    @njit(parallel=True)
    def _construct_index_tuple_list(self):
        index_tuple_list = []
        for i in range(self._X.shape[0]):
            for j in range(i + 1, self._X.shape[0]):
                index_tuple_list.append((i, j))
        self.index_tuple_list = index_tuple_list

    @jit
    def _compute_distance_tuples(self):
        result = []
        for i, j in self.index_tuple_list:
            dist = self.dfun(self._X[i], self._X[j])
            if dist < self.threshold:
                result.append((dist, i, j))
        self.distance_tuples = result

    def _set_range(self):
        """
        Make sure that the bounds are [-1,1] or [0,1]
        """
        if self._X.max() > 0:
            self._X = (self._X - self._X.min()) / (self._X.max() - self._X.min())
        return self

    def _make_sparse(self):
        data, i, j = zip(*self.distance_tuples)
        return coo_matrix((data, (i, j)))

    def fit(self, X, y=None):
        self._X = X
        self._set_range()
        self._construct_index_tuple_list()
        self._compute_distance_tuples()
        self.Sparse = self._make_sparse()

        return self

    def transform(self, X):
        return self.Sparse
