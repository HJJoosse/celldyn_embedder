'''
A class to thinly wrap different imputers in the sklearn api
'''

from statistics import mean
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from miceforest import ImputationKernel
from miceforest import mean_match_default, mean_match_fast_cat, mean_match_shap
from fancyimpute import KNN, SoftImpute, IterativeSVD, BiScaler, 
from fancyimpute import NuclearNormMinimization, MatrixFactorization
from autoimpute.imputations import SingleImputer, MultipleImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# add an output logger that can be used to log the imputation process
import logging
logger = logging.getLogger(__name__)



backend_compat ={
    'miceforest':{
        'imputer': ['forest']
    },
    'fancyimpute':{
        'imputer': ['knn', 'softimpute', 'svd', 'biscaler', 'mf', 'nnm', 'simple'],
        'clf': ['mean', 'median', 'mode']
    },
    'autoimpute':{
        'imputer': ['mice', 'multi', 'simple'],
        'clf': ['pmm', 'lrd', 'bayesian_binary_logistic', 'bayesian_least_squares',
                'least_squares', 'multinomial_logistic', 'mean', 'median', 'binary_logistic']
    },
    'sklearn':{
        'imputer': ['iterative', 'simple', 'knn'],	
        'clf': ['rf', 'hgb', 'lr', 'svr', 'svc', 'bayesian_ridge', 'xgb', 'lgbm', 'mean', 'median']
    }
}

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                imputer: str = 'forest',
                backend: str = 'miceforest',
                add_mia: bool=False,
                meas_cols: list =[], 
                **kwargs):
        '''
        Imputer class to wrap different FCS imputers in the sklearn api

        Parameters for the imputer:
            imputer: the imputer to use
            backend: the backend to use
            meas_cols: the columns to impute
            kwargs: the parameters for the imputer

        Options:
            imputer: 'forest', 'knn', 'softimpute', 'iterative_svd', 'biscaler', 
                     'NuclearNormMinimization', 'iterative_imputer', 'MatrixFactorization',
                     'linear_regression', 'logistic_regression', 'bayesian_ridge',
                     'pmm', 'lrd'

            backend: 'miceforest', 'fancyimpute', 'autoimpute', 'sklearn'
        '''
        self.imputer = imputer
        self.backend = backend
        self.meas_cols = meas_cols
        self.add_mia = add_mia
        self.kwargs = kwargs

        if backend == 'miceforest':
            try:
                num_mean_matches = self.kwargs['num_match_candidates']
            except KeyError:
                num_mean_matches = 5

            self.mean_math = mean_match_default
            self.mean_match.set_mean_match_candidates()
        elif backend == 'fancyimpute':
            
        elif backend == 'autoimpute':

        elif backend == 'sklearn':
            self.imputer = IterativeImputer(**kwargs)

    def fit(self, X, y=None):
        if self.backend == 'miceforest':
            self.imputer = ImputationKernel(
                                X[self.meas_columns],
                                save_all_iterations=False,
                                random_state=1283,
                                mean_match_scheme=self.mean_match
                                )
            self.imputer.mice(iterations=5, verbose=True, n_estimators=100)

        return self

    def transform(self, X):
        if self.backend == 'miceforest':
            imputed_df = self.imputer.complete_data(X[self.meas_columns])

        if self.add_mia:
            mia_df = X.isna().astype(int)
            imputed_df = pd.concat([imputed_df, mia_df], axis=1)

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs
        return self