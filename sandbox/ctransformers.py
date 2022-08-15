'''
Sklearn api class for transforming the CELLDYN data
'''

from re import T
import numpy as np
import pandas as pd
import scipy as sc

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

class CellDynTrans(BaseEstimator, TransformerMixin):
    '''
    Class for transforming the CELLDYN data
    '''
    def __init__(self, scaler='standard', **kwargs):
        '''
        Initialize the class
        '''
        self.scaler = scaler
        self.kwargs = kwargs
        self.scaler_ = None
        self.transformer_ = None

    def fit(self, X, y=None):
        '''
        Fit the scaler and transformer
        '''
        self.scaler_ = self.get_scaler(self.scaler