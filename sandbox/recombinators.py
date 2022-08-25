import numpy as np
import pandas as pd
import scipy as sc

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin


default_combinations = [(['mon','mone'],['lym', 'lyme', 'vlym']), 
                        (['bas', 'eos', 'neu'], ['seg', 'bnd', 'mon', 'vlym',
                                                'lym', 'lyme', 'nrbc']),
                        (['pmon', 'pmone'], ['pvlym','plym','plyme']),
                        (['pneu', 'pbas', 'peos'], ['pseg', 'pbnd', 'pmon', 
                                                'pvlym', 'plyme','plym',
                                                'pnrbc'])]
default_combo_functions = [(lambda x,y: (np.log10(x+1)+1)/(np.log(y+1)+1), lambda col1,col2: f"{col1}:{col2}")]
default_removal = ['pneu:pseg', 'pbas:pseg', 'peos:pseg', 'neu:seg']

class CellDynRecombinator(BaseEstimator, TransformerMixin):
    def __init__(self, combinations: list=[], combo_functions: list=[], removal: list=[]):
        '''
        Combines columns in X according to the specified combinations.
        Params:
            combinations: list of tuples with lists of column names to combine: [(['col1','col2'],['col3','col4'])]
            combo_functions: list of tuples of functions to apply to the combinations: [(fun, colname_fun)]
            removal: list of combinations to remove
        '''
        
        self.combinations = default_combinations if len(combinations)==0 else combinations
        self.combo_functions = default_combo_functions if len(combo_functions)==0 else combo_functions
        self.removal = default_removal if len(removal)==0 else removal

    def _combine(self, X: pd.DataFrame) -> pd.DataFrame:
        for left, right in self.combinations:
            for l in left:
                for r in right:           
                    for func, colname_fun in self.combo_functions:
                        X[colname_fun(l, r)] = func(X['c_b_'+l], X['c_b_'+r]) 
        return X

    def fit(self, X, y=None):
        print('Fitting')
        self._X = self._combine(X.copy())
        if len(self.removal)>0:
            self._X.drop(self.removal, axis=1, inplace=True)
        return self

    def transform(self, X, y=None):
        return self._X