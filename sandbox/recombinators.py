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
default_combo_functions = [(lambda x,y: (np.log10(x+1)+1)/(np.log(y+1)+1), 
                            lambda col1,col2: f"COMBO_F({col1}:{col2})")]
default_removal = ['COMBO_F(pneu:pseg)', 
                   'COMBO_F(pbas:pseg)', 
                   'COMBO_F(peos:pseg)', 
                   'COMBO_F(neu:seg)']

class CellDynRecombinator(BaseEstimator, TransformerMixin):
    def __init__(self, 
                combinations: list=[], 
                combo_functions: list=[],
                removal: list=[],
                scaler=None,
                scaler_kwargs: dict={}):
        '''
        Combines columns in X according to the specified combinations.
        Params:
            combinations: list of tuples with lists of column names to combine: [(['col1','col2'],['col3','col4'])]
            combo_functions: list of tuples of functions to apply to the combinations: [(fun, colname_fun)]
            removal: list of combinations to remove
            scaler: sklearn scaler to apply to the combo features
        '''
        
        self.combinations = default_combinations if len(combinations)==0 else combinations
        self.combo_functions = default_combo_functions if len(combo_functions)==0 else combo_functions
        self.removal = default_removal if len(removal)==0 else removal
        self.scaler = scaler
        self.scaler_kwargs = scaler_kwargs

    def _combine(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = X.columns.tolist()
        combo_cols = []
        for left, right in self.combinations:
            for l in left:
                for r in right:     
                    if ('c_b_'+l in cols) & ('c_b_'+r in  cols): 
                        for func, colname_fun in self.combo_functions:
                            X[colname_fun(l, r)] = func(X['c_b_'+l], X['c_b_'+r]) 
                            combo_cols.append(colname_fun(l, r))
        self.combo_cols = [c for c in combo_cols if c not in self.removal]
        return X

    def fit(self, X, y=None):
        print('Fitting')
        self._X = self._combine(X.copy())
        if len(self.removal)>0:
            self._X.drop(self.removal, axis=1, inplace=True)
        if self.scaler is not None:
            self._X.loc[:, self.combo_cols] = self.scaler(**self.scaler_kwargs)\
                                            .fit_transform(self._X[self.combo_cols])
        self.out_columns = self._X.columns.tolist()
        return self

    def transform(self, X, y=None):
        return self._X