import numpy as np
import pandas as pd
import scipy as sc

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import (
    QuantileTransformer,
    PowerTransformer,
    FunctionTransformer,
)
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

base_transformations = {
    'COMBO_SII': lambda x: np.clip(np.log10(x+1), 0, 6),
    'COMBO_NHL': lambda x: np.clip(np.log10(x+1), 0, 3),
    'COMBO_NLR': lambda x: np.clip(np.log10(x+1), 0, 2),
    'COMBO_PLR': lambda x: np.clip(np.log10(x+1), 0, 5),
    'COMBO_HLR': lambda x: np.clip(np.log10(x+1), 0, 2),
    'COMBO_LMR': lambda x: np.clip(np.log10(x+1), 0, 1.5),
    'COMBO_WRR': lambda x: np.clip(np.log10(x+1), 0, 1.),
    'COMBO_NWR': lambda x: np.clip(np.log10(x+1), 0, 1.),
    'COMBO_WPR': lambda x: np.clip(np.log10(x+1), 0,0.06),
    'COMBO_PRR': lambda x: np.clip(np.log10(x+1), 0.5, 3),
    'COMBO_RHR': lambda x: np.clip(np.log10(x+1), 0, 1.75),
    'COMBO_RIR': lambda x: np.clip(np.log10(x+1), 0, 1),
    'COMBO_HHR': lambda x: np.clip(np.log10(x+1), 0.7, 0.85),
    'COMBO_LSR': lambda x: np.log10(np.clip(x, 1, 200)+1),
    'COMBO_PMR': lambda x: np.log10(np.clip(x, 1, 8000)+1),
    'COMBO_MHR': lambda x: np.log10(x+1),
    'COMBO_MMR': lambda x: np.log10(x+1),
    'c_b_neu': lambda x: np.log10(x+1),
    'c_b_seg': lambda x: np.log10(x+1),
    'c_b_wbc': lambda x: np.log10(x+1),
    'c_b_wvf': lambda x: np.log10(np.clip(1-x, 0, 0.05)*100+1)
}


default_combinations = [
    (["mon", "mone", "plto"], ["lym", "lyme", "vlym"]),
    (["bas", "eos", "neu"], ["seg", "bnd", "mon", "vlym", "lym", "lyme", "nrbc"]),
    (["pmon", "pmone"], ["pvlym", "plym", "plyme"]),
    (
        ["pneu", "pbas", "peos"],
        ["pseg", "pbnd", "pmon", "pvlym", "plyme", "plym", "pnrbc"],
    ),
]

lambda_fun = lambda x, y: (np.log10(x + 1) + 1) / (np.log(y + 1) + 1)
lambda_col_fun = lambda col1, col2: f"COMBO_{col1}_over_{col2}"

default_combo_functions = [(lambda_fun, lambda_col_fun)]
default_removal = [
    "COMBO_pneu_over_pseg",
    "COMBO_pbas_over_pseg",
    "COMBO_peos_over_pseg",
    "COMBO_neu_over_seg",
]


class CellDynRecombinator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        combinations: list = [],
        combo_functions: list = [],
        removal: list = [],
        scaler=None,
        scaler_kwargs: dict = {},
        base_combos: bool = True,
    ):
        """
        Combines columns in X according to the specified combinations.
        Params:
            combinations: list of tuples with lists of column names to combine: [(['col1','col2'],['col3','col4'])]
            combo_functions: list of tuples of functions to apply to the combinations: [(fun, colname_fun)]
            removal: list of combinations to remove
            scaler: sklearn scaler to apply to the combo features
        """

        self.combinations = (
            default_combinations if len(combinations) == 0 else combinations
        )
        self.combo_functions = (
            default_combo_functions if len(combo_functions) == 0 else combo_functions
        )
        self.removal = default_removal if len(removal) == 0 else removal
        self.scaler = scaler
        self.scaler_kwargs = scaler_kwargs
        self.base_combos = base_combos
        self.combo_cols = []
        self.errors = []

    def _combine(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = X.columns.tolist()
        for left, right in self.combinations:
            for l in left:
                for r in right:
                    if ("c_b_" + l in cols) & ("c_b_" + r in cols):
                        for func, colname_fun in self.combo_functions:
                            X[colname_fun(l, r)] = func(X["c_b_" + l], X["c_b_" + r])
                            self.combo_cols.append(colname_fun(l, r))
        self.combo_cols = [c for c in self.combo_cols if c not in self.removal]
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = self._combine(X.copy())

        if self.base_combos:
            _X = _X.assign(COMBO_SII=_X.c_b_plto*_X.c_b_neu/(_X.c_b_lym))
            _X = _X.assign(COMBO_NHL=_X.c_b_neu*_X.c_b_hb/_X.c_b_lym)
            _X = _X.assign(COMBO_NLR=_X.c_b_neu/_X.c_b_lym)
            _X = _X.assign(COMBO_PLR=_X.c_b_plto/_X.c_b_lym)
            _X = _X.assign(COMBO_HLR=_X.c_b_hb/_X.c_b_lym)
            _X = _X.assign(COMBO_LMR=_X.c_b_lym/_X.c_b_mon)
            _X = _X.assign(COMBO_WRR=_X.c_b_wbc/_X.c_b_rbco)
            _X = _X.assign(COMBO_NWR=_X.c_b_neu/(1+_X.c_b_wbc-_X.c_b_neu))
            _X = _X.assign(COMBO_WPR=_X.c_b_wbc/_X.c_b_plto)
            _X = _X.assign(COMBO_PRR=_X.c_b_plto/_X.c_b_rbco)
            _X = _X.assign(COMBO_RHR=_X.c_b_retc/_X.c_b_hb)
            _X = _X.assign(COMBO_RIR=_X.c_b_rbco/(1+_X.c_b_retc+_X.c_b_irf))
            _X = _X.assign(COMBO_HHR=_X.c_b_ht/_X.c_b_hb)
            _X = _X.assign(COMBO_LSR=_X.c_b_limn/_X.c_b_seg)
            _X = _X.assign(COMBO_PMR=_X.c_b_plto/np.clip(_X.c_b_pmac,0.1,100))
            _X = _X.assign(COMBO_LPR=_X.c_b_limn/_X.c_b_pimn)
            _X = _X.assign(COMBO_MHR=_X.c_b_pmac/_X.c_b_hb)
            _X = _X.assign(COMBO_MMR=_X.c_b_pmac/_X.c_b_pmic)
            
            for col, func in base_transformations.items():
                try:
                    X[col] =X[col].apply(func)
                    self.combo_cols.append(col)
                except:
                    self.errors.append(('transformations', f'problem with {col}'))       

        if len(self.removal) > 0:
            _X.drop(self.removal, axis=1, inplace=True)
        if self.scaler is not None:
            _X.loc[:, self.combo_cols] = self.scaler(
                **self.scaler_kwargs
            ).fit_transform(_X[self.combo_cols])
        self.out_columns = _X.columns.tolist()
        return _X

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs
        return self
