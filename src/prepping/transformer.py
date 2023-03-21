"""
Sklearn api class for transforming the CELLDYN data
"""

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

from tqdm import tqdm

cut_offs = {
    "c_b_wbc": {"min": 0, "max": 6},
    "c_b_neu": {"min": 0, "max": 4},
    "c_b_seg": {"min": 0, "max": 4},
    "c_b_wvf": {"min": 0.75, "max": 1.0},
    "c_b_rtcfmn": {"min": 100, "max": 175},
    "c_b_rtcfcv": {"min": 0, "max": 25},
    "c_b_retc": {"min": 0, "max": 250},
    "c_b_rdw": {"min": 0, "max": 25},
    "c_b_rbco": {"min": 0, "max": 8},
    "c_b_rbcicv": {"min": 0, "max": 3},
    "c_b_rbci": {"min": 0, "max": 8},
    "c_b_rbcfmn": {"min": 60, "max": 110},
    "c_b_rbcfcv": {"min": 0, "max": 30},
    "c_b_prp": {"min": 0, "max": 20},
    "c_b_pretc": {"min": 0, "max": 15},
    "c_b_ppmn": {"min": 100, "max": 150},
    "c_b_ppcv": {"min": 0, "max": 30},
    "c_b_pmone": {"min": 0, "max": 40},
    "c_b_pmon": {"min": 0, "max": 40},
    "c_b_pmic": {"min": 0, "max": 15},
    "c_b_pmac": {"min": 0, "max": 25},
    "c_b_plto": {"min": 0, "max": 800},
    "c_b_plti": {"min": 0, "max": 800},
    "c_b_pimn": {"min": 100, "max": 190},
    "c_b_picv": {"min": 10, "max": 25},
    "c_b_phpr": {"min": 0, "max": 2},
    "c_b_phpo": {"min": 0, "max": 50},
    "c_b_peos": {"min": 0, "max": 20},
    "c_b_pdw": {"min": 10, "max": 25},
    "c_b_pct": {"min": 0, "max": 1},
    "c_b_pbas": {"min": 0, "max": 3},
    "c_b_npcv": {"min": 0, "max": 15},
    "c_b_nimn": {"min": 100, "max": 175},
    "c_b_nicv": {"min": 0, "max": 8},
    "c_b_nfmn": {"min": 60, "max": 100},
    "c_b_ndmn": {"min": 0, "max": 50},
    "c_b_ndcv": {"min": 0, "max": 30},
    "c_b_nacv": {"min": 0, "max": 6},
    "c_b_mpv": {"min": 0, "max": 20},
    "c_b_mone": {"min": 0, "max": 5},
    "c_b_mon": {"min": 0, "max": 5},
    "c_b_mcvr": {"min": 60, "max": 160},
    "c_b_mcv": {"min": 40, "max": 140},
    "c_b_mchr": {"min": 0, "max": 40},
    "c_b_mchc_usa": {"min": 25, "max": 40},
    "c_b_mchc": {"min": 0.15, "max": 0.3},
    "c_b_mch_usa": {"min": 0, "max": 50},
    "c_b_mch": {"min": 0, "max": 3},
    "c_b_lyme": {"min": 0, "max": 7},
    "c_b_lym": {"min": 0, "max": 7},
    "c_b_licv": {"min": 0, "max": 10},
    "c_b_lamn": {"min": 60, "max": 140},
    "c_b_lacv": {"min": 0, "max": 12},
    "c_b_ht": {"min": 0, "max": 80},
    "c_b_hgb_usa": {"min": 0, "max": 25},
    "c_b_hdw": {"min": 0, "max": 20},
    "c_b_hb": {"min": 0, "max": 15},
    "c_b_eos": {"min": 0, "max": 2},
    "c_b_bas": {"min": 0, "max": 0.4}
}

ord_scale_cols = [
    "c_b_bnd",
    "c_b_ig",
    "c_b_vlym",
    "c_b_blst",
    "c_b_nrbc",
    "c_b_vlym",
    "c_b_pblst",
]


class CellDynTrans(BaseEstimator, TransformerMixin):
    """
    Class for transforming the CELLDYN data
    """

    def __init__(
        self,
        scaler: str = "standard",
        log_scale: list = [],
        ord_scale: list = [],
        **kwargs,
    ):
        """
        Initialize the class
        """

        self.scaler = scaler
        self.kwargs = kwargs
        assert (log_scale, list)
        self.log_scale = log_scale
        self.cut_offs = cut_offs
        if all([type(e) == str for e in ord_scale]):
            self.ord_cols = ord_scale
        else:
            self.ord_cols = ord_scale_cols

    def get_category_bin_dict(
        self,
        df: pd.DataFrame,
        num_quantiles: int = 3,
        min_num_nonzero: int = 5000,
        min_val: float = 0.0,
    ) -> dict:

        value_cols = self.ord_cols

        category_bin_dict = {}
        if num_quantiles % 2 == 0:
            num_quantiles += 1

        self.quantile_list = []
        for c in range(num_quantiles + 1):
            if c > 0:
                self.quantile_list.append(c * 1.0 / (num_quantiles + 1))

        ord_cols = []
        nonord_cols = []
        for c in value_cols:
            if (df[c] > min_val).sum() > min_num_nonzero:
                quants = (
                    df[df[c] > min_val][c]
                    .quantile(self.quantile_list, interpolation="higher")
                    .round(3)
                )
                category_bin_dict[c] = dict(
                    zip(
                        ["zero"]
                        + [f'q_{str(c).replace("0.","")}' for c in self.quantile_list],
                        [min_val] + list(quants),
                    )
                )
                ord_cols.append(c)
            else:
                nonord_cols.append(c)

        return category_bin_dict, num_quantiles, ord_cols, nonord_cols

    def assign_category_bins(
        self, df: pd.DataFrame, num_quantiles: int = 3, min_val: float = 0.0
    ) -> pd.DataFrame:

        (
            category_bins,
            num_quantiles,
            ord_cols,
            nonord_cols,
        ) = self.get_category_bin_dict(
            df=df, num_quantiles=num_quantiles, min_val=min_val
        )

        for column, qbins in category_bins.items():
            cname = f"ORD_{column}"
            df[cname] = 0
            for k, (q, qbin) in enumerate(qbins.items()):
                if q == "zero":
                    df.loc[df[column] <= qbin, cname] = 0
                    df.loc[df[column] > qbin, cname] = 1 / (num_quantiles + 1)
                else:
                    df.loc[df[column] >= qbin, cname] = (k + 1) / (num_quantiles + 1)
        try:
            if self.kwargs["remove_original_columns"]:
                df.drop(columns=ord_cols + nonord_cols, inplace=True)
        except:
            pass

        return df

    @staticmethod
    def _log_scaler(vec: pd.Series):
        return np.log(vec + 1)

    @staticmethod
    def _map_to_functions(map_dict: dict) -> dict:
        return {
            k: lambda x: np.maximum(v["min"], np.minimum(x, v["max"]))
            for k, v in map_dict.items()
        }

    def _apply_transformers(self, vec: pd.Series):
        try:
            return FunctionTransformer(
                lambda x: np.maximum(
                    self.cut_offs[vec.name]["min"],
                    np.minimum(x, self.cut_offs[vec.name]["max"]),
                )
            ).fit_transform(vec.values)
        except KeyError:
            return vec

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_transformed = X.copy()
        for var in self.log_scale:
            X_transformed[var] = self._log_scaler(X_transformed[var])
        X_transformed = X_transformed.apply(
            lambda x: self._apply_transformers(x), axis=0
        )
        if "c_b_wvf" in X_transformed.columns:
            X_transformed["c_b_wvf"] = np.arcsin(X_transformed["c_b_wvf"])

        if isinstance(self.ord_cols, list):
            X_transformed = self.assign_category_bins(X_transformed)
        return X_transformed

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs
        return self
