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

cut_offs = {
    "c_b_wbc" :  {'min':0,'max':6},
    "c_b_neu" :  {'min':0,'max':4},
    "c_b_seg" :  {'min':0,'max':4},
    "c_b_rtcfmn" : {'min':100,'max':175},
    "c_b_rtcfcv" : {'min':0,'max':25},
    "c_b_retc": {'min':0,'max':250},
    "c_b_rdw": {'min':0,'max':25},
    "c_b_rbco": {'min':0,'max':8},
    "c_b_rbcicv": {'min':0,'max':3},
    "c_b_rbci": {'min':0,'max':8},
    "c_b_rbcfmn": {'min':60,'max':110},
    "c_b_rbcfcv": {'min':0,'max':30},
    "c_b_prP": {'min':0,'max':20},
    "c_b_pretc": {'min':0,'max':15},
    "c_b_Ppmn": {'min':100,'max':150},
    "c_b_Ppcv": {'min':0,'max':30},
    "c_b_pmone": {'min':0,'max':40},
    "c_b_pmon": {'min':0,'max':40},
    "c_b_pMIC": {'min':0,'max':15},
    "c_b_pMAC": {'min':0,'max':25},
    "c_b_plto": {'min':0,'max':800},
    "c_b_plti": {'min':0,'max':800},
    "c_b_Pimn": {'min':100,'max':190},
    "c_b_Picv": {'min':10,'max':25},
    "c_b_pHPR": {'min':0,'max':2},
    "c_b_pHPO": {'min':0,'max':50},
    "c_b_peos": {'min':0,'max':20},
    "c_b_pdw": {'min':10,'max':25},
    "c_b_pct": {'min':0,'max':1},
    "c_b_pbas": {'min':0,'max':3},
    "c_b_npcv": {'min':0,'max':15},
    "c_b_nimn": {'min':100,'max':175},
    "c_b_nicv": {'min':0,'max':8},
    "c_b_nfmn": {'min':60,'max':100},
    "c_b_ndmn": {'min':0,'max':50},
    "c_b_ndcv": {'min':0,'max':30},
    "c_b_nacv": {'min':0,'max':6},
    "c_b_mpv": {'min':0,'max':20},
    "c_b_mone": {'min':0,'max':5},
    "c_b_mon": {'min':0,'max':5},
    "c_b_MCVr": {'min':60,'max':160},
    "c_b_mcv": {'min':40,'max':140},
    "c_b_MCHr": {'min':0,'max':40},
    "c_b_mchc_usa": {'min':25,'max':40},
    "c_b_mchc": {'min':0.15, 'max':0.3},
    "c_b_mch_Usa": {'min':0,'max':50},
    "c_b_mch": {'min':0,'max':3},
    "c_b_lyme": {'min':0,'max':7},
    "c_b_lym": {'min':0,'max':7},
    "c_b_Licv":  {'min':0,'max':10},
    "c_b_Lamn": {'min':60,'max':140},
    "c_b_Lacv": {'min':0,'max':12},
    "c_b_ht": {'min':0,'max':80},
    "c_b_hgb_usa": {'min':0,'max':25},
    "c_b_HDW": {'min':0,'max':20},
    "c_b_hb": {'min':0,'max':15},
    "c_b_eos": {'min':0,'max':2},
    "c_b_bas": {'min':0, 'max':0.4}
}


class CellDynTrans(BaseEstimator, TransformerMixin):
    '''
    Class for transforming the CELLDYN data
    '''
    def __init__(self, scaler='standard', log_scale: list = [], **kwargs):
        '''
        Initialize the class
        '''
        self.scaler = scaler
        self.kwargs = kwargs
        assert(log_scale,list)
        self.log_scale = log_scale
        self.cut_offs = cut_offs
              
    @staticmethod
    def _log_scaler(vec:pd.Series):       
        return np.log(vec+1)

    @staticmethod
    def _map_to_functions(map_dict:dict) -> dict:
        return {
            k:lambda x: \
                np.maximum(v['min'], np.minimum(x, v['max'])) \
                    for k,v in map_dict.items()
                    }

    def _apply_transformers(self,vec:pd.Series):
        try:
            return FunctionTransformer(
                lambda x:np.maximum(self.cut_offs[vec.name]['min'],\
                    np.minimum(x, self.cut_offs[vec.name]['max'])))\
                    .fit_transform(vec.values)
        except KeyError:
            return vec

    def fit(self, X: pd.DataFrame, y=None):
        X_transformed = X.copy()
        for var in self.log_scale:
            X_transformed[var] = self._log_scaler(X_transformed[var])
        self.X_transformed = X_transformed.apply(lambda x:self._apply_transformers(x),axis = 0)
        return self

    def transform(self, X=None, y=None):
        return self.X_transformed