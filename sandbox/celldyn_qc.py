import pandas as pd
import numpy as np
from tqdm import tqdm


fail_mapping = {
    'faillacv': ['c_b_nacv'],
    'failimn': ['c_b_nimn', 'c_b_Limn', 'c_b_Pimn', 'c_b_rbcimn'],
    'failicv': ['c_b_nicv', 'c_b_Picv', 'c_b_rbcicv'],
    'failfmn': ['c_b_nfmn', 'c_b_rbcfmn', 'c_b_rtcfmn'],
    'failHPR': ['c_b_pHPR'],
    'failHPO': ['c_b_pHPO'],
    'failpreti': ['c_b_pretc'],
    'failreti': ['c_b_retc'],
    'failfcv' : ['c_b_nfcv','c_b_rbcfcv','c_b_rtcfcv'],
    'failirf': ['c_b_irf'],
    'failMCHCr': ['c_b_MCHCr'],
    'failHDW': ['c_b_HDW'],
    'failMCHr': ['c_b_MCHr'],
    'failMCVr': ['c_b_MCVr'],
    'failnacv': ['c_b_nacv'],
    'failnicv': ['c_b_nicv'],
    'failndcv': ['c_b_ndcv'],
    'failnfcv': ['c_b_nfcv'],
    'failnpcv': ['c_b_npcv'],
    'faillicv': ['c_b_Licv'],
    'faillacv': ['c_b_Lacv']
}


class QcControl:
# cast in sklearn api

    def __init__(self, param_file=None, reference_file=None):
        if (param_file is not None) and (reference_file is not None):
            self.param_file = param_file
            self.reference_file = reference_file
            self._parse_filter()
    
    def _get_cols(self, df):
        self.count_columns = [c for c in df.columns if 'c_cnt' in c]
        self.meas_columns = [c for c in df.columns if 'c_b' in c]+['PLT']
        self.mode_columns = [c for c in df.columns if 'c_m' in c]
        self.susp_columns = [c for c in df.columns if 'c_s' in c]
        self.alert_columns = [c for c in df.columns if 'Alrt' in c]
        self.fail_columns = [c for c in df.columns if 'fail' in c]
        self.meas_names = ["_".join(c.split("_")[2:]) for c in self.meas_columns]

    def _parse_filter(self):
        
        read_filters = pd.read_excel(self.param_file, sheet_name="Parameters", skiprows=1)
        read_filters.rename(columns={'Unnamed: 0': 'measurement_name', 
                    'Unnamed: 1': 'measurement_description'}, inplace=True)
        read_filters['measurement_name'] = read_filters.measurement_name.str.split('_')\
                                                .apply(lambda x: "_".join(x[2:]))
        self.filter_dict = read_filters[['measurement_name', 'Intra', 'Inter', 'Min', 'Max']]\
                                                .set_index('measurement_name').to_dict('index')

        read_filter_man =  pd.read_excel(self.reference_file)
        read_filter_man['measurement_name'] = read_filter_man.Value.str.split('_')\
                                                .apply(lambda x: "_".join(x[2:]))
        self.filter_dict_man = read_filter_man[['measurement_name', 'min', 'max']]\
                                            .set_index('measurement_name').to_dict('index')

        for k,v in self.filter_dict.items():
            try:
                man_min, man_max  = self.filter_dict_man[k]['min'], \
                                    self.filter_dict_man[k]['max']
                if np.isnan(man_min) == False:
                    self.filter_dict[k]['Min'] = man_min
                if np.isnan(man_max) == False:
                    self.filter_dict[k]['Max'] = man_max
            except Exception as  e:
                print(e,k,v)
                pass        

    @staticmethod
    def qc_leuko(df: pd.DataFrame):
        temp_cols =[c.lower() for c in df]
        real_cols = [c for c in df]
        df.columns = temp_cols
        for chan in ['na', 'ni', 'nd', 'nf', 'np', 'li', 'la']:
            try:
                df.loc[df[f"c_b_{chan}cv"] < 0.00000000000001, 
                [f"c_b_{chan}cv",f"c_b_{chan}mn"]] = np.nan
            except KeyError:
                print(f"{chan}cv or {chan}mn not in data")
        df.columns = real_cols
        return df

    @staticmethod
    def qc_rbc(df: pd.DataFrame):
        temp_cols =[c.lower() for c in df]
        real_cols = [c for c in df]
        df.columns = temp_cols
        df.loc[lambda x: (x.c_mode_rtc == 0) | (x.c_b_retc < 0.0001) |
                              (x.c_b_pretc < 0.0001) | (x.c_b_irf < 0.0001),
            ["c_b_retc","c_b_pretc","c_b_irf","c_b_rbcimn",
             "c_b_rbcicv","c_b_rbcfmn","c_b_rbcfcv",
             "c_b_mchcr","c_b_hdw","c_b_mchr","c_b_mcvr",
             "c_b_phpo","c_b_phpr"]] = np.nan

        df.loc[lambda x: (x.c_b_rbcicv < 0.0001) | (x.c_b_rbcimn < 0.0001) |
                              (x.c_b_rbcfmn <0.0001) | (x.c_b_rbcfcv < 0.0001),
                    ["c_b_rbcfmn","c_b_rbcfcv","c_b_rbcimn","c_b_rbcicv"]] = np.nan

        df.loc[lambda x: x.c_b_hdw <0.0001,["c_b_phpr","c_b_hdw","c_b_phpo"]] = np.nan

        df.loc[lambda x: (x.c_b_mchr < 0.0001)|(x.c_b_mchcr < 0.0001)|(x.c_b_mcvr < 0.0001)|
                         (x.c_b_phpo < 0.000000000000000000000000000001)|
                         (x.c_b_phpr < 0.000000000000000000000000000001),
               ["c_b_mchcr","c_b_hdw","c_b_mchr",
                "c_b_mcvr","c_b_phpo","c_b_phpr"]] = np.nan
        df.columns = real_cols

        return df

    @staticmethod
    def qc_standard_values(df: pd.DataFrame):
        temp_cols =[c.lower() for c in df]
        real_cols = [c for c in df]
        df.columns = temp_cols
        standard_vals = {"c_b_rbcimn":182,"c_b_rbcicv":1.59341,
                         "c_b_rbcfmn":85,"c_b_rbcfcv":7.2,
                         "c_b_namn":140, "c_b_nimn":150, 
                         "c_b_npmn":125, "c_b_ndmn":28, "c_b_nfmn":69,
                         "c_b_lamn":100, "c_b_limn":75, "c_b_hb":6.206e-21, 
                         "c_b_mch":0.6206, "c_b_mchc":0.6202}
        for key, val in standard_vals.items():
            try:
                df = df.loc[lambda x: x[key] != val]
            except KeyError:
                print(key, " not in columns")
        df.columns = real_cols

        return df.reset_index(drop=True)

    def suspect_flag_filter(self, df):
    # if suspect flag is 2, c_b should be set to np.nan
    # if suspect flag is >2, c_b should be check against the bounds of the filter
    # if suspect flag is 1, c_b should be left untouched
        def _flag_val_check(val,flag,_min,_max):
            if flag==1:
                return val
            if flag==2:
                return np.nan
            if (val>=_min) & (val<=_max):
                return val
            return np.nan

        for meas_name in tqdm(self.meas_names):
            meas_val = "c_b_"+meas_name
            meas_flag = "c_s_"+meas_name

            try:
                filters = filter_dict[meas_name]
                min_val = float(filters['Min'])
                max_val = float(filters['Max'])

                if (np.isnan(min_val)) | (np.isnan(max_val)):
                    pass
                else:
                    df.loc[:, meas_val] = df[[meas_val, meas_flag]]\
                                    .apply(lambda  x:
                                    _flag_val_check(x[0],x[1], min_val, max_val),
                                    axis=1)

            except Exception as e:
                pass
        return df

    def fail_filter(self, df):
        for fail_name in tqdm(self.fail_columns):
            try:
                for meas_col in fail_mapping[fail_name]:
                    df.loc[:,meas_col] = \
                        df[[meas_col, fail_name]]\
                        .apply(lambda x: x[0] if x[1]==0 else np.nan, axis=1)
            except Exception as e:
                print(e,fail_name)
                pass
        return df