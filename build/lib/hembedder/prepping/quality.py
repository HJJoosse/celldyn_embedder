# import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# import polars as pl
# import modin.pandas as pd
import pandas as pd

"""
I want to add a logger to this script.
"""
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")


fail_mapping = {
    "faillacv": ["c_b_nacv"],
    "failimn": ["c_b_nimn", "c_b_limn", "c_b_pimn", "c_b_rbcimn"],
    "failicv": ["c_b_nicv", "c_b_picv", "c_b_rbcicv"],
    "failfmn": ["c_b_nfmn", "c_b_rbcfmn", "c_b_rtcfmn"],
    "failhpr": ["c_b_phpr"],
    "failhpo": ["c_b_phpo"],
    "failpreti": ["c_b_pretc"],
    "failreti": ["c_b_retc"],
    "failfcv": ["c_b_nfcv", "c_b_rbcfcv", "c_b_rtcfcv"],
    "failirf": ["c_b_irf"],
    "failmchcr": ["c_b_mchcr"],
    "failhdw": ["c_b_hdw"],
    "failmchr": ["c_b_mchr"],
    "failmcvr": ["c_b_mcvr"],
    "failnacv": ["c_b_nacv"],
    "failnicv": ["c_b_nicv"],
    "failndcv": ["c_b_ndcv"],
    "failnfcv": ["c_b_nfcv"],
    "failnpcv": ["c_b_npcv"],
    "faillicv": ["c_b_licv"],
    "faillacv": ["c_b_lacv"],
}


class QcControl(BaseEstimator, TransformerMixin):
    # cast in sklearn api

    def __init__(self, param_file=None, filters=[], backend="pandas"):
        """
        Initialize the QC control object.

        Parameters
        ----------
        param_file : str -- path to the parameter file with hard bounds
        filters : list of str -- list of filters to apply to the data
        backend : str -- pandas, polars
        """

        if param_file is not None:
            self.param_file = param_file
            self._parse_filter()

        self.backend = backend

        assert (filters, list), "filters must be a list"
        self._parse_filter_list(filters)

    # @staticmethod
    # def _convert_to_polars(df: pd.DataFrame):
    #    '''
    #    Convert the dataframe to polars.
    #    '''
    #    return pl.from_pandas(df)

    def _start_logger(self):
        """
        Start the logger.
        """
        real_local_directory = os.path.dirname(os.path.realpath(__file__))
        file_handler = logging.FileHandler("celldyn_qc.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    def _parse_filter_list(self, filters):
        self.filters = dict()
        if len(filters) == 0:
            self.filters = {
                "leuko": self.qc_leuko,
                "rbc": self.qc_rbc,
                "ranges": self.qc_plausible_range_filter,
                "suspect": self.suspect_flag_filter,
            }
        else:
            for _filter in filters:
                if "leuko" in _filter:
                    self.filters["leuko"] = self.qc_leuko
                if "rbc" in _filter:
                    self.filters["rbc"] = self.qc_rbc
                if "ranges" in _filter:
                    self.filters["ranges"] = self.qc_plausible_range_filter
                if "suspect" in _filter:
                    self.filters["suspect"] = self.suspect_flag_filter
                if "fail" in _filter:
                    self.filters["fail"] = self.fail_filter
                if "standard" in _filter:
                    self.filters["standard"] = self.qc_standard_values

    def _get_cols(self, df):
        self.count_columns = [c.lower() for c in df.columns if "c_cnt" in c]
        self.meas_columns = [c.lower() for c in df.columns if "c_b" in c] + ["PLT"]
        self.mode_columns = [c.lower() for c in df.columns if "c_m" in c]
        self.susp_columns = [c.lower() for c in df.columns if "c_s" in c]
        self.alert_columns = [c.lower() for c in df.columns if "Alrt" in c]
        self.fail_columns = [c.lower() for c in df.columns if "fail" in c]
        self.meas_names = ["_".join(c.split("_")[2:]) for c in self.meas_columns]

    def _parse_filter(self):
        try:
            read_filters = pd.read_excel(
                self.param_file
            )  # , sep=";", encoding="latin1")
        except Exception as e:
            print("Could not read the parameter file: {}".format(e))
            raise e

        assert all(
            [
                c in read_filters.columns
                for c in [
                    "measurement_name",
                    "measurement_description",
                    "Intra",
                    "Inter",
                    "Min",
                    "Max",
                    "Eenheid",
                ]
            ]
        ), "The parameter file is not in the correct format. It should contain: \
                    measurement_name,\
                    measurement_description,\
                    Intra,\
                    Inter,\
                    Min,\
                    Max,\
                    Eenheid"

        read_filters["measurement_name"] = read_filters.measurement_name.str.split(
            "_"
        ).apply(lambda x: "_".join(x[2:]))

        self.filter_dict = (
            read_filters[["measurement_name", "Intra", "Inter", "Min", "Max"]]
            .set_index("measurement_name")
            .to_dict("index")
        )

        """  read_filter_man =  pd.read_excel(self.reference_file)
        read_filter_man['measurement_name'] = read_filter_man.Value.str.split('_')\
                                                .apply(lambda x: "_".join(x[2:]))
        self.filter_dict_man = read_filter_man[['measurement_name', 'min', 'max']]\
                                            .set_index('measurement_name').to_dict('index')

        for k, v in self.filter_dict.items():
            try:
                man_min, man_max  = self.filter_dict_man[k]['min'], \
                                    self.filter_dict_man[k]['max']
                if np.isnan(man_min) == False:
                    self.filter_dict[k]['Min'] = man_min
                if np.isnan(man_max) == False:
                    self.filter_dict[k]['Max'] = man_max
            except Exception as  e:
                logger.debug(f'Exception:{e}, key:{k}')
        """

    def qc_leuko(self, df: pd.DataFrame):
        temp_cols = [c.lower() for c in df]
        real_cols = [c for c in df]
        df.columns = temp_cols
        for chan in ["na", "ni", "nd", "nf", "np", "li", "la"]:
            try:
                df.loc[
                    df[f"c_b_{chan}cv"] < 0.00000000000001,
                    [f"c_b_{chan}cv", f"c_b_{chan}mn"],
                ] = np.nan
            except KeyError:
                logger.debug(f"{chan}cv or {chan}mn not in data")
        df.columns = real_cols
        return df

    def qc_rbc(self, df: pd.DataFrame):
        def _replacer(x, conds, change_cols):
            for k, t in conds.items():
                try:
                    tf = lambda x: eval(f"x[{k}]{t[0]}{str(t[1])}")
                    df.loc[tf, change_cols] = t[2]
                except:
                    logger.info(f"The {k} variable was not present in CELLDYN")
            return x

        temp_cols = [c.lower() for c in df]
        real_cols = [c for c in df]
        df.columns = temp_cols

        change_cols = [
            "c_b_retc",
            "c_b_pretc",
            "c_b_irf",
            "c_b_rbcimn",
            "c_b_rbcicv",
            "c_b_rbcfmn",
            "c_b_rbcfcv",
            "c_b_mchcr",
            "c_b_hdw",
            "c_b_mchr",
            "c_b_mcvr",
            "c_b_phpo",
            "c_b_phpr",
        ]
        change_cols = list(set(change_cols).intersection(set(temp_cols)))

        conds = {
            "c_mode_rtc": ("==", 0, np.nan),
            "c_b_retc": ("<", 0.0001, np.nan),
            "c_b_pretc": ("<", 0.0001, np.nan),
            "c_b_irf": ("<", 0.0001, np.nan),
        }
        df = _replacer(df, conds, change_cols)

        ##################################################################
        ##################################################################

        change_cols = ["c_b_rbcfmn", "c_b_rbcfcv", "c_b_rbcimn", "c_b_rbcicv"]
        change_cols = list(set(change_cols).intersection(set(temp_cols)))

        conds = {
            "c_b_rbcicv": ("<", 0.0001, np.nan),
            "c_b_rbcimn": ("<", 0.0001, np.nan),
            "c_b_rbcfmn": ("<", 0.0001, np.nan),
            "c_b_rbcfcv": ("<", 0.0001, np.nan),
        }
        df = _replacer(df, conds, change_cols)

        ##################################################################
        ##################################################################

        change_cols = ["c_b_phpr", "c_b_hdw", "c_b_phpo"]
        change_cols = list(set(change_cols).intersection(set(temp_cols)))

        df.loc[lambda x: x.c_b_hdw < 0.0001, change_cols] = np.nan

        change_cols = [
            "c_b_mchcr",
            "c_b_hdw",
            "c_b_mchr",
            "c_b_mcvr",
            "c_b_phpo",
            "c_b_phpr",
        ]

        conds = {
            "c_b_mchr": ("<", 0.0001, np.nan),
            "c_b_mchcr": ("<", 0.0001, np.nan),
            "c_b_mcvr": ("<", 0.0001, np.nan),
            "c_b_phpo": ("<", 0.000000000000000000000000000001, np.nan),
            "c_b_phpr": ("<", 0.000000000000000000000000000001, np.nan),
        }
        df = _replacer(df, conds, change_cols)

        df.columns = real_cols
        return df

    def qc_standard_values(self, df: pd.DataFrame):
        temp_cols = [c.lower() for c in df]
        real_cols = [c for c in df]
        df.columns = temp_cols
        standard_vals = {
            "c_b_rbcimn": 182,
            "c_b_rbcicv": 1.59341,
            "c_b_rbcfmn": 85,
            "c_b_rbcfcv": 7.2,
            "c_b_namn": 140,
            "c_b_nimn": 150,
            "c_b_npmn": 125,
            "c_b_ndmn": 28,
            "c_b_nfmn": 69,
            "c_b_lamn": 100,
            "c_b_limn": 75,
            "c_b_hb": 6.206e-21,
            "c_b_mch": 0.6206,
            "c_b_mchc": 0.6202,
        }
        for key, val in standard_vals.items():
            try:
                df = df.loc[lambda x: x[key] != val]
            except KeyError:
                logger.debug(key, " not in columns")
        df.columns = real_cols

        return df

    def qc_plausible_range_filter(self, df: pd.DataFrame):
        cut_offs = pd.read_csv(self.param_file, sep=";", encoding="latin1")
        col_names = cut_offs.columns
        cut_offs = pd.DataFrame(
            np.where(cut_offs == "-", np.nan, cut_offs), columns=col_names
        )
        for c in self.meas_columns:
            try:
                min_val = float(
                    cut_offs.loc[cut_offs.measurement_name == c, "Min"].iloc[0]
                )
                max_val = float(
                    cut_offs.loc[cut_offs.measurement_name == c, "Max"].iloc[0]
                )
                if not pd.isna(min_val):
                    df.loc[lambda x: (x[c] > max_val) & (x[c] < min_val), c] = np.nan
            except IndexError as e:
                logger.debug(e)
        return df

    def suspect_flag_filter(self, df: pd.DataFrame):
        # if suspect flag is 2, c_b should be set to np.nan
        # if suspect flag is >2, c_b should be check against the bounds of the filter
        # if suspect flag is 1, c_b should be left untouched
        def _flag_val_check(val, flag, _min, _max):
            if flag == 1:
                return val
            if flag == 2:
                return np.nan
            if (val >= _min) & (val <= _max):
                return val
            return np.nan

        for meas_name in self.meas_names:
            meas_val = "c_b_" + meas_name
            meas_flag = "c_s_" + meas_name

            try:
                filters = self.filter_dict[meas_name]
                min_val = float(filters["Min"])
                max_val = float(filters["Max"])

                if (np.isnan(min_val)) | (np.isnan(max_val)):
                    pass
                else:
                    df.loc[:, meas_val] = df[[meas_val, meas_flag]].apply(
                        lambda x: _flag_val_check(x[0], x[1], min_val, max_val), axis=1
                    )

            except Exception as e:
                pass
        return df

    def fail_filter(self, df: pd.DataFrame):
        for fail_name in self.fail_columns:
            logger.debug(f"Processing {fail_name}")
            try:
                for meas_col in fail_mapping[fail_name]:
                    df.loc[:, meas_col] = df[[meas_col, fail_name]].apply(
                        lambda x: x[0] if x[1] == 0 else np.nan, axis=1
                    )
            except Exception as e:
                logger.debug(f"Exception: {e}, for {fail_name}")
        return df

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        self._start_logger()

        X.columns = [c.lower() for c in X.columns]
        self._get_cols(X)
        _X = X.copy()

        for k, filter_fun in self.filters.items():
            logger.debug(f"Start filtering: {k}")
            _X = filter_fun(_X.copy())
            logger.debug(f"Completed filtering: {k}")

        return _X

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs
        return self
