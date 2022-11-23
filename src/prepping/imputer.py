"""
A class to thinly wrap different imputers in the sklearn api

# Impute

* [FCS, Fancy impute](https://pypi.org/project/fancyimpute/)
* [FCS, Auto impute](https://pypi.org/project/autoimpute/)
* [FCS, Skearn](https://scikit-learn.org/stable/modules/impute.html)
* [FCS, Statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.imputation.mice.MICE.html)
* [FCS, Miceforest](https://miceforest.readthedocs.io/en/latest/)

*  normalisatie --> KNN impute (hoeveel neighbors, wat voor distance..)
* https://arxiv.org/pdf/0805.4471.pdf: in fancyimpute; NuclearNormMinimization
* Iterative imputer met PMM : normalisation -> sklearn voor iterative imputer, autoimpute.imputations.series.pmm


"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# add an output logger that can be used to log the imputation process

import logging

import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")


from miceforest import ImputationKernel
from miceforest import mean_match_default, mean_match_fast_cat, mean_match_shap

backend_compat = {
    "miceforest": {"imputer": ["forest"]},
    "fancyimpute": {
        "imputer": ["knn", "softimpute", "svd", "biscaler", "mf", "nnm", "simple"],
        "clf": ["mean", "median", "mode"],
    },
    "autoimpute": {
        "imputer": ["mice", "multi", "simple"],
        "clf": [
            "pmm",
            "lrd",
            "bayesian_binary_logistic",
            "bayesian_least_squares",
            "least_squares",
            "multinomial_logistic",
            "mean",
            "median",
            "binary_logistic",
        ],
    },
    "sklearn": {
        "imputer": ["iterative", "simple", "knn"],
        "clf": [
            "rf",
            "hgb",
            "lr",
            "svr",
            "svc",
            "bayesian_ridge",
            "xgb",
            "lgbm",
            "mean",
            "median",
        ],
    },
}


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        imputer: str = "forest",
        backend: str = "miceforest",
        add_mia: bool = False,
        remove_working_data: bool = False,
        synthesize_working_data: bool = False,
        meas_cols: list = [],
        **kwargs,
    ):
        """
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
        """
        self.imputer = imputer
        self.backend = backend
        self.meas_cols = meas_cols
        self.add_mia = add_mia
        self.synthesize_working_data = synthesize_working_data
        self.remove_working_data = remove_working_data
        self.kwargs = kwargs

        if backend == "miceforest":
            try:
                self.num_estimators = self.kwargs["num_estimators"]
            except KeyError:
                self.num_estimators = 100

            try:
                self.iterations = self.kwargs["iterations"]
            except KeyError:
                self.iterations = 10

            try:
                num_match_candidates = self.kwargs["num_match_candidates"]
            except KeyError:
                num_match_candidates = 5

            self.mean_match = mean_match_default
            self.mean_match.set_mean_match_candidates(num_match_candidates)

            try:
                self.save_all_iterations = self.kwargs["save_all_iterations"]
            except KeyError:
                self.save_all_iterations = False

            try:
                self.data_subset = self.kwargs["data_subset"]
            except KeyError:
                self.data_subset = None

        # elif backend == 'fancyimpute':
        # from fancyimpute import KNN, SoftImpute, IterativeSVD, BiScaler
        # from fancyimpute import NuclearNormMinimization, MatrixFactorization
        # elif backend == 'autoimpute':
        # from autoimpute.imputations import SingleImputer, MultipleImputer
        # elif backend == 'sklearn':
        # from sklearn.experimental import enable_iterative_imputer
        # from sklearn.impute import IterativeImputer
        # from sklearn.impute import KNNImputer
        #    self.imputer = IterativeImputer(**kwargs)

    def _start_logger(self):
        """
        Start the logger.
        """
        real_local_directory = os.path.dirname(os.path.realpath(__file__))
        file_handler = logging.FileHandler("imputer.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    def _add_noise(self, X, perc=0.05):
        """
        Add noise to the data to make it more realistic.
        """
        noise = np.random.uniform(0, perc, X.shape)
        return X + np.multiply(X, noise)
    
    def _add_mask(self, X, perc=0.25):
        """
        Add mask to the data to make it more realistic.
        """
        X.ravel()[np.random.choice(X.size, int(X.size * perc), replace=False)] = np.nan
        return X

    def fit(self, X: pd.DataFrame, y=None):
        if self.backend == "miceforest":
            self.imputer = ImputationKernel(
                X[self.meas_cols],
                copy_data=False,
                random_state=1283,
                mean_match_scheme=self.mean_match,
                save_all_iterations=self.save_all_iterations,
                data_subset=self.data_subset,
                train_nonmissing=self.synthesize_working_data
            )
            self.imputer.mice(
                iterations=self.iterations,
                verbose=False,
                n_estimators=self.num_estimators,
            )
            self.imputer.compile_candidate_preds()
            if self.remove_working_data:
                # will probaby fail because exemplars are needed for inference
                self.imputer.working_data = np.ones((1, 1))
            if self.synthesize_working_data:
                # random distortion of the working data
                distorted = self._add_noise(X[self.meas_cols], perc=0.01)
                # random mask of the working data
                distorted = self._add_mask(distorted, perc=0.3)
                # imputation of the masked/distorted working data
                self.imputer.working_data  = self.imputer.impute_new_data(
                                new_data=distorted, copy_data=False, datasets=[0]
                            ).complete_data(0, inplace=False)
                    
        return self

    def transform(self, X: pd.DataFrame = None):
        self._start_logger()

        if self.backend == "miceforest":
            # self.X[self.meas_cols]
            # imputed_df = self.imputer.complete_data(variables=self.meas_cols)
            logger.debug("Imputing with miceforest")
            logger.debug(f"Input type:{type(X)}")
            imputed_df = self.imputer.impute_new_data(
                new_data=X[self.meas_cols], copy_data=False, datasets=[0]
            ).complete_data(0, inplace=False)

        X_out = X.copy()
        X_out.loc[:, self.meas_cols] = imputed_df

        if self.add_mia:
            mia_df = X.isna().astype(int)
            mia_df.columns = [f"MISSING_{col}" for col in mia_df.columns]
            X_out = pd.concat([X_out, mia_df], axis=1)

        return X_out

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs
        return self
