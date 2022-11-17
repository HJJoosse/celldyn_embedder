import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import hdbscan
from evaluation.fast_dunn_index import dunn_fast
from data.load_data import *
import ast
from graphic_stuffs import *
import pickle
from evaluation.util import *





## Evaluators

evaluators= {'silhouette_score': silhouette_score,
            'davies_bouldin_score': davies_bouldin_score,
            'dunn_fast': dunn_fast}

db= {'davies_bouldin_score': davies_bouldin_score}

## data loading
dm6_with_labels = pd.read_feather("data/embedded_celldyn_ALL_nn50_ndim6_w_labels_.feather")
others_labels = dm6_with_labels.loc[:,dm6_with_labels.columns.isin(['sex', 'age','study_id', 'analysis_dt', 'sample_dt'])]
dm6 = load_data().set_df(dm6_with_labels.iloc[:, 0:6]).get_x_from_df(contain_y=False)
