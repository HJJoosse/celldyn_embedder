import os

os.chdir("T:/laupodteam/AIOS/Chontira/CellDynClustering")

import numpy as np
import pandas as pd
from data.load_data import *


## data loading
dm6_with_labels = pd.read_feather("L:/lab_research/RES-Folder-UPOD/Celldynclustering/E_ResearchData/2_ResearchData/embedded_celldyn_FULL_nn50_ndim6_w_labels_w_ratio_mf100.feather")
others_labels = dm6_with_labels.loc[:,dm6_with_labels.columns.isin(['sex', 'age','study_id', 'analysis_dt', 'sample_dt'])]
dm6 = dm6_with_labels.iloc[:, 0:6]

#dm6_with_labels_without_ratio = pd.read_feather("data/embedded_celldyn_ALL_nn50_ndim6_w_labels_.feather")
#dm6_without_ratio = dm6_with_labels_without_ratio.iloc[:, 0:6]

cell_dyn_with_labels = pd.read_feather("data/cell_dyn_gender_Full_100_encoded.feather")

cols_c_b_combo = [c for c in cell_dyn_with_labels.columns if ('c_b' in c) | ("COMBO" in c)]
cell_dyn = cell_dyn_with_labels.loc[:, cols_c_b_combo]

cluster_assignment_df = pd.read_csv("data/cluster_assignment.csv")
cluster_labels = cluster_assignment_df["cluster_assignment"].values



