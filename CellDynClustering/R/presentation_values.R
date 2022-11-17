setwd("T:/laupodteam/AIOS/Chontira/CellDynClustering")

library(tidyverse)
library(feather)
library(arrow)


dm6 <- arrow::read_feather("data/embedded_celldyn_ALL_nn50_ndim6_w_labels_.feather")
cell_dyn <- arrow::read_feather("data/celldyn_cleaned_transformed_imputed_MCAR_with_ratios.feather")

dm6 %>% select(study_id) %>% unique()
dm6 %>% select(sample_dt) %>% arrange()
length(colnames(cell_dyn))

