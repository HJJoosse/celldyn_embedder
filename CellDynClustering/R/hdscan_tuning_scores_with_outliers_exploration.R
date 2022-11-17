setwd("T:/laupodteam/AIOS/Chontira/CellDynClustering")

library(tidyverse)
library(feather)
library(arrow)
scores_full <- read_csv("models/cell_dyn_hdbscan_hyper_op_outliers_results_20000.csv")


colnames(scores_full)

scores_full %>% ggplot(aes(x= metric, y = silhouette_score))+
  geom_boxplot()

## locating combinations with compatible number of labels and sh score
scores_full %>% 
  sample_n(250) %>%
  ggplot(aes(x=num_labels, y=silhouette_score))+
  geom_point(size=2, shape=23)+
  xlim(5,20)+
  geom_text(aes(label=index), size=3)

print(filter(scores_full, index == 328) %>% select(params, silhouette_score))
#index 328 is compatible
#{'min_samples': 5, 'min_cluster_size': 5, 'cluster_selection_epsilon': 0.1, 'cluster_selection_method': 'eom', 'metric': 'manhattan'
#0.416

# Comapring min_sample values to see which one
#can give a good num labels with good sh scores (>0.4)
scores_full %>% 
  mutate(across(min_samples,factor)) %>%
  filter(silhouette_score > 0.4)%>%
  ggplot(aes(x=min_samples, y=num_labels))+
  geom_boxplot()+
  ylim(5,20)

#min sample 5

# Comapring min_cluster_size values to see which one
#can give a good num labels with good sh scores (>0.4)
scores_full %>% 
  mutate(across(min_cluster_size,factor)) %>%
  filter(silhouette_score > 0.4)%>%
  ggplot(aes(x=min_cluster_size, y=num_labels))+
  geom_boxplot()+
  ylim(5,20)

# cluster size 2,5

# Comapring cluster_selection_epsilon values to see which one
#can give a good num labels with good sh scores (>0.4)
scores_full %>% 
  mutate(across(cluster_selection_epsilon,factor)) %>%
  filter(silhouette_score > 0.4)%>%
  ggplot(aes(x=cluster_selection_epsilon, y=num_labels))+
  geom_boxplot()+
  ylim(5,20)

# cluster_selection_epsilon 0.1 - 1

# Comapring cluster_selection_method values to see which one
#can give a good num labels with good sh scores (>0.4)
scores_full %>% 
  mutate(across(cluster_selection_method,factor)) %>%
  filter(silhouette_score > 0.4)%>%
  ggplot(aes(x=cluster_selection_method, y=num_labels))+
  geom_boxplot()+
  ylim(5,20)

# cluster_selection_method eom is better but leaf seems neck and neck

# Comapring metric values to see which one
#can give a good num labels with good sh scores (>0.4)
scores_full %>% 
  mutate(across(metric,factor)) %>%
  filter(silhouette_score > 0.4)%>%
  ggplot(aes(x=metric, y=num_labels))+
  geom_boxplot()+
  ylim(5,20)

# metric manhatten has a wider range but mahalanobis is stable overall 


