install.packages("tidyverse")
install.packages("feather")
install.packages("lubridate")
install.packages("rvest")
install.packages("arrow")

setwd("T:/laupodteam/AIOS/Chontira/CellDynClustering")

library(tidyverse)
library(feather)
library(arrow)


dm6 <- arrow::read_feather("data/embedded_celldyn_nn50_ndim6_w_labels.feather")
cell_dyn <- arrow::read_feather("data/celldyn_cleaned_transformed_imputed_MCAR_with_ratios.feather")


sub <- sample_n(dm6, 5000) 
sub_cell <- sample_n(cell_dyn, 100000)

cell_dyn %>% group_by(gender) %>% 
  select(c_b_wbc, c_b_neu) %>% 
  summarise(avg_wbc = mean(c_b_wbc), avg_neu = mean(c_b_neu))

dm6 %>% group_by(study_id) %>% summarise(number_of_samples = n()) %>% arrange(desc(number_of_samples))


age_tbl <- mutate(cell_dyn,
                          age_classification 
                          = cut(age, breaks = c(0, 4, 14, 24, 64, Inf), 
                                labels = c("Toddler", "Children", "Youth","Adult", "Senior")))

age_tbl %>% 
  group_by(age_classification) %>% 
  select(age_classification,age ,c_b_wbc) %>% 
  summarise(mean_age = mean(age),mean_wbc = mean(c_b_wbc), num_samples = n())

age_tbl %>%
  count( age_classification ,studyid_alle_celldyn) %>%
  group_by(age_classification)%>%
  summarise(num_patients = n())

dm6 %>%
  mutate(time_different= analysis_dt-sample_dt) %>%
  select(time_different, study_id) %>%
  filter(time_different <= 0) %>%
  group_by(study_id) %>%
  summarise(occurance = n())


#store names of columns with c_b for later usage 
col <- c('age')
for(val in colnames(cell_dyn)){
  
  if(startsWith(val, 'c_b_'))
      col <- c(col, val)
}
length(col)

col <- c('c_b_bas')
aggr_cell <- sub_cell %>% 
  select(all_of(col)) %>% 
  pivot_longer(cols = col, names_to = 'features', values_to = 'values') 


aggr_cell %>%
  group_by(features) %>%
  dplyr::summarise(averages = sd('values'))#, standard_deviations = sd(colnames(aggr_cell)[2]),medians = median(colnames(aggr_cell)[2]), highest = max(colnames(aggr_cell)[2]), lowest = min(colnames(aggr_cell)[2]) )

mean(aggr_cell$values)
mean(cell_dyn$'c_b_bas')

cell_dyn_encoded <- cell_dyn %>%
  mutate(gender_encoded = ifelse(gender == 'M', 0,1))

cell_dyn_encoded

write_feather(cell_dyn_encoded, 'data/cell_dyn_gender_encoded.feather')
