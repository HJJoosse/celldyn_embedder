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
                  = cut(age, breaks = c(-1, 3, 17, 64, Inf), 
                        labels = c("Toddler", "Children","Adult", "Senior")))

age_tbl %>% 
  group_by(age_classification) %>% 
  select(age_classification,age ,c_b_wbc) %>% 
  summarise(mean_age = mean(age),mean_wbc = mean(c_b_wbc), num_samples = n())
age_tbl %>% filter(age < 0) %>% ggplot(.,aes(x = age)) + geom_density()

age_tbl %>% filter(age < 0) %>% qplot(.['age'],geom = "density")

age_tbl %>% filter(age < 0) %>%  ggplot()+geom_density()

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
