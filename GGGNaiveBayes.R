# Load libraries
library(tidymodels)
library(DataExplorer)
library(ggplot2)
library(discrim)
library(glmnet)
library(embed)
library(themis) # for smote

# Load libraries
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Check for nas
any(is.na(train))

# The data looks pretty normal
train |> 
  plot_density()

# Correlation plot, few higher than .5, consider getting rid of them in recipe? 
train |> 
  plot_correlation()

# Look at balance of type, ghost is slightly lower
ggplot(data = train, mapping = aes(x=type, fill=type)) + geom_bar() 

# Recipe - 0.70888
# my_recipe <- recipe(type ~ ., data = train) |> 
#   step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
#   step_smote(type)

# Recipe - 0.72022
# my_recipe <- recipe(type ~ ., data = train) |> 
#   step_dummy(all_nominal_predictors(), one_hot = TRUE) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_smote(type) 

# Recipe - 0.72211
# my_recipe <- recipe(type ~ ., data = train) |> 
#   step_dummy(all_nominal_predictors()) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_smote(type)
  
# Threshold at 0.5 = 0.74102, and threshold at 0.60 = 0.74291
# my_recipe <- recipe(type ~ ., data = train) |> 
#   step_mutate_at(all_nominal_predictors(), fn=factor) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_corr(all_numeric_predictors(), threshold = 0.60)

# Recipe - 0.74858
my_recipe <- recipe(type ~ ., data = train) |> 
  step_mutate_at(all_nominal_predictors(), fn=factor) |> 
  step_normalize(all_numeric_predictors()) %>% # all_numeric(), -all_outcomes()
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_smote(all_outcomes(), neighbors = 5)

# Recipe - 0.75236
my_recipe <- recipe(type ~ ., data = train) |> 
  step_mutate_at(color, fn=factor) |> 
  step_normalize(all_numeric(), -all_outcomes()) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>%
  step_smote(all_outcomes(), neighbors = 5)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

nb_wf <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(nb_model)

# Grid of values to tune over
grid_of_tuning_params <- grid_regular(Laplace(), smoothness(), levels = 20) # L^2 total tuning possibilties

# Split data for CV (5-10 groups)
folds <- vfold_cv(train, v=10, repeats = 1)

# Run the CV, no difference between accuracy and roc_ac
CV_results <- nb_wf %>%
  tune_grid(resamples=folds, 
            grid=grid_of_tuning_params, 
            metrics=metric_set(roc_auc))

# Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

# Finalize the workflow and fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>% 
  fit(data=train)

# Make predictions, type=class
predictions <- predict(final_wf, new_data = test, type = "class")
predictions

# Kaggle submission 
kaggle_submission <- predictions %>%  
  bind_cols(., test) %>% 
  dplyr::select(id,.pred_class) %>% 
  rename(type= .pred_class)

vroom::vroom_write(kaggle_submission, "NaiveBayesPreds.csv", delim = ",")
