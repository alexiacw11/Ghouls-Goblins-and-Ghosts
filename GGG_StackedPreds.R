# Stacked preds of random forest + Naive Bayes
# Didn't perform that well, 0.708 on Kaggle

# Load libraries
library(stacks) # for stacking
library(tidymodels)
library(DataExplorer)
library(ggplot2)
library(discrim)

# Load libraries
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# For stacked predictions
untunedModel <- control_stack_grid() 

# Recipe - 0.74291
my_recipe <- recipe(type ~ ., data = train) |> 
  step_mutate_at(all_nominal_predictors(), fn=factor) |> 
  step_normalize(all_numeric_predictors())

# Naive Bayes
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

nb_wf <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(nb_model)

grid_of_tuning_params <- grid_regular(Laplace(), smoothness(), levels = 5) # L^2 total tuning possibilties

nb_models <- tune_grid(nb_wf, resamples = folds, 
                       grid = grid_of_tuning_params, metrics = metric_set(roc_auc), 
                       control = untunedModel)

# Random forest
rf_model <- rand_forest(mtry=tune(), min_n = tune(), trees=tune()) |> 
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(rf_model)

tree_grid <- grid_regular(mtry(range = c(1, 10)), min_n(), trees(), 
                          levels = 5)

# Random forest model
rf_models <- tune_grid(rf_wf, resamples = folds, grid = tree_grid, 
                       metrics = metric_set(roc_auc), control = untunedModel)

my_stack <- stacks() %>%
  add_candidates(nb_models) %>%
  add_candidates(rf_models) 

stack_mod <-my_stack %>%
  blend_predictions() %>%
  fit_members()

stack_mod_predictions <- predict(stack_mod, new_data = test)
stack_mod_predictions

# Kaggle submissions
kaggle_submission <- predictions %>%  
  bind_cols(., test) %>% 
  dplyr::select(id,.pred_class) %>% 
  rename(type= .pred_class)

vroom::vroom_write(x=kaggle_submission, file="./StackedPreds.csv", delim=",")
