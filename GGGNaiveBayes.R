# Load libraries
library(tidymodels)
library(missing_trainExplorer)
library(ggplot2)

# Load libraries
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Check for nas
any(is.na(train))

# Recipe - 0.74291
my_recipe <- recipe(type ~ ., data = train) |> 
  step_mutate_at(all_nominal_predictors(), fn=factor) |> 
  step_normalize(all_numeric_predictors())

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")

nb_wf <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(nb_model)

# Grid of values to tune over
grid_of_tuning_params <- grid_regular(Laplace(), smoothness(), levels = 5) # L^2 total tuning possibilties

# Split data for CV (5-10 groups)
folds <- vfold_cv(train, v=5, repeats = 1)

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
