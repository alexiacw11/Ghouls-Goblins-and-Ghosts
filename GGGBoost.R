# Practicing Boosting

# Using libraries
library(bonsai)
library(lightgbm)
library(tidymodels)
library(DataExplorer)

# Read in data
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Correlation Plot
train |> 
  plot_correlation()

# Recipe
my_recipe <- recipe(type ~ ., data = train) |> 
  step_normalize(all_numeric_predictors()) 

# Boosted Model 
boost_model <- boost_tree(tree_depth=tune(),trees=tune(),learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

# Workflow
boost_workflow <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(boost_model) 


## CV tune, finalize and predict here and save results
# Create 10-fold cross-validation
cv_folds <- vfold_cv(train, v = 10, strata = type)
grid_of_tuning_params <- grid_regular(trees(),tree_depth(),
                                      learn_rate(), levels = 5)

# Fit with cross-validation
cv_results <- boost_workflow %>%
  tune_grid(
    resamples = cv_folds,  # Cross-validation folds
    grid = grid_of_tuning_params, # Grid of tuning parameters
    metrics = metric_set(accuracy), # Relevant metrics
    control = control_grid(save_pred = TRUE) # Use control_grid for tuning
  )

# Select Best Parameter
best_params <- select_best(cv_results, metric = "accuracy")

# Finalize workflow
final_wf <- boost_workflow %>%
  finalize_workflow(best_params) %>% 
  fit(data=train)

# Make predictions, type=class
predictions <- predict(final_wf, new_data = test, type = "class")
predictions

# Kaggle submission 
kaggle_submission <- predictions %>%  
  bind_cols(., test) %>% 
  dplyr::select(id,.pred_class) %>% 
  rename(type= .pred_class)

# 0.7088
vroom::vroom_write(kaggle_submission, "BoostedPreds.csv", delim = ",")
#------------------------------------------------------------------------------------

# BART
boost_model <- bart(trees=tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

grid_of_tuning_params <- grid_regular(trees(),levels = 5)

# Fit with cross-validation
cv_results <- boost_workflow %>%
  tune_grid(
    resamples = cv_folds,  # Cross-validation folds
    grid = grid_of_tuning_params, # Grid of tuning parameters
    metrics = metric_set(accuracy), # Relevant metrics
    control = control_grid(save_pred = TRUE) # Use control_grid for tuning
  )

# Select Best Parameter
best_params <- select_best(cv_results, metric = "accuracy")

# Finalize workflow
final_wf <- boost_workflow %>%
  finalize_workflow(best_params) %>% 
  fit(data=train)

# Make predictions, type=class
predictions <- predict(final_wf, new_data = test, type = "class")
predictions

# Kaggle submission 
kaggle_submission <- predictions %>%  
  bind_cols(., test) %>% 
  dplyr::select(id,.pred_class) %>% 
  rename(type= .pred_class)

# 0.55, awful! 
vroom::vroom_write(kaggle_submission, "BartPreds.csv", delim = ",")
