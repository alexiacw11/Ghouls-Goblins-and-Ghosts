# Light EDA and recipe imputation practice

# Load libraries
library(tidymodels)
library(missing_trainExplorer)
library(ggplot2)

# Read in the missing_train
missing_train <- vroom::vroom("trainwithmissingvalues.csv")
train <- vroom::vroom("train.csv")
test <- vroom::vroom("test.csv")

# Only have 7 columns: 2 chrs and 5 numeric, 371 rows
glimpse(missing_train)

# Get numerical summaries
missing_train |> 
  summary() |> 
  knitr::kable("rst")

# Missing values per columns, not too bad
missing_train |> 
  plot_missing()

# Balance of target variable, they seem fairly similar
ggplot(data = missing_train, mapping = aes(x=type, fill = type)) +
  geom_bar()

# Different recipe imputation methods

# Mean/mode imputation, 0.1526155 RMSE
mm_recipe <- recipe(type ~ ., data=missing_train) |> 
  step_impute_mean(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors())

# Median/mode imputation, 0.1520589 RMSE
mm_recipe <- recipe(type ~ ., data=missing_train) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors())

# Prep and bake mm_recipe
prepped_recipe <- prep(mm_recipe, new_data=NULL)
baked <- bake(prepped_recipe, new_data = missing_train)

# Linear regression, 0.1461071 RMSE
lr_recipe <- recipe(type ~ ., data = missing_train) |> 
  step_mutate(color = factor(color), type = factor(type)) |> 
  step_impute_bag(all_numeric_predictors(), 
                  impute_with=imp_vars(all_predictors()),
                  trees = 5000)

# Prep and bake regression recipe
prepped_recipe <- prep(lr_recipe, new_data=NULL)
baked <- bake(prepped_recipe, new_data = missing_train)

# Bagged trees - 0.1462306 RMSE
bt_recipe <- recipe(type ~ ., data=missing_train) |> 
  step_impute_bag(all_numeric_predictors(), impute_with=imp_vars(all_predictors()), 
                  trees= 1000)

# Prep and bake bagged recipe
prepped_recipe <- prep(bt_recipe, new_data=NULL)
baked <- bake(prepped_recipe, new_data = missing_train)

# Knn -  RMSE
knn_recipe <- recipe(type ~ ., data=missing_train) |> 
  step_impute_knn(all_numeric_predictors(), impute_with=imp_vars(all_predictors()), neighbors = 5)

prepped_recipe <- prep(knn_recipe, new_data=NULL)
baked <- bake(prepped_recipe, new_data = missing_train)

# Calculate the RMSE of imputations
rmse_vec(train[is.na(missing_train)], baked[is.na(missing_train)])













