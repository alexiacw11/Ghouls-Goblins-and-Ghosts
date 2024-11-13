install.packages("remotes")
remotes::install_github("rstudio/tensorflow")

reticulate::install_python()

install.packages("keras")
keras::install_keras()

library(remotes)
library(keras)
library(tidymodels)
library(vroom)

train <- vroom("train.csv")
test<- vroom("test.csv")

train <- train %>%
  mutate(across(where(is.character), as.factor))

nn_recipe <- recipe(type ~ ., data = train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min=0, max=1)

nn_model <- mlp(hidden_units = tune(), epochs = 50) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_model(nn_model) %>%
  add_recipe(nn_recipe) 

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 100)), levels=5)

cv_folds <- vfold_cv(train, v = 10, repeats = 1)

tuned_results <- nn_wf |>
  tune_grid(resamples = cv_folds, grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

tuned_results %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

best_params <- tuned_results |>
  select_best(metric = "accuracy")

nn_final_wf <- nn_wf %>%
  finalize_workflow(best_params) %>%
  fit(train)
  
predictions <- predict(nn_final_wf, new_data = test, type = "class") 

kaggle_submission <- predictions %>%
  bind_cols(., test) %>%
  dplyr::select(id,.pred_class) %>%
  rename(type = .pred_class)

vroom_write(kaggle_submission, "NNpreds.csv", delim = ",")
