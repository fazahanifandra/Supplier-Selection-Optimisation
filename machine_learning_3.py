""" 
This module contains 8 functions for Machine Learning fitting and scoring:
   1. filter_columns_by_name: Filters a DataFrame to include only columns present in a reference DataFrame.
   2. calculate_task_errors: Computes task-level errors by comparing true and predicted costs.
   3. calculate_rmse: Calculates Root Mean Squared Error (RMSE) based on Equation 2.
   4. train_and_evaluate_pipeline: Trains a pipeline and evaluates its performance on train and test datasets.
   5. train_and_evaluate_logo_pipeline: Performs Leave-One-Group-Out cross-validation (LOGO CV) using a pipeline.
   6. perform_grid_search_with_logo_pipeline: Conducts hyperparameter tuning using Grid Search with LOGO CV.
   7. testdata_pipeline: Evaluates a trained pipeline on a test dataset and calculates RMSE.
"""
# Import the required Python packages
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Part 3: Machine Learning
# Filters a DataFrame to include only columns present in a reference DataFrame
def filter_columns_by_name(df, filtered_df):
    # Get the column names from the reference DataFrame
    reference_columns = filtered_df.columns

    #  Return a new DataFrame containing only the columns present in `filtered_df`
    new_df = df[reference_columns]
    return new_df

# Function to calculates task-level errors based on the difference between 
# the minimum true cost and the predicted cost for each task
def calculate_task_errors(y_true, y_pred, task_ids):
    errors = []
    unique_task_ids = task_ids.unique()
    
    for task_id in unique_task_ids:
        # Get indices for the current task
        task_indices = task_ids == task_id

        # Extract true and predicted costs for the task
        y_true_task = y_true[task_indices]
        y_pred_task = y_pred[task_indices]

        # Calculate task-level error
        error = y_true_task.min() - y_true_task.iloc[y_pred_task.argmin()]
        errors.append(error)
    
    return errors

# Function to calculate RMSE across tasks based on Equation 2
def calculate_rmse(errors):
    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse

# Function to trains a machine learning pipeline and evaluates it on train and test sets.
# Pipelines ensure consistent model training while preventing data leakage.
def train_and_evaluate_pipeline(model, X_train, y_train, X_test, y_test):
    # Define pipeline with scaling and model
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('model', model)])
    pipeline.fit(X_train.drop(columns=['Task ID', 'Supplier ID']), y_train)

    # Predictions for train and test sets
    y_pred_train = pipeline.predict(X_train.drop(columns=['Task ID', 'Supplier ID']))
    y_pred_test = pipeline.predict(X_test.drop(columns=['Task ID', 'Supplier ID']))

    # Calculate task-level errors
    errors_train = calculate_task_errors(y_train, y_pred_train, X_train['Task ID'])
    errors_test = calculate_task_errors(y_test, y_pred_test, X_test['Task ID'])

    # Calculate RMSE using function calculate_rmse
    rmse_train = calculate_rmse(errors_train)
    rmse_test = calculate_rmse(errors_test)

    # Calculate model score
    model_score = pipeline.score(X_test.drop(columns=['Task ID', 'Supplier ID']), y_test)

    return {'RMSE on Train': rmse_train,
            'RMSE on Test': rmse_test,
            'Model Score': model_score}

# Function to perform Leave-One-Group-Out cross-validation (LOGO CV) using a machine learning pipeline.
# Pipelines ensure consistent model training while preventing data leakage.
def train_and_evaluate_logo_pipeline(model, X_train, y_train, groups, custom_scorer):
    # Define pipeline with scaling and model
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('model', model)])
    pipeline.fit(X_train.drop(columns=['Task ID', 'Supplier ID']), y_train)
    
    start_time = time.time() # start timer to know how long each model tooks

    # Perform LOGO cross-validation
    scores = cross_val_score(pipeline,
                             X_train.drop(columns=['Task ID', 'Supplier ID']), # Training feature dataset
                             y_train, # Training target dataset
                             groups=groups[X_train.index], # CV grouped by Task ID
                             cv=LeaveOneGroupOut(),
                             scoring=custom_scorer, # use custom_scorer function based on (Eq 2) to evaluate model performance
                             n_jobs=-1) # use all available processors for faster processing
                             
    end_time = time.time() # stop the timer after cv finished
    
    # Calculate RMSE
    total_errors = []
    for score in scores:
        total_errors.append(score)    
    rmse =  calculate_rmse(total_errors) # get the rmse for each model
    
    print(f"LOGO CV - Score : {scores}")
    print(f"{len(scores)}-fold LOGO CV - RMSE Score: {rmse}")
    print(f"{len(scores)}-fold LOGO CV - Time Taken: {end_time - start_time:.2f} seconds")
    
    # Returns a dictionary containing RMSE and the total time taken
    return {'RMSE': rmse,
            'Time Taken (s)': end_time - start_time}

"""
The machine learning model needs to be optimised, hence pipelines is used to ensure 
consistent model training while preventing data leakage.
Using the param_grid for each model, perform Grid Search using the pipeline and LOGO CV and
return the best parameters and time taken for each model
"""
# Function performs hyperparameter tuning using Grid Search with 
# Leave-One-Group-Out cross-validation (LOGO CV) using a machine learning pipeline
def perform_grid_search_with_logo_pipeline(model, param_grid, X, y, groups, custom_scorer, verbose=2):   
    # Create a pipeline with scaling and the model
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('model', model)])
    
    # Update parameter grid to account for the pipeline
    pipeline_param_grid = {f'model__{key}': value for key, value in param_grid.items()}

    # Define LeaveOneGroupOut CV
    logo = LeaveOneGroupOut() 
    cv = logo.split(X, y, groups[X.index]) # Generate train/test splits based on groups

    # Define GridSearchCV with the pipeline
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_param_grid, # Dictionary containing all hyperparameters grid for all models.
        scoring=custom_scorer, # use custom_scorer function based on (Eq 2) to evaluate model performance
        cv=cv, # use LOGO cross-validation
        refit=True, # # Fit the model with the best parameters on the full dataset
        n_jobs=-1, # use all available processors for faster processing
        verbose=verbose) # Displays progress for each fold and parameter combination
        
    start_time = time.time() # start timer to know how long each model tooks
    
    # Fit the Grid Search to the data
    grid_search.fit(X, y)
    
    end_time = time.time() # stop the timer after grid search finished fitting
    
    # Extract the best parameters and time taken
    best_params = grid_search.best_params_
    time_taken = end_time - start_time

    print(f"Grid Search Completed: Best Parameters: {best_params}")
    print(f"Time Taken: {time_taken:.2f} seconds")
    
    return best_params, time_taken, grid_search

# Evaluates the model on a test dataset using a machine learning pipeline.
def testdata_pipeline(model, X_train, y_train, X_test, y_test):
    # Create and train the pipeline
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('model', model)])
    pipeline.fit(X_train.drop(columns=['Task ID', 'Supplier ID']), y_train)

    # Predict target values on the test dataset
    y_pred_test = pipeline.predict(X_test.drop(columns=['Task ID', 'Supplier ID']))

    # Calculate task-level errors
    errors_test = calculate_task_errors(y_test, y_pred_test, X_test['Task ID'])

    # Calculate RMSE
    rmse_test = calculate_rmse(errors_test)

    # Return the RMSE result for the test dataset
    return {'RMSE on Test': rmse_test}