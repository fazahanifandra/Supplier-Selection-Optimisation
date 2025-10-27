"""
Only this main.py file is needed to be run. It will automatically run custom functions from other required modules.

The steps in this main.py are: 
1. Loading and preparing the datasets (using functions from `data_preparation_1.py`
2. Performing Exploratory Data Analysis (using functions from `eda_2.py`)
3. Train and evaluate machine learning models (using functions from `machine_learning_3.py`)

Before running this main.py, ensure that:
- All the datasets (task.csv, suppliers.csv, costs.csv.zip) are in the same folder
- All the modules (`data_preparation_1.py`,`eda_2`, `machine_learning_3`) are in the same folder

Notes: Running CV LOGO & Grid Search (step 3.7 and 3.8) might be computationally expensive, reducing the number of model (from `models` dictionary) or 
       adjusting the `param_grids` to have shorter runtime

After running this main.py script:
- Optionally run `report.py` for additional analysis of machine learning results

"""

# Importing pandas and numpy libraries
import pandas as pd
import numpy as np

# Importing sklearn packages 
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer

# Importing custom functions from modules
from data_preparation_1 import *
from eda_2 import *
from machine_learning_3 import *

# ==== Step 1: Loading Data & Data Preparation ====

## 1.1 Load the datasets
## load cost, tasks, and suppliers. Assuming files are in the same folder
cost, tasks, suppliers = load_df('cost.csv.zip','tasks.csv', 'suppliers.csv') 

## 1.2 Check for missing values & data info
## Verify if theres any missing values and print dataset info for cost, tasks, and suppliers datasets.
check_mv_info(cost, suppliers, tasks) 

### 1.2.1 Converting object to numeric in dataset
### Ensures objects are converted into numeric type (except for Task ID, Supplier ID)
convert_object_to_numeric([tasks, cost, suppliers])

### 1.2.2 Transposing supplier dataset
### Transpose row and column of 'supplier' dataset, then rename the column name to ease the data processing
suppliers = transpose_data(suppliers)
suppliers.info() # to confirm that the transpose are applied correctly

## 1.3 Check for missing IDs in cost data
## Identify missing 'Task ID' values in the 'cost' dataset compared to the 'tasks' and 'suppliers' dataset.
missing_task_ids = check_missing_ids(tasks, cost, 'Task ID', 'Tasks', 'Cost')              
missing_suppliers_ids = check_missing_ids(suppliers, cost, 'Supplier ID', 'Suppliers', 'Cost')   
## Note: Missing ID's will later be automatically addressed when we merge tasks, suppliers, and cost data into a single dataset

## 1.4 Scale the features in tasks and suppliers data (to be used for EDA) 
## Scaling the tasks and supplier data using standard scaler
tasks_filtered = scaling_data_std(tasks)
suppliers_filtered = scaling_data_std(suppliers)

## 1.5 Remove features with low variances in tasks and suppliers data 
## Filters out features that have variance below a specified threshold (default: 0.01)
tasks_filtered = remove_low_variances(tasks_filtered, threshold=0.01)          
suppliers_filtered = remove_low_variances(suppliers_filtered, threshold=0.01)   
## Note: the threshold can be adjusted based on acceptable level of variance

### 1.5.1 Merge all dataset to prepare for features ranking 
### Combined task and cost data set using "Task ID" as key
task_cost = merge_data(tasks_filtered, cost, "Task ID", "Task ID")
### Combined task_cost and supplier data set using "Supplier ID" as key
task_supplier_cost = merge_data(task_cost, suppliers_filtered, "Supplier ID", "Supplier ID")

### 1.5.2 Rank the features based on importance
### Compute the importance of features in predicting 'Cost' using a Random Forest model
### - A subset of tasks is randomly selected as a test group based on unique 'Task ID'
### - The model is trained on the remaining tasks, and permutation importance is used to evaluate feature significance
data_feat_imp = feature_importance(task_supplier_cost)
### Note: This step may take some time due to the computation of permutation importance

### 1.5.3 Call to remove features with high correlation in tasks and suppliers data (thershold can be modified)
## Filters out features that with correlation above a specified threshold (default: 0.8)
## - For each pair of features exceeding the threshold, the feature with lower importance is removed.
tasks_filtered = remove_high_correlation(tasks_filtered, threshold=0.8,importance_df=data_feat_imp)         
suppliers_filtered = remove_high_correlation(suppliers_filtered, threshold=0.8,importance_df=data_feat_imp)
## Note: the threshold can be adjusted based on acceptable level of correlation

## 1.6 Apply the filter to remove suppliers with cost above average in both suppliers and cost data
## Filter out suppliers whose costs are above the average cost for each task.
suppliers_filtered = filter_suppliers_above_average(suppliers_filtered, cost)

# ==== Step 2: Exploratory Data Analysis ====

## 2.1 Create a boxplot graph for distribution of task features
DistributionOfTaskFeatures(tasks_filtered)

## 2.2 Create a heatmap graph for the cost values for various tasks across various suppliers
CostOfTasksAndSuppliers(cost)

## 2.3 Create a boxplot graph for distribution of errors (Eq. 1) by suppliers,
## Along with RMSE values (Eq. 2) for each supplier for all tasks
DistributionOfErrorsBySuppliers(cost)



# ==== Step 3: Train & Evaluate Machine Learning Models ====

## 3.1 Define and run the custom score function (using Eq. 2) in this module so the models can run without issues
### This function calculates errors in the same manner as function 'calculate_task_errors', but specifically for cross-validation
def custom_score(y_true, y_pred):
    return y_true.min() - y_true.iloc[y_pred.argmin()]
custom_scorer = make_scorer(custom_score)

## 3.2 Filter original data (before scaling) to include only columns present in the filtered datasets
tasks = filter_columns_by_name(tasks, tasks_filtered)
suppliers = filter_columns_by_name(suppliers, suppliers_filtered)
## Note: Scaling will be applied later using a pipeline, so these datasets remain unscaled for now.

### 3.2.1 Re-run the filter suppliers above average function to maintain the original cost data 
suppliers = filter_suppliers_above_average(suppliers, cost)

## 3.3 Merge all dataset into one to prepare for machine learning
task_cost = merge_data(tasks, cost, "Task ID", "Task ID")
merged_data = merge_data(task_cost, suppliers, "Supplier ID", "Supplier ID")

## 3.4 Split data into training and testing sets randomly
np.random.seed(555) # to ensure reproducibility
## Identify unique 'Task ID' in the merged dataset and randomly select 20 unique Task ID as 'TestGroup'
unique_tasks = merged_data['Task ID'].unique()
TestGroup = np.random.choice(unique_tasks, size=20, replace=False)  

### 3.4.1 Split the input (x) and output (y)
x = merged_data.drop(columns=['Cost'])  # dropping 'Cost' to make 'TF' and 'SF' as input variable
y = merged_data['Cost']                 # assigning 'Cost' as the output variable
groups = merged_data['Task ID']         # retaining 'Task ID' as grouping variable

### 3.4.2 Split the data into training and testing sets based on TestGroup
### Use TestGroup to split the dataset into:
###  - `X_train` and `y_train`: Contain data for tasks not in the TestGroup (training set).
###  - `X_test` and `y_test`: Contain data for tasks in the TestGroup (testing set).
X_train = x[~x['Task ID'].isin(TestGroup)].copy()
y_train = y[~x['Task ID'].isin(TestGroup)].copy()
X_test = x[x['Task ID'].isin(TestGroup)].copy()
y_test = y[x['Task ID'].isin(TestGroup)].copy()
### Note: .copy() is used to avoid modifying the original data

### 3.5 Make a Dictionary for each model so each function can be looped 
models = {'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42),
          'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
          'Lasso': Lasso(alpha=0.001, random_state=42),
          'Ridge': Ridge(),
          'SVR': SVR(),
          'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform')}
### Note: The computation time will increase if more models are added. It can also be decreased if we remove some models.
   
## 3.6. Loop through the models to calculate RMSE (Train and Test) and Model Score (R²)
## For each ML in the models dictionary, it will train and evaluate the model using `train_and_evaluate_pipeline` function
## - During pipeline, the scaling will be done to the training set 
## - Results will be saved in the `results_train_test` dictionary and printed in a dataframe format for readabliity
results_train_test = {}
for name, model in models.items():
    print(f"Training and evaluating {name}:")
    results_train_test[name] = train_and_evaluate_pipeline(model, X_train, y_train, X_test, y_test)
    print(f"Results for {name}: {results_train_test[name]}")
results_train_test = pd.DataFrame(results_train_test).T  # make it into dataframe & transposed for readability
print(f"Model Performance Summary \nSimple Train-Test Split:\n{results_train_test}")

## 3.7 Loop through the models to Calculate RMSE (Train) using LOGO (each TaskGroup) Cross-Validation 
## For each ML in the models dictionary, it will train and evaluate the model using `train_and_evaluate_logo_pipeline` function
## - During pipeline, the scaling will be done to the training set 
## - scoring is done using `custom_scorer` 
## - LOGO defined for each TaskGroup
## - Results will be saved in the `results_CV` dictionary and printed in a dataframe format for readabliity
results_CV = {}
for name, model in models.items():
    print(f"Training and evaluating {name}:")
    results_CV[name] = train_and_evaluate_logo_pipeline(model, X_train, y_train, groups, custom_scorer)
    print(f"Results for {name}: {results_CV[name]}")
results_CV_df = pd.DataFrame(results_CV).T  # make it into dataframe & transposed for readability
print(f"Model Performance Summary\nCV LOGO - Before Hyperparameter (Train Data):\n{results_CV_df}")

## 3.8 Find the best hyperparameters using Grid Search 
### 3.8.1 Set the param_grids for each model so it can be looped
param_grids = {'RandomForest': {'n_estimators': [50, 100],
                                'max_depth': [10, 20],
                                'min_samples_split': [2, 5],
                                'min_samples_leaf': [1, 2],
                                'max_features': ['sqrt', 'log2']},
               'HistGradientBoosting': {'max_depth': [10, 20],
                                        'min_samples_leaf': [2, 5],
                                        'max_iter': [100],
                                        'learning_rate': [0.01, 0.05, 0.1],
                                        'l2_regularization': [0.0, 1.0]},
               'KNN': {'n_neighbors': [3, 5, 7, 10],
                       'weights': ['uniform', 'distance'],
                       'metric': ['euclidean', 'manhattan'],
                       'p': [1, 2]},
               'SVR': {'C': [0.5, 1.0],
                       'kernel': ['linear', 'rbf']},
               'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
               'Ridge': {'alpha': np.logspace(-4, 4, 10),
                         'solver': ['auto', 'cholesky', 'sparse_cg', 'svd', 'lsqr', 'sag', 'saga']}}
### Note: The param grids can be modified or expanded depending on the dataset

### 3.8.2 Loop through the models for performing Grid Search with LOGO CV to find best hyperparameters
### For each ML in the models dictionary, it will find the best hyperparameters using `perform_grid_search_with_logo_pipeline` function
### - During pipeline, the scaling will be done to the training set 
### - Scoring is done using `custom_scorer` 
### - LOGO defined for each TaskGroup
### - Results will be saved in the `results_grid_search` dictionary and printed in a dataframe format for readabliity
results_grid_search = {}
for model_name, model in models.items():
    print(f"Performing Grid Search for {model_name}:")   
    param_grid = param_grids[model_name]        
    best_params, time_taken, fitted_grid_search = perform_grid_search_with_logo_pipeline(
        model=model,
        param_grid=param_grid,
        X=X_train.drop(columns=['Task ID', 'Supplier ID']),
        y=y_train,
        groups=groups,
        custom_scorer=custom_scorer) # to show our progress       
    results_grid_search[model_name] = {'Best Params': best_params, 'Time Taken (s)': time_taken} # storing result
results_grid_search_df = pd.DataFrame(results_grid_search).T # make it into dataframe & transposed for readability
print(f"Grid Search Results Summary:\n{results_grid_search_df}")
### Note: This process can be computationally intensive, especially with multiple models and large parameter grids

## 3.9 Loop through the models to recalculate RMSE (Train and Test) using best parameters
### 3.9.1 Calculating RMSE of Training set (with best params) using CV LOGO
### It uses the same function as step 3.7
### - In the loop, best_params from Grid Search result are used for each models
results_CV_best_param = {}
for name, model in models.items():
    print(f"Training and evaluating {name} with best hyperparameters:")
    best_params = results_grid_search[name]['Best Params'] # Retrieve the best params from the grid search
    model.set_params(**{key.split('__')[-1]: value for key, value in best_params.items()}) # Setting the best params for the model 
    results_CV_best_param[name] = train_and_evaluate_logo_pipeline(model, X_train, y_train, groups, custom_scorer)
    print(f"Results for {name}: {results_CV_best_param[name]}")  
results_CV_best_param_df = pd.DataFrame(results_CV_best_param).T  # Transpose for better readability
print(f"Model Performance Best Parameters (Train Data):\n{results_CV_best_param_df}")

### 3.9.2 Calculating RMSE of Test set (with best params)
### It uses similar function as step 3.6 but only for evaluating Test set
### - In the loop, best_params from Grid Search result are used for each models
results_test_best_param = {}
for name, model in models.items():
    print(f"Evaluate Test Data for {name} with best parameters:")
    best_params = results_grid_search[name]['Best Params'] # Retrieve the best params from the grid search
    model.set_params(**{key.split('__')[-1]: value for key, value in best_params.items()}) # Setting the best params for the model
    results_test_best_param[name] = testdata_pipeline(model, X_train, y_train, X_test, y_test)
    print(f"Results for {name}: {results_test_best_param[name]}")
results_test_best_param_df = pd.DataFrame(results_test_best_param).T # Transpose for better readability
print(f"Model Performance Summary with Best Parameters on Test Data:\n{results_test_best_param_df}")

