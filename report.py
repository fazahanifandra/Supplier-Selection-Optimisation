"""
This is an optional module. This file should be run after main.py has finished running.

It contains additional analysis for:
1. Appendix A: Analyzing best suppliers based on actual vs predicted costs by ML.
2. Appendix B: Predicting costs for missing Task IDs.

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict


## Creating functions
### 1. Function to create a new DataFrame containing the predicted cost by best ML model
def create_predicted_costs_df(x_data, y_pred):
    return pd.DataFrame({'Task ID': x_data['Task ID'].values,
                         'Supplier ID chosen by ML': x_data['Supplier ID'].values,
                         'Predicted cost by ML': y_pred})

### 2. Function to find the best ML supplier for each task based on minimum predicted cost
def find_best_suppliers(predicted_costs):
    return (predicted_costs.loc[predicted_costs.groupby('Task ID')['Predicted cost by ML'].idxmin()].reset_index(drop=True))

### 3. Function to merge the predicted best suppliers with the actual cost dataset
def merge_with_actual_costs(best_suppliers, merged_data):
    return best_suppliers.merge(merged_data[['Task ID', 'Supplier ID', 'Cost']],
                                left_on=['Task ID', 'Supplier ID chosen by ML'],
                                right_on=['Task ID', 'Supplier ID'],
                                how='left').rename(columns={'Cost': 'Actual cost of ML-chosen supplier'})

### 4. Function to calculate RMSE for train and test sets separately
def calculate_rmse(comparison):
    train_comparison = comparison[comparison['Set'] == 'Train']
    test_comparison = comparison[comparison['Set'] == 'Test']
    train_rmse = np.sqrt(np.mean(train_comparison['Error'] ** 2))
    test_rmse = np.sqrt(np.mean(test_comparison['Error'] ** 2))
    
    return train_rmse, test_rmse


# ==== Appendix A: Best suppliers based on actual vs predicted by ML ====
## A.1 Identify the best ML model based on the minimum cross-validation RMSE (after hyperparameter optimisation)
### It will select the best ML model along with the best parameters found for the ML model from Grid Search
best_model_name = results_CV_best_param_df['RMSE'].idxmin()
best_model_params = results_grid_search[best_model_name]['Best Params']
best_model = models[best_model_name]

## A.2 Set the best parameters of the best model for the pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('model', best_model)])
pipeline.set_params(**{f'model__{key.split("__")[-1]}': value for key, value in best_model_params.items()})

## A.3 Fit the pipeline on the entire training set used
pipeline.fit(X_train.drop(columns=['Task ID', 'Supplier ID']), y_train)

## A.4 Predict costs for the test set based on pipeline 
### As the test set is considered unseen data, the cost can be predicted by the fitted pipeline 
y_pred_hyper_test = pipeline.predict(X_test.drop(columns=['Task ID', 'Supplier ID']))
predicted_costs_test = create_predicted_costs_df(X_test, y_pred_hyper_test)
best_suppliers_test = find_best_suppliers(predicted_costs_test)

## A.5 Predict costs for the training set using cross-validation predict
### By using cross-validation predict, it will use the same input and output from each cross-validation iteration
### Therefore, the result will be align with the cross-validation RMSE in the main module
logo = LeaveOneGroupOut()
y_pred_cv_train = cross_val_predict(pipeline,
                                    X_train.drop(columns=['Task ID', 'Supplier ID']),
                                    y_train,
                                    cv=logo.split(X_train.drop(columns=['Task ID', 'Supplier ID']), y_train, groups.loc[X_train.index]),
                                    n_jobs=-1)
predicted_costs_train = create_predicted_costs_df(X_train, y_pred_cv_train)
best_suppliers_train = find_best_suppliers(predicted_costs_train)

## A.6 Combine the train and test results and create column for identifier named 'Set'
best_suppliers_test['Set'] = 'Test'
best_suppliers_train['Set'] = 'Train'
best_suppliers_all = pd.concat([best_suppliers_train, best_suppliers_test], ignore_index=True)

## A.7 Find actual best suppliers from actual cost dataset
### This applies to both train and test dataset in the concatenated dataset
actual_min_costs = merged_data.loc[merged_data.groupby('Task ID')['Cost'].idxmin(), ['Task ID', 'Supplier ID', 'Cost']]
actual_min_costs = actual_min_costs.rename(columns={'Cost': 'Actual cost of actual best supplier', 'Supplier ID': 'Actual best supplier ID'}).reset_index(drop=True)

## A.8 Merge the predicted best suppliers with the actual cost dataset
best_suppliers_with_actual = merge_with_actual_costs(best_suppliers_all, merged_data)
comparison = best_suppliers_with_actual.merge(actual_min_costs, on='Task ID', how='left')

## A.9 Calculate error and RMSE for train and test set separately
### The result is consistent with the RMSE train and test for of the best model (using cross-validation and hyperparameter optimisation)
comparison['Error'] = comparison['Actual cost of ML-chosen supplier'] - comparison['Actual cost of actual best supplier']
comparison['Set'] = best_suppliers_all['Set']
comparison['Squared error'] = comparison['Error'] **2
train_rmse, test_rmse = calculate_rmse(comparison)

## A.10 Check the number and percentage of suppliers correctly predicted by the ML
### If the best supplier per ML model matches the actual best suppliers, the column will show TRUE. Otherwise, FALSE
### Then we calculate how many TRUE suppliers
comparison['Correct Supplier'] = comparison['Supplier ID chosen by ML'] == comparison['Actual best supplier ID']
num_correct = comparison['Correct Supplier'].sum()
total_tasks = len(comparison)
percentage_correct = (num_correct / total_tasks) * 100

print(f"Train RMSE (Cross-Validation): {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"Number of correct suppliers: {num_correct}")
print(f"Percentage of correct suppliers: {percentage_correct:.2f}%")


# ==== Appendix B: Predicting cost for missing task IDs ==== 
## B.1 Filter the tasks dataset corresponding to the 10 missing task IDs, to extract the tasks features
### List of missing task IDs is obtained during the data tidying part
missing_tasks = tasks[tasks['Task ID'].isin(missing_task_ids)]

## B.2 Combine the missing task IDs data and suppliers dataset 
### This is to merge the features for both 10 missing task IDs and complete suppliers features for 64 suppliers
missing_tasks_expanded = missing_tasks.merge(suppliers, how='cross')

## B.3 Predict costs for missing Task IDs
### Prepare list the features used in the train set (both task and supplier features after features selection process)
training_features = X_train.drop(columns=['Task ID', 'Supplier ID']).columns

## B.4 Align missing task features with the training feature set
X_missing = missing_tasks_expanded[training_features]

## B.5 Use the fitted pipeline to predict costs
### The fitted pipeline is based on the best ML model (the same pipeline used to run Appendix A above)
missing_tasks_expanded['Predicted Cost'] = pipeline.predict(X_missing)

## B.6 Filter and pivot the output for better presentation
predicted_costs_missing = missing_tasks_expanded[['Task ID', 'Supplier ID', 'Predicted Cost']]
pivoted_missing_cost = predicted_costs_missing.pivot(index = 'Supplier ID', columns = 'Task ID', values = 'Predicted Cost')
