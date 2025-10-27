""" 
There are 11 functions in this module for data preparation such as:
    1. load_df : loading the dataset
    2. check_mv_info: check any missing value in each dataset & shows each data info
    3. convert_object_to_numeric: converts object-type columns (e.g., percentages) to numeric in datasets
    4. transpose_data: transpose rows and columns in a dataset for better processing
    5. check_missing_ids: check if there is missing ID(s) between 2 datasets
    6. scaling_data_std: standardize numeric columns using standardscaler
    7. remove_low_variance: remove features with variances below a specific threshold 
    8. merge_data: combine 2 datasets given key column
    9. feature_importance: compute and rank the feature importance
    10. remove_high_correlation: remove features with high correlation (above threshold)
    11. filter_suppliers_above_average: remove suppliers which have above average cost for each task
"""
# Import the required Python packages
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# Function to load the data needed
def load_df(cost_file, tasks_file, suppliers_file):
    cost = pd.read_csv(cost_file)
    tasks = pd.read_csv(tasks_file)
    suppliers = pd.read_csv(suppliers_file)
    
    return cost, tasks, suppliers

# Function to check for any missing values & info of each dataset
def check_mv_info(cost, suppliers, tasks):
    # Create a dictionary with DataFrames and their names
    files = {'cost': cost, 'suppliers': suppliers, 'tasks': tasks}

    for filename, df in files.items():
        print(f"=== {filename} info ===")
        
        # Display info  
        print(f"{filename} Info:")
        df.info()
        print("-" * 40)
       
        # Check for missing values
        if df.isna().values.any():
            print(f"Number of missing values per column in {filename}:\n{df.isna().sum()}")
        else:
            print(f"There are no missing values in {filename}.")
        
        print("=" * 40)

# Converting object columns to numeric(float)
def convert_object_to_numeric(dataframes, index_column='Task ID'):
    for df in dataframes:
        # Set the specified column as the index if it exists
        if index_column in df.columns:
            df.set_index(index_column)
    
        for column in df.select_dtypes(include=['object']).columns:
            # Skip columns that are 'Task ID' or 'Supplier ID'
            if column in ['Task ID', 'Supplier ID']:
                continue
                
            # Check if the column contains percentage strings
            if df[column].str.contains('%').any():
                # Remove the percentage sign and convert to numeric
                df[column] = pd.to_numeric(df[column].str.rstrip('%'), errors='coerce') / 100
            else:
                # Convert to numeric without modification
                df[column] = pd.to_numeric(df[column], errors='coerce')
        print(df.info())
        
# Transpose row and column for supplier dataset for better processing
def transpose_data(df):
    # Save Supplier IDs before transposition (assuming 'Features' column contains them)
    supplier_features = df['Features'].values
        
    # Transpose the DataFrame (excluding 'Features' column)
    df = df.drop(columns=['Features']).T
        
    # Reset index of transposed DataFrame
    df = df.reset_index(drop=False)
        
    # Add Supplier IDs back as the first column after transposition
    # Ensure the length of supplier_ids matches the number of rows in the transposed DataFrame
    if len(supplier_features) == len(df):
        df.insert(0, 'Supplier ID', supplier_features)
    else:
        print(f"Warning: The length of 'Supplier Features' ({len(supplier_features)}) does not match the number of rows in the transposed data ({len(df.columns)})")

    # Rename the columns to SF1, SF2, SF3, ..., SFn
    df.columns = ['Supplier ID'] + [f'SF{i+1}' for i in range(df.shape[1] - 1)]
        
    # Convert columns back to numeric where possible, excluding 'Supplier ID'
    for col in df.columns[1:]:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except ValueError:
            pass  # Skip if conversion fails
    
    # Return a dataframe where each supplier becomes a row, each feature becomes a column 
    # and the first column contains the Supplier IDs. 
    return df    

# Check if there is missing ID(s) between 2 datasets
def check_missing_ids(df1, df2, col_name, df1_name, df2_name):
    # Check for IDs in dataset1 that are not in dataset2
    missing_in_df2 = df1[~df1[col_name].isin(df2[col_name])]
    
    # Count how many task IDs are missing in dataset2
    missing_count = missing_in_df2.shape[0]
   
    if missing_count > 0:
        print(f"There are {missing_count} task IDs in {df1_name} that are not found in {df2_name}:")
        
        # Save the missing IDs from the DataFrame into a list
        missing_ids = missing_in_df2[col_name].tolist()
        print(missing_ids)
        
    else:
        print(f"All {col_name} in {df1_name} are found in {df2_name}")
        missing_ids = []
        
    # Return list of missing IDs from tasks and suppliers dataset (if any)
    return missing_ids    

# Standardizes numeric columns in the dataset using standardscaler (mean = 0, standard deviation = 1)
# All numeric features are on the same scale to improves the performance of machine learning models
def scaling_data_std(df):
    # Preserve "Task ID" and "Supplier ID" columns if they exist
    preserve_cols = ['Task ID', 'Supplier ID']
    cols_to_preserve = [col for col in preserve_cols if col in df.columns]
    
    # Select only numeric columns for scaling
    numeric_data = df.select_dtypes(include=['number'])
    
    # Apply Standard scaling to the numeric data
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(numeric_data)
    
    # Convert back to DataFrame with original column names
    scaled_data = pd.DataFrame(scaled_array, columns=numeric_data.columns, index=df.index)
    
    # Concatenate the preserved columns back to the scaled data
    final_df = pd.concat([scaled_data, df[cols_to_preserve]], axis=1)
    
    # Return the scaled DataFrame with both numeric and preserved columns
    return final_df
   
# Remove features with variances below threshold. Features with low variances may not impact significantly on the cost determination
def remove_low_variances(df, threshold=0.01):
    
    # Preserve "Task ID" and "Supplier ID" columns if they exist
    preserve_cols = ['Task ID', 'Supplier ID']
    cols_to_preserve = [col for col in preserve_cols if col in df.columns]
    
    # Apply VarianceThreshold to numeric columns only
    selector = VarianceThreshold(threshold=threshold)
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Learn and apply VarianceThreshold to the numeric data
    df_transformed = selector.fit_transform(numeric_df)
    
    # Obtain the filtered feature names
    feature_names = numeric_df.columns[selector.get_support()]
    
    # Identify the dropped features by comparing original and filtered feature names
    dropped_features = list(set(numeric_df.columns) - set(feature_names))
    
    # Print the dropped features and their count
    if len(dropped_features) > 0:
        print(f"\nDropped features ({len(dropped_features)}): {', '.join(dropped_features)}")
    else:
        print("\nNo features were dropped.") 
    
    # Convert the transformed data back to a DataFrame with the filtered feature names
    transformed_df = pd.DataFrame(df_transformed, columns=feature_names, index=df.index)
    
    # Concatenate the preserved columns back to the transformed data
    final_data = pd.concat([transformed_df, df[cols_to_preserve]], axis=1)
    
    # Print the final shape
    print("Data shape:", final_data.shape)
    
    return final_data
    
# Merge 2 datasets given a column (parameter) as key
def merge_data(df1, df2, left_parameter, right_parameter):
    if 'index' in df2.columns:
        df2 = df2.drop(columns=['index'])        
    merged_data = pd.merge(df1, df2, left_on=left_parameter, right_on=right_parameter)

    return merged_data
    
"""
Function to randomly select a subset of tasks as the test group based on unique 'Task ID'.
Then train a Random Forest model on the remaining tasks (training set) and 
compute feature importance using permutation importance.
The function return a DataFrame with feature names and their importance scores.
"""
# Compute and rank the feature importance using a Random Forest model and permutation importance
def feature_importance(df):
    np.random.seed(5555) # Ensure reproducibility
    unique_tasks = df['Task ID'].unique()
    TestGroup = np.random.choice(unique_tasks, size=20, replace=False) # Randomly select 20 tasks for the test group

    # Define features and target
    x = df.drop(columns=['Cost', 'Task ID', 'Supplier ID']) # Drop non-feature columns
    y = df['Cost'] # Target variable

    # Train-test split based on Task ID
    x_train = x[~df['Task ID'].isin(TestGroup)].copy() # Independent variables for training set
    y_train = y[~df['Task ID'].isin(TestGroup)].copy() # Target variable for training set

    print(x_train.head())

    # Fit a baseline Random Forest model
    forest = RandomForestRegressor(random_state=5555)  # Random state for reproducibility
    forest.fit(x_train, y_train)

    # Compute permutation importance on the training set
    perm_importance = permutation_importance(forest, x_train, y_train, n_repeats=10, random_state=5555)

    # Store feature importance in a DataFrame
    importances = perm_importance.importances_mean # Mean importance scores from permutation
    indices = np.argsort(-importances)  # Sort by descending importance
    
    # Create a DataFrame to store the feature names and their importance scores
    df_imp = pd.DataFrame(dict(feature = x_train.columns[indices], # Sorted feature names
                               importance = importances[indices])) # Corresponding importance scores
    print("Feature importance calculation finished.") 
    return df_imp 


# Remove features with high correlation (above threshold)
# If there are two features with high correlation, the method will remove the features with highest sum of absolute correlation first
def remove_high_correlation(df1, threshold=0.8, importance_df=None):
    # Select only numeric columns for correlation check
    numeric_df = df1.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Set the diagonal to 0 to avoid self-correlation
    np.fill_diagonal(corr_matrix.values, 0)

    # Create a list to keep track of dropped features
    dropped_features = set()

    # Iterate through the correlation matrix and drop highly correlated features
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:  # If correlation exceeds threshold
                feature_1 = corr_matrix.columns[i]
                feature_2 = corr_matrix.columns[j]

                # Only proceed if both features are still in the DataFrame
                if feature_1 in df1.columns and feature_2 in df1.columns:
                    # Drop the feature with least importance
                    if importance_df is not None:
                        if importance_df.loc[importance_df['feature'] == feature_1, 'importance'].values[0] < \
                            importance_df.loc[importance_df['feature'] == feature_2, 'importance'].values[0]:
                            df1 = df1.drop(columns=[feature_1])
                            dropped_features.add(feature_1)
                        else:
                            df1 = df1.drop(columns=[feature_2])
                            dropped_features.add(feature_2)

    print("Dropped Features due to high correlation:", dropped_features)
    return df1

# Identifying top-performing suppliers and remove the worst performing suppliers from dataset
# In this case, we remove suppliers which have above average cost for each task
def filter_suppliers_above_average(suppliers, cost):
    # Calculate average cost per task
    average_cost = cost.groupby('Task ID')['Cost'].mean().reset_index()
    average_cost.rename(columns={'Cost': 'Average Cost'}, inplace=True)
        
    # Merge average cost data to the cost DataFrame
    cost_with_avg = cost.merge(average_cost, on='Task ID')
        
    # Filter out suppliers with costs above the average
    filtered_cost = cost_with_avg[cost_with_avg['Cost'] <= cost_with_avg['Average Cost']].drop(columns='Average Cost')
    
    # # Identify suppliers to keep
    suppliers_to_keep = filtered_cost['Supplier ID'].unique()
        
    # # Filter the suppliers DataFrame to keep only suppliers in `suppliers_to_keep`
    suppliers_filtered = suppliers[suppliers['Supplier ID'].isin(suppliers_to_keep)]
        
    # Update the suppliers dataframes   
    suppliers_filtered = suppliers_filtered
        
    # Print the count of removed pairs
    print(f"\nNumber of Suppliers left: {suppliers_filtered.shape[0]}")
        
    # Return the updated dataframes if needed
    return suppliers_filtered
