""" 
This module contains 3 functions for exploratory data analysis (EDA):
    1. DistributionOfTaskFeatures : generates a boxplot to visualize the distribution of task features (TF)
    2. CostOfTasksAndSuppliers : creates a heatmap to display the cost of various tasks across suppliers
    3. DistributionOfErrorsBySuppliers : generates a boxplot to show supplier error distribution with RMSE values
"""
import numpy as np
import matplotlib.pyplot as plt

# 2.1
# Function to create a boxplot graph showing distribution of task features (TF)
def DistributionOfTaskFeatures(tasks):

    # Create a boxplot for task feature columns ("TF")
    plt.figure(figsize=(14, 8)) # Set the figure size of the boxplot
    plt.boxplot([tasks[col] for col in tasks.columns if "TF" in col], labels=[col for col in tasks.columns if "TF" in col], vert=True) # Creates a list of values from columns containig "TF" to use it for the boxplot's labels

    # Add labels and title
    plt.title('Distribution of Task Features', fontsize=16, fontweight='bold') # Add title
    plt.xlabel('Task Features', fontweight='bold') # Add label for x-axis
    plt.ylabel('Value', fontweight='bold') # Add label for y-axis

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Adjust the plot
    plt.tight_layout() # Automatically adjust plot to ensure that elements inside the graph won't overlap or get cut off

    # Display the plot
    plt.show()

# 2.2
# Function to create a heatmap graph showing the cost values for various tasks (y-axis) across various suppliers (x-axis)
def CostOfTasksAndSuppliers(cost):

    # Pivot the cost data to create a matrix of cost values with Task ID as rows and Supplier ID as columns
    cost_full = cost.pivot(index='Task ID', columns='Supplier ID', values='Cost')
    
    # Sort rows and columns numerically (e.g., T1, T2, ..., S1, S2, ...)
    cost_full = cost_full.sort_index(axis=1, key=lambda x: x.str.strip('S').astype(int)) # Sort the columns of 'cost_full' data 
    cost_full = cost_full.sort_index(axis=0, key=lambda x: x.str.strip('T').astype(int)) # Sort the rows of `cost_full` data

    # Reverse the Task IDs so it will start with T1 at the top for better readability
    cost_full_reversed = cost_full.iloc[::-1]

    # Define the numeric values
    heatmap_data = cost_full_reversed.values

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(45, 30))
    heatmap = ax.pcolormesh(heatmap_data, cmap='BuPu', edgecolors='white')

    # Set the x and y ticks
    ax.set_xticks(np.arange(heatmap_data.shape[1])+0.5)
    ax.set_yticks(np.arange(heatmap_data.shape[0])+0.5)

    # Set the labels
    ax.set_xticklabels(cost_full_reversed.columns, rotation=90)  # Supplier IDs
    ax.set_yticklabels(cost_full_reversed.index)  # Task IDs

    # Add labels and title
    ax.set_title("Cost Values of Tasks and Suppliers", fontsize=30, fontweight='bold') # Add title
    ax.set_xlabel("Suppliers", fontsize=24, fontweight='bold') # Add label for x-axis
    ax.set_ylabel("Tasks", fontsize=24, fontweight='bold') # Add lavel for y-axis

    # Add a colorbar
    colorbar = fig.colorbar(heatmap)
    colorbar.ax.set_ylabel("Cost Value", fontsize=16, fontweight='bold') # Add label for the colorbar
    colorbar.ax.tick_params(labelsize=18) # Set the colorbar's ticks size

   # Adjust the plot
    plt.tight_layout() # Automatically adjust plot to ensure that elements inside the graph won't overlap or get cut off

    # Display the plot
    plt.show()

# 2.3
# Function to create a boxplot graph showing the distribution of errors for each supplier, 
# along with RMSE values
def DistributionOfErrorsBySuppliers(cost):

    # Create a copy of cost file to be able to modify the data without interfering the original data
    error = cost.copy()

    # Set index as 'Task ID'
    error.set_index('Task ID', inplace=True)

    # Find minimal cost per task and combine it to error data
    minimal_cost = cost.groupby('Task ID')['Cost'].min()
    error['Minimal Cost'] = minimal_cost

    # Calculate error of supplier being selected to perform a task
    error['Error'] = error['Cost'] - error['Minimal Cost']

    # Reset the index and pivot data for visualization
    error.reset_index(inplace=True) # Reset the index first to ease the process of pivoting data
    error_pivot = error.pivot(index='Task ID', values='Error', columns='Supplier ID') # Pivot the error data and set the error as values
    
    # Sort the Supplier ID based on ascending order
    # Originally, the order of data is S1, S10,.. S2, S20.. While we want the data to be in ascending order
    error_pivot = error_pivot[sorted(error_pivot.columns, key=lambda x: int(x.strip('S')))] 
                                                                                            
    # Calculate RMSE for each supplier                                                                                 
    rmse = np.sqrt(np.mean(error_pivot ** 2, axis=0))
    
    # Create a list of Supplier IDs along with their RMSE values for x-axis
    # - rmse.index contains the Supplier IDs
    # - rmse.values containds the rmse values of each supplier
    # - round up to 5 decimal places for better readability
    supplier_with_rmse = [
    f"{id} (RMSE: {round(value, 5)})"
    for id, value in zip(rmse.index, rmse.values)]
    
    # Create boxplot for the errors
    plt.figure(figsize=(15, 10)) # Set the figure size of the plot
    plt.boxplot(error_pivot, labels=supplier_with_rmse)

    # Add labels and title
    plt.title('Distribution of Errors by Suppliers', fontsize=16, fontweight='bold') # Add title
    plt.xlabel('Supplier ID', fontsize=12, fontweight='bold') # Add label for x-axis
    plt.ylabel('Error Value', fontsize=12, fontweight='bold') # Add label for y-axis
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Adjust the plot
    plt.tight_layout() # Automatically adjust plot to ensure that elements inside the graph won't overlap or get cut off

    # Display the plot
    plt.show()