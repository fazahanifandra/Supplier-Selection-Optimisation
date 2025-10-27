# Supplier Selection Optimization (Acme Corporation)
This project develops a machine learning framework to help Acme Corporation select the most cost-efficient suppliers for daily tasks. Using three datasets (`tasks.csv`, `suppliers.csv`, `cost.csv.zip`), the model predicts supplier costs based on task and supplier features, supporting data-driven decision-making in procurement.

## Key Features
- Data cleaning, transformation, and feature selection (low variance, high correlation removal)
- Exploratory data analysis with distribution, heatmap, and error visualizations
- Regression model comparison: HGBT, Random Forest, KNN, Lasso, Ridge, SVR
- Cross-validation with **Leave-One-Group-Out (LOGO)** and **Grid Search** for optimization
- Reproducible pipeline structure to avoid data leakage
- 
## Results
- Best model: **Histogram Gradient Boosting (HGBT)**
- Lowest RMSE: **0.021 (train)** and **0.021 (test)**
- Accurately predicted 15 correct suppliers compared to actual best suppliers
- 
## Tech Stack
Python · pandas · scikit-learn · matplotlib · numpy

## Repository Structure
- `main.py` – Main script; runs data prep, EDA, and ML modeling  
- `data_preparation_1.py` – Data cleaning, scaling, and feature selection  
- `eda_2.py` – Exploratory data analysis visuals  
- `machine_learning_3.py` – Model training, cross-validation, and tuning  
- `report.py` – Optional: generate best supplier results and missing cost predictions  
- `tasks.csv`, `suppliers.csv`, `cost.csv.zip` – Original datasets  

## ▶️ How to Run
1. Ensure all datasets (task.csv, suppliers.csv, costs.csv.zip) and modules (`data_preparation_1.py`,`eda_2`, `machine_learning_3`) are in the same folder.
2. Run:  
   ```
   python main.py
   ```
3. (Optional) Run `report.py` for additional analysis.
   
