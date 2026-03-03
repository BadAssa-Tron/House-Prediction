# House Price Prediction Model

## Overview
This project involves building a machine learning regression model to predict the median house values in California districts. The predictions are based on various demographic and geographical features, such as median income, housing age, and ocean proximity.

## Dataset
The model is trained on the **California Housing Dataset** (`California_housing_data.csv`). 
Key features used for prediction include:
* **Geographical:** `longitude`, `latitude`, `ocean_proximity`
* **Housing Details:** `housing_median_age`, `total_rooms`, `total_bedrooms`
* **Demographics:** `population`, `households`, `median_income`
* **Target Variable:** `median_house_value`

## Data Preprocessing & Pipeline
To ensure robust model training, the data went through a comprehensive preprocessing pipeline:
1. **Stratified Train-Test Split:** The data was split into training (80%) and testing (20%) sets based on stratified income categories to maintain representative data distribution.
2. **Numerical Transformations:**
   - Missing values were handled using a median `SimpleImputer`.
   - Features were scaled using `StandardScaler`.
4. **Categorical Transformations:** - The `ocean_proximity` text attribute was transformed using `OneHotEncoder` (ignoring unknown categories).

## Models Evaluated
Multiple regression models were trained and cross-validated (10 folds) to find the best predictive performance:
* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor *(Hyperparameters tuned via GridSearchCV)*
* XGBoost Regressor

## Best Model & Final Scores
After evaluating the models based on their Cross-Validation Mean RMSE, **XGBoost Regressor** was selected as the best fit for this dataset.

When evaluated on the unseen testing data, the model achieved the following scores:
* **R² Score:** `0.8375` *(The model explains ~83.75% of the variance in house prices)*
* **Test RMSE:** `46,485.40`
* **Mean Absolute Percentage Error (MAPE):** `17.87%`

## Requirements & Libraries
To run the notebook, you will need the following Python libraries installed:
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib` (for actual vs. predicted visualization)

## Installation
To run this project, you must have Python installed. You can install all necessary dependencies using the following command:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
