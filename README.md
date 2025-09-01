# SHAP-Analysis-of-California-Housing-Prices-with-XGBoost
This project uses the California Housing dataset to train an XGBoost regression model and employs SHAP (SHapley Additive exPlanations) for model interpretability. It includes data loading, model training, and visualization of SHAP values via force plots, waterfalls, beeswarms, bars, and heatmaps. 

## Overview
This repository contains a Python script that demonstrates the use of XGBoost and SHAP (SHapley Additive exPlanations) to predict and interpret housing prices using the California Housing dataset. The project showcases a complete machine learning workflow, from data preparation to model interpretation, emphasizing the importance of explainability in predictive modeling.
Workflow

## Data Loading: 
The California Housing dataset is fetched using **sklearn.datasets.fetch_california_housing** and converted into a pandas DataFrame with features like median income, house age, and target median house value.

## Model Training:
An **XGBoost regressor** is trained on an 80-20 train-test split, using 100 estimators, a max depth of 4, and a learning rate of 0.1 for robust predictions.

## SHAP Explanation:
A SHAP explainer is initialized with the trained model and training data, computing SHAP values for the test set to quantify feature contributions.

## Visualization: 
Interactive **force plots (via shap.initjs()), waterfall, beeswarm, bar, and heatmap plots** are generated to visualize how features influence predictions, requiring a Jupyter notebook for full interactivity.

## Importance of SHAP Tools:
SHAP is a game-theoretic approach to explain machine learning models by assigning each feature an importance value for a given prediction. Its key benefits include:

**Interpretability:** Provides insights into which features most impact predictions, crucial for trust and transparency in models like XGBoost.
**Feature Importance:** Offers local (instance-level) and global (dataset-level) explanations, surpassing traditional methods like feature importance scores.
**Debugging:** Helps identify biases or unexpected behaviors in the model.
**Decision Support:** Assists stakeholders (e.g., policymakers) in understanding housing price drivers, enhancing decision-making.

This project serves as a practical example of integrating SHAP into a machine learning pipeline, making it valuable for data scientists and researchers.

## Requirements
**Python 3.x**
**Libraries: pandas, xgboost, shap, scikit-learn, matplotlib**
**Jupyter notebook (recommended for interactive plots)
**

## Usage

**Install dependencies:** pip install pandas xgboost shap scikit-learn matplotlib
Run the script in a Jupyter notebook environment.
Explore visualizations separately (e.g., force plot in one cell) to avoid output conflicts.
