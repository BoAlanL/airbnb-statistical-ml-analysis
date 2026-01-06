# Airbnb Price Prediction (Statistical & Machine Learning Analysis)

## Overview
This project analyzes Airbnb listing prices using statistical modeling and machine learning techniques.
The response variable is the **log-transformed nightly price**, and predictors include listing
characteristics, host features, and location information.

The goal is to:
- Understand key drivers of Airbnb pricing
- Compare interpretability vs. predictive performance
- Evaluate linear regression against random forest models

## Methods
- Data cleaning and feature engineering
- Exploratory data analysis (EDA)
- Multiple linear regression
- Variance Inflation Factor (VIF) for multicollinearity
- 10-fold cross-validation
- Random forest modeling
- Model comparison using RMSE and MSE

## Models
- **Linear Regression**
  - Interpretable coefficients
  - Checked assumptions and multicollinearity
- **Random Forest**
  - Captures nonlinear relationships
  - Variable importance analysis

## Results
- Random forest achieves lower cross-validated MSE than linear regression
- Accommodates, property type, room type, city, and bathrooms are among the most important predictors
- Linear regression provides interpretability but underfits nonlinear patterns

## Tools
- R
- tidyverse
- caret
- randomForest
- GGally
- doParallel

- ## Data
The dataset used in this project is too large to be hosted on GitHub.

To reproduce the analysis:
1. Obtain the Airbnb dataset from the kaggle
2. Place the file in the `data/` directory
3. Rename it to `Airbnb_Data.csv`

