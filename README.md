# boston-house-price-prediction
A simple project to predict Boston housing prices using Python and Linear Regression.

## Dataset
- Source: [Kaggle Boston Housing Dataset](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices)
- 506 houses, 14 features (CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV)

## Features
- X: All features except MEDV (target)
- y: MEDV (Median value of owner-occupied homes in $1000s)

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Model
- Linear Regression
- Train/Test split: 80/20
- Performance:
  - Mean Squared Error (MSE): 24.29
  - R2 Score: 0.67

## Visualization
- Scatter plot of Actual vs Predicted prices
- Heatmap to show correlation between features

## How to run
1. Clone this repository
2. Make sure required libraries are installed:
`bash
pip install pandas numpy matplotlib seaborn scikit-learn
