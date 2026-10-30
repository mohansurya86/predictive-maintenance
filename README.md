# Predictive Maintenance: Machine Failure Prediction

A machine learning project that predicts machine failures (binary task) and identifies the failure cause (multi-class task) from sensor data, using the AI4I-style predictive maintenance dataset (10,000 observations, 10 columns).

## Overview

Machine failures are rare (~3.3% of observations), so the project focuses on handling heavy class imbalance, exploring sensor relationships, and comparing several classifiers with metrics that matter for failure detection (F2, which favors recall).
Two tasks are solved:
1. **Binary task**: Will the machine fail? (`Target` = 0/1)
2. **Multi-class task**: If it fails, which failure type? (No Failure, Power Failure, Overstrain Failure, Heat Dissipation Failure, Tool Wear Failure)

## Dataset

| Column | Description |
|---|---|
| UDI | Unique row identifier (dropped) |
| Product ID | Product identifier (dropped after exploration) |
| Type | Machine quality type: L (60%), M (30%), H (10%) |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Spindle speed |
| Torque [Nm] | Torque applied |
| Tool wear [min] | Cumulative tool usage time |
| Target | Machine failure flag (0/1) |
| Failure Type | Cause of failure (or No Failure) |

## Workflow

### 1. Exploratory Analysis
- Load data, inspect types, rename columns to clean names, cast numeric features to float.
- Clean ID columns (strip first character of `Product ID`, convert to numeric) and visualize by machine `Type`.
- Pie chart of machine Type distribution.

### 2. Target Anomalies
- Removed **18** rows labeled *Random Failures* (Target = 0 but failure type set; random in nature, not predictable).
- Removed **9** ambiguous rows (Target = 1 but Failure Type = No Failure).
- Total removed: about 0.27% of the data.

### 3. Outlier Inspection
- Histograms, boxplots, and `describe()` for numeric features.
- Rotational speed and Torque show outliers; they are retained as potentially informative for failures.

### 4. Resampling with SMOTE-NC
- Failures are ~3.31% of the data. `SMOTENC` (handles the categorical `Type` feature) oversamples the four failure classes so failures reach ~20% of the dataset, with the four causes balanced equally.
- Dataset size grows by about 20.9%.

### 5. Comparison After Resampling
- Pie charts (original vs resampled failure causes) and KDE plots of feature distributions by Type, Target, and Failure Type.

### 6. Feature Scaling and Encoding
- `Type` encoded as L=0, M=1, H=2; `Failure Type` encoded 0-4.
- Numeric features standardized with `StandardScaler`.

### 7. PCA and Correlation Heatmap
- 3 principal components explain about 94.3% of variance, interpreted as **Temperature**, **Power**, and **Tool Wear** axes.
- 3D PCA plots with and without the "No Failure" class.
- Strong correlations: Air vs Process temperature (0.87), Rotational speed vs Torque (-0.85).

### 8. Metrics
- **Accuracy**, **AUC**, **F1**, and **F2** (recall-weighted, preferred for failure detection).

## Binary Task

- Split: **80% train / 10% validation / 10% test** (stratified), chosen after comparing 70/15/15, 80/10/10, and 60/20/20.
- Feature selection experiments (combining temperature features and power features) showed the original features perform best.
- **Benchmark**: Logistic Regression (test ACC 0.883, AUC 0.917, F1 0.660, F2 0.602).
- **Tuned models** (GridSearchCV, 5-fold, scoring on F2): KNN, SVC, Random Forest, XGBoost.

| Test set | KNN | SVC | RFC | XGB |
|---|---|---|---|---|
| ACC | 0.951 | 0.974 | 0.974 | 0.978 |
| AUC | 0.979 | 0.992 | 0.996 | 0.998 |
| F1 | 0.884 | 0.937 | 0.937 | 0.945 |
| F2 | 0.913 | 0.945 | 0.947 | 0.950 |

- Permutation importance: **Torque** and **Tool wear** are the most influential features.

## Multi-class Task

- **Benchmark**: Logistic Regression (one-vs-rest), test ACC 0.922, F1 0.904.
- Same tuned models and grids as the binary task (with `multi:softprob` for XGBoost).

| Test set | KNN | SVC | RFC | XGB |
|---|---|---|---|---|
| ACC | 0.965 | 0.973 | 0.974 | 0.985 |
| AUC | 0.957 | 0.995 | 0.997 | 0.999 |
| F1 | 0.965 | 0.973 | 0.975 | 0.985 |
| F2 | 0.965 | 0.973 | 0.974 | 0.985 |

- **XGBoost** is the best performer in both tasks.

## Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (preprocessing, PCA, model selection, metrics)
- imbalanced-learn (SMOTENC)
- XGBoost

## Getting Started

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

1. Place the dataset CSV in your working directory (the notebook was written for Google Colab and uses `files.upload()`; adjust `pd.read_csv(...)` for local use).
2. Run the notebook cells in order.

## Notes and Possible Improvements

- SMOTE is applied to the full dataset before the train/validation/test split, so synthetic samples may leak information into evaluation sets. For stricter evaluation, split first and resample only the training set.
- The final section (Decision Paths / tree interpretability) is a work in progress.
- Consider threshold tuning and cost-sensitive learning as alternatives to oversampling.

## Author
-Meesala Mohan Surya Sai
-Intern ID: CITS9842
