# Methodology

## Overview

The Water Network AI Analyzer combines data preprocessing, supervised machine learning, and Particle Swarm Optimization to create an experimental analytical workflow for water distribution network data.

The methodology is organized into four primary stages:

1. Data validation and preprocessing
2. Machine learning-based predictive modeling
3. PRV setting optimization
4. Analytical visualization and result export

The current implementation is intended as an AI-driven engineering analytics framework and a foundation for future integration with dedicated hydraulic simulation engines.

---

## 1. Data Acquisition and Input

The application accepts structured CSV datasets containing operational measurements from a water distribution network.

The current implementation identifies network variables through predefined column naming conventions.

The primary data categories are:

* Pressure Reducing Valve measurements
* Critical point pressure measurements
* Point-after-valve measurements
* Deby or flow-related measurements

A minimum of 24 observations is required for workflows that operate across a 24-hour analysis horizon.

---

## 2. Dataset Validation

Before analytical processing begins, the application validates the input dataset.

The validation process checks whether the dataset contains variables corresponding to the required analytical categories.

The current schema detection rules identify:

* PRV columns by searching for `PRV` in column names
* Critical points through the `J-` prefix
* Point-after-valve variables through the `-B` suffix
* The flow-related variable using the exact `P-676` column name

If one or more required categories cannot be identified, the dataset is rejected before model training or optimization.

---

## 3. Missing-Value Processing

Missing numerical values are handled using K-Nearest Neighbors imputation.

The application uses `KNNImputer` with five neighboring observations.

For each incomplete observation, the imputation algorithm estimates missing values using information from similar samples in the dataset.

This approach was selected to preserve relationships between operational network variables more effectively than simple constant or mean-based replacement.

---

## 4. Outlier Filtering

After missing-value processing, the application applies an Interquartile Range-based filtering strategy.

For each numerical variable, the first quartile and third quartile are calculated.

The interquartile range is defined as:

```text
IQR = Q3 - Q1
```

The lower and upper filtering boundaries are calculated as:

```text
Lower Boundary = Q1 - 1.5 × IQR

Upper Boundary = Q3 + 1.5 × IQR
```

Observations containing values outside these boundaries are excluded from the processed dataset.

The objective of this stage is to reduce the influence of extreme observations on downstream predictive modeling.

---

## 5. Feature Preparation

The feature preparation strategy depends on the analytical workflow.

For pressure-related predictive modeling, PRV measurements and the flow-related variable may be used as model inputs.

For critical-point prediction, point-after-valve measurements and the `P-676` flow-related variable are used as predictive features.

The corresponding critical-point measurements are treated as prediction targets.

For multi-target prediction, multiple critical-point columns are modeled simultaneously.

---

## 6. Feature Scaling

Before model training, input features are standardized using `StandardScaler`.

The transformation centers features around their mean and scales them using their standard deviation.

Feature scaling is applied to the input variables before the training and testing datasets are created.

The fitted scaler is retained with the trained model and reused during prediction.

---

## 7. Train and Test Split

The standardized dataset is divided into training and testing subsets.

The current implementation uses:

```text
Test Size: 20%
Random State: 42
```

The training subset is used for model fitting and hyperparameter search.

The testing subset is used to evaluate predictive performance on held-out observations.

---

## 8. XGBoost Predictive Modeling

The predictive modeling layer is based on XGBoost regression.

XGBoost uses an ensemble of gradient-boosted decision trees to model nonlinear relationships between network variables.

The implementation supports two modeling configurations.

### Single-Output Regression

When the target contains a single output variable, an `XGBRegressor` model is trained directly.

### Multi-Output Regression

When multiple critical-point targets are present, the XGBoost estimator is wrapped using `MultiOutputRegressor`.

This allows the system to train independent regression estimators for multiple critical-point outputs within a unified prediction workflow.

---

## 9. Hyperparameter Optimization

Model hyperparameters are explored using `RandomizedSearchCV`.

The current search space includes:

| Hyperparameter     | Candidate Values |
| ------------------ | ---------------- |
| `n_estimators`     | 50, 100, 150     |
| `max_depth`        | 3, 5, 7          |
| `learning_rate`    | 0.01, 0.1, 0.2   |
| `subsample`        | 0.8, 1.0         |
| `colsample_bytree` | 0.8, 1.0         |

The randomized search evaluates ten sampled hyperparameter configurations.

Three-fold cross-validation is used during the search process.

The optimization scoring metric is R-squared.

The best-performing estimator identified by the search procedure is retained as the final predictive model.

---

## 10. Model Evaluation

Predictive performance is evaluated using regression metrics.

### Mean Absolute Error

Mean Absolute Error measures the average absolute difference between actual and predicted values.

Lower MAE values indicate smaller average prediction errors.

### Root Mean Squared Error

RMSE calculates the square root of the mean squared prediction error.

Because larger errors receive greater weight, RMSE can help identify models affected by substantial prediction deviations.

### R-Squared

R-squared measures the proportion of target variance explained by the regression model.

Higher values generally indicate stronger predictive agreement with the observed target values.

### Mean Absolute Percentage Error

MAPE is calculated in selected prediction workflows to express prediction error relative to observed values.

The interpretation of MAPE requires caution when observed target values are close to zero.

---

## 11. Particle Swarm Optimization

Particle Swarm Optimization is used to search for PRV setting configurations.

Each particle represents a candidate vector of PRV operating settings.

The particle population is initialized within the configured PRV operating range.

The current PSO configuration uses:

| Parameter              | Value |
| ---------------------- | ----- |
| Number of Particles    | 30    |
| Maximum Iterations     | 50    |
| Maximum Inertia Weight | 0.9   |
| Minimum Inertia Weight | 0.4   |
| Cognitive Coefficient  | 2.0   |
| Social Coefficient     | 2.0   |

Particle velocities are also initialized and constrained to reduce excessive changes in the search space.

---

## 12. Dynamic PSO Parameters

The inertia weight decreases progressively during optimization.

The purpose of this strategy is to provide broader search behavior during the early optimization stages and more localized search behavior during later iterations.

The cognitive and social coefficients are also adjusted during the optimization process.

The cognitive contribution increases over time, while the social contribution decreases.

This implementation creates a dynamically changing balance between particle-specific search behavior and global swarm information.

---

## 13. PSO Objective Function

The optimization objective is based on pressure constraint penalties and PRV setting stability.

Pressure values below the minimum pressure threshold receive a quadratic penalty.

Pressure values above the maximum pressure threshold also receive a quadratic penalty.

Conceptually:

```text
Pressure Penalty =
    Low-Pressure Violation²
    +
    High-Pressure Violation²
```

When PRV settings from the previous operational period are available, an additional stability penalty is applied.

```text
Stability Penalty =
    0.1 × Σ(Current PRV Setting - Previous PRV Setting)²
```

The final optimization score combines pressure constraint violations and the stability penalty.

Lower objective values represent more favorable candidate configurations within the current optimization model.

---

## 14. Twenty-Four-Hour Optimization Workflow

The PRV optimization process is executed sequentially across 24 operational periods.

For each hour:

1. Point-after-valve values are extracted as the demand-related input.
2. The PSO algorithm searches for a candidate PRV configuration.
3. The best PRV settings are retained.
4. Optimized pressure estimates are calculated.
5. The current PRV settings are passed to the next hour as the previous operating configuration.

This sequential process allows the stability penalty to consider changes between consecutive operating periods.

---

## 15. Simplified Pressure Calculation

The current optimization workflow uses a simplified pressure calculation layer.

Pressure estimates are derived from:

* Mean PRV settings
* Demand-related values
* A simplified head-loss approximation
* Optional elevation values

The resulting pressure estimates are constrained within the configured minimum and maximum pressure range.

This component is intended for analytical workflow development and optimization experimentation.

It is not a replacement for a complete hydraulic simulation engine.

---

## 16. Critical-Point Prediction

Following optimization, the system can estimate pressure behavior at critical network points.

The machine learning workflow uses network variables to model relationships between operational measurements and critical-point pressures.

The current implementation supports prediction across multiple critical points.

Predictions can be displayed as:

* Structured tables
* Hourly line plots
* Comparative visualizations

Prediction results can also be exported to CSV files.

---

## 17. Visualization and Analysis

The analytical visualization layer is implemented using Matplotlib and Seaborn.

The application can generate:

* Distribution boxplots
* Actual-versus-predicted scatter plots
* Regression trend visualizations
* Multi-series hourly pressure plots
* Simulated-versus-predicted comparison plots
* Pressure heatmaps
* PSO convergence histories
* Feature importance plots

Visualization is integrated into the desktop interface using Matplotlib's Tkinter backend.

---

## 18. Result Export

The application supports exporting analytical data and predictions to CSV files.

The INP generation module can also create structured water network configuration files containing network elements and simulation settings.

The generated INP workflow is intended as a foundation for future hydraulic engine integration.

---

## 19. Methodological Limitations

The current methodology has several limitations that should be considered when interpreting results.

First, the simplified pressure calculation does not represent the full nonlinear hydraulic behavior of a real water distribution network.

Second, the current train/test split is randomized and does not explicitly preserve temporal ordering.

Third, the dataset schema relies on predefined column naming conventions.

Fourth, model persistence and experiment tracking are not yet integrated into the workflow.

Finally, optimization results should be validated using a dedicated hydraulic simulation engine before operational application.

---

## 20. Future Methodological Development

Future versions of the project are intended to explore:

* Direct EPANET or WNTR simulation integration
* Time-series cross-validation
* Physics-informed optimization
* Multi-objective PRV optimization
* Leak and anomaly detection
* Network topology-aware machine learning
* Automated experiment tracking
* Model persistence
* Real-time sensor data processing
* Uncertainty analysis
* Robust optimization under variable demand conditions

The long-term objective is to develop a more comprehensive AI-assisted analytical framework for intelligent water distribution network management.
