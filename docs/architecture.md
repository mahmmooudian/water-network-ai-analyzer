# System Architecture

## Overview

Water Network AI Analyzer is designed as a desktop-based analytical application for water distribution network data analysis, machine learning-based pressure prediction, and PRV optimization.

The current version follows a modular object-oriented structure implemented inside the main application file.

The system combines:

* Data validation
* Data preprocessing
* Machine learning modeling
* Particle Swarm Optimization
* Visualization
* CSV input/output
* INP file generation
* Desktop GUI workflow management

## High-Level Architecture

```text
CSV Water Network Dataset
            │
            ▼
Data Loading & Validation
            │
            ▼
Data Preprocessing
            │
 ┌──────────┼──────────┐
 │          │          │
 ▼          ▼          ▼
PRV Data  Critical   Point After
         Point Data  Valve Data
            │
            ▼
Machine Learning Workflow
            │
            ▼
XGBoost Predictive Modeling
            │
            ▼
Critical Point Pressure Prediction

            │
            ▼

PSO Optimization Workflow
            │
            ▼
Optimized PRV Settings
            │
            ▼
Visualization & Export
```

## Core Software Components

| Component                   | Responsibility                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Constants`                 | Defines pressure limits, PRV limits, PSO parameters, XGBoost tuning parameters, and hydraulic defaults        |
| `WaterNetworkData`          | Stores PRV, critical point, point-after-valve, Deby, and original dataset partitions                          |
| `ModelResults`              | Stores trained model, scaler, predictions, metrics, feature importance, and hyperparameters                   |
| `PSOResults`                | Stores optimized PRV settings, optimized pressures, demands, score, convergence history, and final parameters |
| `WaterNetworkDataProcessor` | Handles validation, missing-value imputation, and outlier filtering                                           |
| `XGBoostTrainer`            | Trains XGBoost models, performs hyperparameter tuning, and generates predictions                              |
| `PSOOptimizer`              | Executes Particle Swarm Optimization for PRV setting search                                                   |
| `PlotGenerator`             | Creates analytical plots and visualization outputs                                                            |
| `FileHandler`               | Handles CSV loading and saving                                                                                |
| `SimpleSimulator`           | Provides a simplified pressure simulation workflow                                                            |
| `INPGenerator`              | Generates structured INP files for future hydraulic simulation integration                                    |
| `WaterNetworkGUI`           | Manages the desktop graphical user interface and connects all workflows                                       |

## Data Flow

The system begins with a CSV file containing water network operational variables.

The application validates the dataset based on expected naming conventions and separates the data into four main categories:

1. PRV data
2. Critical point data
3. Point-after-valve data
4. Deby / flow-related data

After validation, the preprocessing stage applies missing-value handling and outlier filtering.

The processed data is then passed into the machine learning and optimization workflows.

## Machine Learning Layer

The machine learning layer is based on XGBoost regression.

The training workflow includes:

* Feature scaling using `StandardScaler`
* Train/test splitting
* Randomized hyperparameter search
* Regression model training
* Multi-output regression support when multiple targets exist
* Evaluation using MAE, RMSE, and R²

The model can be used to predict critical-point pressure behavior from selected network features.

## Optimization Layer

The optimization layer uses Particle Swarm Optimization.

The PSO process searches for PRV setting combinations within defined operating constraints.

The objective function penalizes:

* Pressure values below the minimum threshold
* Pressure values above the maximum threshold
* Large changes between consecutive PRV settings

This creates a basic pressure-management optimization workflow with stability awareness.

## Visualization Layer

The visualization layer generates analytical figures for both model evaluation and optimization results.

Supported visualizations include:

* Boxplots
* Actual vs. predicted scatter plots
* Regression plots
* Line plots
* Pressure heatmaps
* PSO convergence plots
* Feature importance plots

These plots are displayed inside the desktop application and can be exported.

## GUI Layer

The GUI is implemented using Tkinter and ttk.

The interface provides user access to:

* Data loading
* Searchable data tables
* Data editing
* Data export
* Pressure prediction
* XGBoost training
* Critical-point prediction
* PSO optimization
* Simulation workflow
* Visualization windows

## Simulation Layer

The current version includes a simplified simulation component for demonstration and workflow validation.

This layer does not replace a full hydraulic solver.

The INP generator provides a foundation for future integration with hydraulic engines such as EPANET or WNTR.

## Current Architecture Limitation

The current implementation is organized primarily inside a single Python application file.

For future production-level development, the codebase can be refactored into separate modules such as:

```text
src/
├── data_processing/
├── models/
├── optimization/
├── simulation/
├── visualization/
└── gui/
```

This repository keeps the current implementation transparent while documenting the intended modular architecture for future development.
