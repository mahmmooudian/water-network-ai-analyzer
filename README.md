# 💧 Water Network AI Analyzer

### Leakage-Safe Machine Learning & PRV Optimization for Water Distribution Networks

**Water Network AI Analyzer** is a desktop-based Industrial AI application for analyzing water-distribution data, predicting critical-point pressures, and optimizing Pressure Reducing Valve (PRV) settings using **XGBoost**, **Particle Swarm Optimization (PSO)**, and optional **WNTR / EPANET hydraulic simulation**.

The project focuses on combining **data-driven machine learning**, **optimization**, and **engineering analytics** while maintaining a clear distinction between learned surrogate models and physics-based hydraulic simulation.

---

## 🚀 Key Features

### 🤖 Machine Learning

* XGBoost regression
* Single-output and multi-output regression
* Leakage-safe preprocessing
* KNN missing-value imputation
* Fold-local IQR outlier clipping
* Automated hyperparameter optimization
* RandomizedSearchCV
* Hold-out model evaluation
* Feature importance analysis
* Model persistence with Joblib

### ⚡ PRV Optimization

* Particle Swarm Optimization
* Data-driven PRV optimization
* Automatic PRV detection from dataset
* Dataset-derived PRV operating bounds
* Sequential multi-hour optimization
* Pressure constraint penalties
* Stability-aware optimization
* Reference-setting penalties
* PSO convergence analysis

### 💧 Hydraulic Analysis

* Optional WNTR integration
* Real EPANET `.inp` file loading
* EPANET hydraulic simulation through WNTR
* Pressure simulation for selected network nodes
* Hydraulic pressure visualization

### 📊 Desktop Analytics

* Interactive Tkinter GUI
* CSV dataset loading
* Editable data table
* Automatic schema detection
* Model training interface
* Critical-pressure prediction
* Feature-importance visualization
* Actual-vs-predicted plots
* PSO result visualization
* Optimization-result export
* Model save/load
* Application logging

---

# 🎯 Project Objective

Water distribution systems operate under continuously changing conditions caused by:

* Variable consumer demand
* Pressure changes
* Valve configurations
* Network topology
* Operational constraints

Determining appropriate PRV settings manually can become difficult as system complexity increases.

This project investigates how **machine learning and optimization algorithms** can support engineering decision-making by learning pressure relationships from historical operational data and searching for improved PRV configurations.

The system follows two complementary approaches:

### Data-Driven AI

Historical network measurements are used to train predictive models.

### Physics-Based Simulation

When a real EPANET `.inp` model is available, WNTR can execute hydraulic simulation independently from the ML optimization pipeline.

---

# 🧠 System Architecture

```text
                    WATER NETWORK DATA
                           │
                           ▼
                 ┌──────────────────┐
                 │ Schema Detection │
                 └────────┬─────────┘
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
   Machine Learning                  PRV Optimization
          │                                │
          ▼                                ▼
 Train/Test Split                Historical PRV Settings
          │                           + Demand
          ▼                                │
 Leakage-Safe Pipeline                   ▼
          │                       Downstream Surrogate
    ┌─────┴─────┐                          │
    │           │                          ▼
    ▼           ▼                    Particle Swarm
KNN Imputer  IQR Clipper              Optimization
    │           │                          │
    └─────┬─────┘                          ▼
          │                      Optimized PRV Settings
          ▼                                │
       XGBoost                             ▼
          │                     Predicted Downstream
          ▼                            Pressure
Critical-Point Prediction                  │
          │                                ▼
          └───────────────► Critical-Point Prediction
                                           │
                                           ▼
                                  Engineering Analysis
```

A separate optional physics-based path is also available:

```text
EPANET .INP File
      │
      ▼
    WNTR
      │
      ▼
EPANET Simulator
      │
      ▼
Hydraulic Pressure Results
```

---

# 🔒 Leakage-Safe Machine Learning Pipeline

One of the main design improvements in the current version is preventing preprocessing leakage.

The workflow is:

```text
Raw Dataset
     │
     ▼
Schema Validation
     │
     ▼
Remove Rows with Missing Targets
     │
     ▼
Train / Test Split
     │
     ▼
scikit-learn Pipeline
     │
     ├── KNNImputer
     │
     ├── IQRClipper
     │
     └── XGBoost
     │
     ▼
Cross Validation
     │
     ▼
Hyperparameter Search
     │
     ▼
Hold-Out Evaluation
```

Preprocessing components are fitted only on the appropriate training data.

This prevents information from the hold-out test set from influencing model preprocessing.

### Why No StandardScaler?

XGBoost is based on decision trees and does not require feature normalization in the same way that distance-based or gradient-based models often do.

For this reason, unnecessary feature scaling was removed from the pipeline.

---

# 🧹 Data Preprocessing

## KNN Imputation

Missing feature values are handled using:

```python
KNNImputer(
    n_neighbors=5,
    weights="distance"
)
```

The imputer is contained inside the sklearn Pipeline.

Target values are **never imputed**, because generating artificial target labels could compromise model evaluation.

---

## IQR Outlier Handling

The application includes a custom sklearn-compatible transformer:

```text
IQRClipper
```

It estimates the Interquartile Range only from training data and clips extreme feature values using learned limits.

Conceptually:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Because the transformer is inside the Pipeline, the outlier limits are recalculated independently inside cross-validation folds.

---

# 🤖 Machine Learning Model

The main predictive model is:

**XGBoost Regressor**

The application supports both:

```text
Single Target
```

and:

```text
Multiple Critical Points
```

For multiple targets the application uses:

```python
MultiOutputRegressor(XGBRegressor(...))
```

---

# 🎯 Critical-Point Prediction

The critical-point model learns the relationship:

```text
Downstream Pressure
        +
Network Demand
        │
        ▼
     XGBoost
        │
        ▼
Critical-Point Pressure
```

This allows several critical network locations to be predicted simultaneously.

---

# 🔧 Hyperparameter Optimization

Model tuning is performed using:

```text
RandomizedSearchCV
```

The search space includes:

* Number of estimators
* Maximum tree depth
* Learning rate
* Subsample ratio
* Column sampling
* Minimum child weight
* L1 regularization
* L2 regularization

Cross-validation uses shuffled K-Fold splitting with a fixed random seed for reproducibility.

---

# 📊 Evaluation Metrics

The application calculates:

### Mean Absolute Error

```text
MAE
```

### Root Mean Squared Error

```text
RMSE
```

### Coefficient of Determination

```text
R²
```

### Mean Absolute Percentage Error

```text
MAPE
```

Metrics are calculated both globally and separately for individual target variables when using multi-output regression.

---

# 📈 Model Diagnostics

The GUI provides:

* Actual vs Predicted visualization
* Feature Importance
* Per-target evaluation
* Best hyperparameters
* Training duration
* Cross-validation result
* Hold-out metrics

---

# ⚡ Particle Swarm Optimization

The project uses **Particle Swarm Optimization (PSO)** to search for PRV configurations.

Importantly:

> The PSO module is a data-driven surrogate optimizer and is not presented as a hydraulic solver.

---

## Surrogate Model A

The first learned relationship is:

```text
PRV Settings + Demand
          │
          ▼
       XGBoost
          │
          ▼
Downstream Pressure
```

Historical PRV operation is therefore used to learn how changes in PRV settings relate to downstream pressure.

---

## Surrogate Model B

A second model learns:

```text
Downstream Pressure + Demand
             │
             ▼
          XGBoost
             │
             ▼
    Critical-Point Pressure
```

Together these two models provide the predictive environment used by PSO.

---

# 🐝 PSO Workflow

For each optimization period:

```text
Initialize Particle Population
            │
            ▼
Candidate PRV Settings
            │
            ▼
Downstream Surrogate Model
            │
            ▼
Predicted Downstream Pressure
            │
            ▼
Critical-Point Surrogate
            │
            ▼
Predicted Critical Pressure
            │
            ▼
Objective Function
            │
            ▼
Update Particle Velocity
            │
            ▼
Update Particle Position
            │
            ▼
Repeat Until Convergence
```

---

# 🎯 Optimization Objective

The optimization score combines several engineering considerations.

### Pressure Constraint Penalty

Large penalties are applied when predicted pressure exceeds operational limits.

Default pressure range:

```text
10 ≤ Pressure ≤ 60
```

### Pressure Target

The optimizer encourages pressures toward a desired operating region.

Default target:

```text
30
```

### Stability Penalty

Large PRV-setting changes between sequential periods are penalized.

This helps prevent unrealistic control strategies.

### Historical Reference Penalty

The optimizer also considers historical PRV settings, discouraging unnecessarily large deviations from previously observed operation.

---

# 🔎 Automatic PRV Detection

The previous implementation used the number of PSO configuration parameters as a proxy for the number of PRVs.

That behavior has been removed.

The current system detects PRV variables directly from the dataset.

The number of optimization variables therefore corresponds to the **actual PRV columns available in the loaded network data**.

---

# 📏 PRV Bounds

PRV bounds are derived from historical values in the loaded dataset.

The application calculates approximate lower and upper operating ranges from historical quantiles and limits them using configured safety bounds.

This produces more realistic optimization ranges than applying the same arbitrary bounds to every valve.

---

# ⏱ Sequential Optimization

The application supports optimization across consecutive network observations.

By default, up to:

```text
24 periods
```

can be optimized in sequence.

The user can select the number of periods directly from the GUI.

The previous period's optimized PRV configuration is used when calculating the stability penalty for the next period.

---

# 🌊 WNTR / EPANET Integration

The application also includes an optional physics-based hydraulic simulation path.

When `wntr` is installed, the user can load a real:

```text
*.inp
```

EPANET network file.

The application then uses:

```python
wntr.network.WaterNetworkModel(...)
```

and:

```python
wntr.sim.EpanetSimulator(...)
```

to execute hydraulic simulation.

Pressure results can then be visualized inside the application.

### Important

The WNTR hydraulic simulator and data-driven PSO optimizer currently operate as separate analysis paths.

The current PSO implementation does **not** repeatedly call EPANET during every optimization iteration.

This distinction is intentional and avoids presenting the surrogate optimizer as a full hydraulic optimization engine.

---

# 🗂 Expected Dataset Schema

The application automatically attempts to detect semantic column groups.

## PRV Columns

Examples:

```text
PRV1
PRV_01
PRV_Setting_1
```

Columns containing `PRV` are interpreted as valve-setting variables unless they appear to represent identifiers or status fields.

---

## Downstream / Point-After-Valve Columns

Supported naming patterns include:

```text
*-B
*_B
after_valve
after valve
downstream
```

Examples:

```text
PRV-01-B
Valve1_B
Downstream_1
```

---

## Critical Points

Supported patterns include:

```text
J-*
critical*
critical_point*
```

Examples:

```text
J-101
J-205
Critical_Point_1
```

---

## Demand

The application recognizes names such as:

```text
P-676
Demand
Deby
Flow
Total_Demand
```

The legacy `P-676` naming convention is supported for compatibility with the original project dataset.

---

# 🖥 Desktop Application

The GUI provides dedicated areas for:

### Data

* Load CSV
* Inspect detected schema
* Browse records
* Edit cells
* Save modified datasets

### Machine Learning

* Train critical-point model
* View evaluation metrics
* Inspect per-target performance
* View feature importance
* Visualize actual vs predicted values
* Make manual predictions
* Save model
* Load model

### Optimization

* Train downstream surrogate
* Run PSO
* Select optimization horizon
* Inspect optimized PRV settings
* Analyze predicted pressures
* View convergence curves
* Export optimization results

### Hydraulic Simulation

* Load EPANET INP files
* Run WNTR / EPANET simulation
* Visualize node-pressure results

---

# 📁 Project Structure

```text
water-network-ai-analyzer/
│
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│
├── docs/
│   ├── architecture.md
│   └── methodology.md
│
└── results/
```

> The application is currently implemented as a consolidated desktop application. Modularization into a dedicated `src/` package is planned as part of the next software-engineering stage.

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/mahmmooudian/water-network-ai-analyzer.git
cd water-network-ai-analyzer
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🧪 Smoke Test

Before launching the GUI, the complete core pipeline can be tested using:

```bash
python main.py --smoke-test
```

A successful run ends with:

```text
SMOKE TEST PASSED
```

The smoke test validates:

* Missing-value handling
* Outlier handling
* XGBoost training
* Multi-output prediction
* Leakage-safe preprocessing
* Surrogate pressure prediction
* PSO execution

### Important

The smoke-test dataset is synthetic and is intended for software validation only.

Its accuracy metrics must **not** be interpreted as real-world hydraulic model performance.

---

# ▶️ Run Application

Launch the desktop application with:

```bash
python main.py
```

---

# 🔬 Typical Workflow

```text
1. Load water-network CSV
        ↓
2. Review automatically detected schema
        ↓
3. Train critical-point model
        ↓
4. Review evaluation metrics
        ↓
5. Analyze feature importance
        ↓
6. Generate predictions
        ↓
7. Run surrogate PSO optimization
        ↓
8. Analyze optimized PRV settings
        ↓
9. Export optimization results
```

For physics-based simulation:

```text
Load EPANET INP
        ↓
Run WNTR Simulation
        ↓
Analyze Pressure Results
```

---

# 💾 Model Persistence

Trained models can be saved using:

```text
.joblib
```

and loaded again without retraining.

This allows experiments and trained model states to be reused between sessions.

---

# 📤 Optimization Export

PSO results can be exported to CSV.

The exported data includes information such as:

```text
Hour
Demand
Objective
PRV Settings
Downstream Pressures
Critical-Point Pressures
Minimum Pressure
Mean Pressure
Maximum Pressure
```

---

# 🧪 Reproducibility

The project uses fixed random seeds for:

* Train/test splitting
* Cross-validation
* XGBoost
* Hyperparameter search
* PSO initialization

Default seed:

```text
42
```

This improves experiment reproducibility.

---

# ⚠️ Current Limitations

The project is an applied AI research and engineering prototype.

Current limitations include:

* Performance depends strongly on the quality and coverage of historical network data.
* Surrogate predictions should not replace engineering validation.
* Real-world calibration is dataset-specific.
* The PSO optimizer is data-driven rather than a direct hydraulic optimizer.
* WNTR simulation is currently separate from the PSO inner loop.
* Real-time SCADA / IoT ingestion is not yet implemented.
* The desktop application is currently consolidated in a single main application file.
* Extensive automated unit and integration testing is still planned.

---

# 🛣 Roadmap

Planned improvements include:

* [ ] Modular `src/` architecture
* [ ] Automated unit tests
* [ ] GitHub Actions CI pipeline
* [ ] Direct WNTR-in-the-loop optimization
* [ ] Time-series demand forecasting
* [ ] Leak and anomaly detection
* [ ] Sensor / SCADA integration
* [ ] Model monitoring
* [ ] Experiment tracking
* [ ] Web-based dashboard
* [ ] Docker support
* [ ] Benchmark datasets
* [ ] Expanded engineering validation

---

# 🧰 Technology Stack

### Language

* Python

### Machine Learning

* XGBoost
* Scikit-learn
* NumPy
* Pandas

### Optimization

* Particle Swarm Optimization

### Hydraulic Simulation

* WNTR
* EPANET

### Visualization

* Matplotlib

### Desktop UI

* Tkinter

### Model Persistence

* Joblib

---

# 🧩 Engineering Principles

The current implementation emphasizes:

* Preventing data leakage
* Reproducibility
* Explicit separation of ML and hydraulic simulation
* Honest model limitations
* Data-driven operating bounds
* Engineering-aware optimization
* Clear model evaluation
* Reusable trained models

---

# 📌 Project Status

**Active Development**

Current version:

```text
Professional Edition v5.0
```

The core ML pipeline, surrogate optimization system, desktop interface, model persistence, visualization tools, and optional WNTR simulation are operational.

Further work is focused on software modularization, automated testing, CI/CD, and deeper hydraulic integration.

---

# 👨‍💻 Author

**Amir Mohammad Mahmoudian**

AI/ML Engineer focused on:

* Machine Learning
* Industrial AI
* Intelligent Infrastructure
* Optimization
* Computer Vision
* MLOps

GitHub: [@mahmmooudian](https://github.com/mahmmooudian)

---

# 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file for details.

---

## ⭐ Support

If you find this project useful or interesting, consider starring the repository.

Contributions, technical discussions, and suggestions are welcome.
