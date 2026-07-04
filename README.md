<div align="center">

# 💧 Water Network AI Analyzer

### Industrial AI System for Water Distribution Intelligence & Optimization

A machine learning-driven engineering platform for **pressure prediction, PRV optimization, and operational intelligence in water distribution networks**.

<br>

**Industrial AI · Water Infrastructure Optimization · Predictive Modeling · Engineering Decision Systems**

<br>

</div>
---

## 📌 Overview

**Water Network AI Analyzer** is a desktop-based intelligent analytics platform designed to support the analysis and optimization of water distribution network data.

The project combines **Machine Learning**, **Particle Swarm Optimization (PSO)**, data preprocessing, predictive analytics, and interactive visualization within a unified graphical environment.

The system is designed to process operational water network datasets, analyze pressure-related variables, predict critical-point behavior, and optimize Pressure Reducing Valve (**PRV**) settings across a 24-hour operational period.

The primary objective of this project is to explore how AI-driven analytical methods can support data-informed pressure management and intelligent decision-making in water distribution systems.

---

## 🎯 Problem Statement

Water distribution networks generate complex operational data involving pressure, demand, valve settings, and critical network points.

Analyzing these variables manually can be difficult, particularly when multiple network components interact over time.

This project addresses the problem by developing an integrated analytical workflow capable of:

* Processing structured water network data
* Identifying relevant PRV and pressure variables
* Predicting network behavior using machine learning
* Optimizing PRV settings using metaheuristic optimization
* Analyzing critical-point pressure behavior
* Visualizing operational and model outputs
* Exporting analytical results for further engineering evaluation

---

## ✨ Key Features

### 🧠 Machine Learning-Based Prediction

The system integrates **XGBoost regression models** for predictive analysis of water network variables.

The machine learning pipeline includes:

* Feature standardization
* Training and testing data separation
* Randomized hyperparameter search
* Multi-output regression support
* Model performance evaluation
* Critical-point pressure prediction

Supported evaluation metrics include:

* Mean Absolute Error (**MAE**)
* Root Mean Squared Error (**RMSE**)
* R-squared (**R²**)
* Mean Absolute Percentage Error (**MAPE**)

---

### ⚙️ Particle Swarm Optimization

A custom **Particle Swarm Optimization (PSO)** implementation is used to explore optimized PRV operating settings.

The optimization process includes:

* Particle initialization within PRV operating constraints
* Velocity control
* Dynamic inertia weight adjustment
* Adaptive cognitive and social coefficients
* Personal-best and global-best tracking
* Pressure constraint penalties
* Inter-hour PRV stability penalties
* Convergence history monitoring

The optimization workflow is executed across a **24-hour operating horizon**.

---

### 💧 PRV Pressure Management

The platform supports intelligent analysis of **Pressure Reducing Valve (PRV)** data.

The optimization objective considers:

* Minimum pressure constraints
* Maximum pressure constraints
* PRV operating limits
* Pressure stability
* Changes between consecutive PRV settings

This enables the system to investigate more stable and operationally constrained valve configurations.

---

### 📊 Advanced Data Preprocessing

The preprocessing pipeline includes:

* CSV data validation
* Automatic water network column identification
* Numeric data conversion
* Missing-value handling using **KNN Imputation**
* Interquartile Range (**IQR**) based outlier filtering
* Feature scaling using **StandardScaler**
* Structured separation of network variables

The application automatically identifies:

* PRV variables
* Critical points
* Points after valves
* Deby / flow-related data

---

### 🔬 Critical Point Prediction

The platform uses machine learning to model pressure behavior at critical network points.

Point-after-valve measurements and flow-related variables can be used as model inputs to estimate critical-point pressure values.

The system also supports **multi-output regression**, allowing multiple critical points to be predicted within a unified modeling workflow.

---

### 📈 Interactive Data Visualization

The application provides multiple analytical visualization tools, including:

* Data distribution boxplots
* Actual vs. predicted scatter plots
* Regression visualization
* Multi-series pressure plots
* Simulated vs. predicted comparison plots
* Pressure heatmaps
* PSO convergence plots
* Feature importance visualization

Plots can be viewed directly inside the application and exported for further analysis.

---

### 🖥️ Desktop Graphical User Interface

The project includes a desktop GUI developed using **Tkinter**.

The interface provides access to:

* CSV data loading
* Data visualization
* Searchable network data tables
* Interactive cell editing
* Data restoration
* Deby data extraction
* Point-after-valve analysis
* Pressure prediction
* XGBoost model training
* Critical-point prediction
* PSO optimization
* Simulation workflow
* CSV result export

The GUI is designed to provide a unified environment for analytical and experimental workflows.

---

### 📁 INP File Generation

The system includes an INP generation component for creating structured water network configuration files.

Generated files may include:

* Junction definitions
* Reservoir configuration
* Pipe definitions
* PRV definitions
* Demand patterns
* PRV patterns
* Control definitions
* Hydraulic options
* Simulation timing configuration

This component provides a foundation for future integration with dedicated hydraulic simulation engines.

---

## 🏗️ System Architecture

The application follows a modular object-oriented design.

```text
Water Network Data
        │
        ▼
Data Validation
        │
        ▼
Data Preprocessing
 ├── KNN Imputation
 ├── Numeric Conversion
 └── IQR Outlier Filtering
        │
        ▼
Feature Preparation
        │
        ├───────────────────────┐
        │                       │
        ▼                       ▼
 XGBoost Modeling         PSO Optimization
        │                       │
        ▼                       ▼
Pressure Prediction       PRV Setting Search
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
          Critical Point Analysis
                    │
                    ▼
        Visualization & Reporting
                    │
                    ▼
            CSV / INP Export
```

---

## 🧩 Core Components

| Component                   | Responsibility                            |
| --------------------------- | ----------------------------------------- |
| `WaterNetworkData`          | Structured container for network datasets |
| `WaterNetworkDataProcessor` | Data validation and preprocessing         |
| `XGBoostTrainer`            | Model training, tuning, and prediction    |
| `PSOOptimizer`              | PRV setting optimization                  |
| `PlotGenerator`             | Analytical visualization                  |
| `FileHandler`               | CSV input and output operations           |
| `SimpleSimulator`           | Simplified pressure simulation layer      |
| `INPGenerator`              | Water network INP file generation         |
| `WaterNetworkGUI`           | Main desktop application interface        |

---

## 🤖 Machine Learning Workflow

The XGBoost modeling pipeline follows the workflow below:

```text
Raw Network Data
        ↓
Data Validation
        ↓
Missing Value Imputation
        ↓
Outlier Filtering
        ↓
Feature Scaling
        ↓
Train / Test Split
        ↓
Randomized Hyperparameter Search
        ↓
XGBoost Training
        ↓
Model Evaluation
        ↓
Pressure Prediction
```

Hyperparameter tuning explores combinations of:

* Number of estimators
* Maximum tree depth
* Learning rate
* Subsample ratio
* Column sampling ratio

The implementation also supports `MultiOutputRegressor` for multi-target critical-point prediction.

---

## ⚡ PSO Optimization Workflow

The PSO algorithm searches for PRV configurations within predefined operating limits.

```text
Initialize Particle Population
            ↓
Initialize Particle Velocities
            ↓
Evaluate Pressure Penalty
            ↓
Update Personal Best
            ↓
Update Global Best
            ↓
Adjust Inertia Weight
            ↓
Update Cognitive / Social Terms
            ↓
Update Velocity
            ↓
Update PRV Settings
            ↓
Apply Operating Constraints
            ↓
Repeat Until Maximum Iterations
```

The objective function considers pressure constraint violations and PRV setting stability between consecutive operating periods.

---

## 🛠️ Technologies

### Programming

* Python

### Machine Learning

* XGBoost
* Scikit-learn
* Multi-output Regression
* Randomized Hyperparameter Search

### Data Processing

* Pandas
* NumPy
* KNN Imputation
* IQR-Based Outlier Detection

### Optimization

* Particle Swarm Optimization

### Visualization

* Matplotlib
* Seaborn

### User Interface

* Tkinter
* ttk

---

## 📂 Expected Data Structure

The application expects a CSV dataset containing water network operational variables.

Column categories are identified using naming conventions.

| Data Type            | Expected Naming Pattern        |
| -------------------- | ------------------------------ |
| PRV                  | Column name containing `PRV`   |
| Critical Point       | Column name starting with `J-` |
| Point After Valve    | Column name ending with `-B`   |
| Deby / Flow Variable | `P-676`                        |

The dataset must contain at least **24 rows** for the 24-hour analytical workflow.

> Dataset naming conventions may be adapted in future versions through configurable schema mapping.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/water-network-ai-analyzer.git
```

Navigate to the project directory:

```bash
cd water-network-ai-analyzer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the main Python application:

```bash
python main.py
```

The Water Network Analyzer graphical interface will open automatically.

---

## 📊 Application Workflow

1. Load a compatible CSV water network dataset.
2. Allow the application to validate and preprocess the data.
3. Explore PRV, critical-point, point-after-valve, and Deby variables.
4. Analyze point-after-valve distributions.
5. Train the XGBoost predictive model.
6. Evaluate model performance.
7. Predict critical-point pressure behavior.
8. Run PSO-based PRV optimization.
9. Review optimization metrics and convergence behavior.
10. Export analytical results or generated INP files.

---

## ⚠️ Current Simulation Scope

The current version includes a **simplified pressure simulation layer** for analytical demonstration and workflow validation.

It should not be interpreted as a replacement for a full hydraulic solver.

The INP generation module provides a foundation for future integration with hydraulic simulation frameworks such as EPANET or WNTR.

Future versions are intended to connect the optimization workflow directly to a dedicated hydraulic simulation engine.

---

## 🔮 Future Development

Planned improvements include:

* Direct EPANET / WNTR integration
* Physics-informed optimization objectives
* Network topology-aware modeling
* Time-series validation strategies
* Advanced anomaly detection
* Leak detection workflows
* Model persistence and experiment tracking
* Automated MLOps pipelines
* Web-based analytical dashboard
* Real-time sensor data integration
* Configurable network schema mapping
* Advanced multi-objective optimization

---

## 🎯 Project Motivation

This project was developed to investigate the practical application of **Artificial Intelligence and optimization algorithms in water infrastructure systems**.

The broader goal is to explore how machine learning, operational data, and intelligent optimization can support more efficient and data-informed management of complex engineering infrastructure.

---

## 👨‍💻 Author

**Amir Mohammad Mahmoudian**

AI/ML Developer focused on Machine Learning, Deep Learning, Computer Vision, Industrial AI, and intelligent engineering systems.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you find this project interesting, consider giving the repository a star.

Contributions, technical discussions, and research collaborations are welcome.
