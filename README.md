<div align="center">

# 💧 Water Network AI Analyzer

### Industrial AI System for Water Distribution Intelligence & Optimization

An AI-driven engineering analytics platform for modeling pressure behavior, optimizing PRV configurations, and supporting decision-making in water distribution networks.

<br>

**Machine Learning · Optimization Systems · Water Infrastructure AI · Engineering Analytics**

</div>

---

## 📌 Overview

Water Network AI Analyzer is a desktop-based AI system designed for intelligent analysis and optimization of water distribution networks.

The system integrates machine learning, optimization algorithms, and data processing pipelines to model pressure behavior and evaluate PRV (Pressure Reducing Valve) configurations under operational constraints.

The goal of this project is to explore how AI techniques can support infrastructure-level decision-making in complex hydraulic systems.

---

## 🎯 Problem Statement

Water distribution networks are complex dynamic systems where pressure behavior is influenced by interconnected variables such as demand, valve configurations, and network topology.

Traditional analysis methods are often limited by:

- High system complexity
- Nonlinear relationships between variables
- Time-dependent operational changes
- Manual PRV tuning processes

This project addresses these challenges by introducing an AI-based framework capable of:

- Learning nonlinear pressure behavior patterns
- Predicting critical network conditions
- Optimizing PRV configurations under constraints
- Supporting sequential decision-making over time

---

## ⚙️ Core Capabilities

### Machine Learning System
- XGBoost-based regression modeling
- Multi-output prediction for critical points
- Automated hyperparameter tuning (RandomizedSearchCV)
- Robust preprocessing pipeline

### Optimization System
- Particle Swarm Optimization (PSO)
- Constraint-based PRV optimization
- Stability-aware sequential decision modeling
- 24-hour operational optimization workflow

### Data Intelligence Layer
- Automatic schema detection from raw CSV data
- Missing value handling (KNN Imputation)
- Outlier filtering (IQR method)
- Feature scaling and transformation

### Decision Support System
- Pressure prediction for network nodes
- Critical-point estimation
- PRV configuration evaluation
- Analytical reporting and visualization

---

## 🧠 System Architecture

```text
Raw Water Network Data
        │
        ▼
Data Validation & Cleaning
        │
        ▼
Preprocessing Layer
 ├── KNN Imputation
 ├── IQR Outlier Removal
 └── Feature Scaling
        │
        ▼
Machine Learning Layer
 ├── XGBoost Regression
 ├── Multi-Output Learning
 └── Hyperparameter Optimization
        │
        ▼
Optimization Layer
 ├── Particle Swarm Optimization
 ├── PRV Constraint Handling
 └── 24-Hour Sequential Optimization
        │
        ▼
Decision Layer
 ├── Pressure Prediction
 ├── Critical Point Analysis
 └── PRV Strategy Evaluation
        │
        ▼
Output Layer
 ├── Visualization
 ├── Reports
 └── Data Export (CSV / INP)
```

---

## 📊 Machine Learning Pipeline

Raw Data → Validation → KNN Imputation → IQR Outlier Removal → Scaling → Train/Test Split → XGBoost → Optimization → Evaluation → Prediction

### Metrics
- MAE
- RMSE
- R² Score

---

## ⚡ Optimization System (PSO)

- Particle Swarm Optimization for PRV tuning
- Pressure constraint penalties
- Stability penalty across time steps
- Sequential 24-hour optimization

---

## 🧹 Data Processing

- KNN Imputation for missing values
- IQR-based outlier filtering
- StandardScaler normalization
- Automatic detection of PRV / J-* / -B / P-676 columns

---

## 🖥️ Desktop Application

- Dataset loader
- Interactive table view
- Model training panel
- Prediction module
- PSO optimization module
- Visualization dashboard

---

## 📁 Project Structure

water-network-ai-analyzer/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── data/
├── docs/
└── results/

---

## 🚀 Installation

git clone https://github.com/mahmmooudian/water-network-ai-analyzer.git  
cd water-network-ai-analyzer  
python -m venv .venv  
.venv\Scripts\activate  
pip install -r requirements.txt  

---

## ▶️ Usage

python main.py

Steps:
1. Load dataset  
2. Preprocess data  
3. Train model  
4. Run prediction  
5. Optimize PRV  
6. Analyze results  

---

## ⚠️ Limitations

- Simplified hydraulic simulation
- No full EPANET integration
- Limited real-world validation
- No real-time sensor input

---

## 🔮 Roadmap

- EPANET / WNTR integration
- Real-time monitoring
- Leak detection models
- Web dashboard
- Time-series modeling

---

## 👨‍💻 Author

Amir Mohammad Mahmoudian  
GitHub: @mahmmooudian  
LinkedIn: amirmohmmadmahmoudian  
Email: mahmmooudian@gmail.com  

---

## 📄 License

MIT License
