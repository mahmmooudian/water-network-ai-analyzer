"""
Water Network Analyzer - Professional Edition v4.0
A comprehensive application for water network analysis using machine learning, 
PSO optimization, and hydraulic simulation.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, ClassVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font, simpledialog
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,   
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('water_network_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants with validation
@dataclass(frozen=True)
class Constants:
    """Application constants with validation"""
    MIN_PRESSURE: float = 10.0
    MAX_PRESSURE: float = 60.0
    MIN_PRV: float = 10.0
    MAX_PRV: float = 60.0
    NUM_HOURS: int = 24
    MAX_SAMPLES_FOR_TRAINING: int = 10000
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_NEIGHBORS_IMPUTER: int = 5
    OUTLIER_THRESHOLD: float = 1.5
    
    PSO_PARAMS: ClassVar[Dict[str, Union[int, float]]] = {
        "num_particles": 30,
        "max_iterations": 50,
        "w_max": 0.9,
        "w_min": 0.4,
        "c1": 2.0,
        "c2": 2.0
    }
    
    XGBOOST_PARAMS: ClassVar[Dict[str, Any]] = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Hydraulic parameters
    DEFAULT_ELEVATION: float = 100.0
    DEFAULT_BASE_DEMAND: float = 0.01
    DEFAULT_RESERVOIR_HEAD: float = 150.0
    DEFAULT_PIPE_LENGTH: float = 1000.0
    DEFAULT_PIPE_DIAMETER: float = 0.3
    DEFAULT_PIPE_ROUGHNESS: float = 100.0

# Data Models
@dataclass
class WaterNetworkData:
    """Data model for water network information"""
    prv_data: pd.DataFrame
    critical_point_data: pd.DataFrame
    point_after_valve_data: pd.DataFrame
    deby_data: pd.DataFrame
    original_data: Dict[str, pd.DataFrame]
    
    def validate(self) -> bool:
        """Validate the data structure"""
        required_keys = ['PRV', 'Critical Point', 'Point After Valve', 'Deby']
        for key in required_keys:
            if key not in self.original_data or self.original_data[key].empty:
                logger.error(f"Missing required data: {key}")
                return False
        return True

@dataclass
class ModelResults:
    """Container for model results"""
    model: Any
    scaler: Any
    predictions: Dict[str, Any]
    metrics: Dict[str, float]
    feature_importance: Optional[pd.DataFrame] = None
    training_time: float = 0.0
    hyperparameters: Dict[str, Any] = None

@dataclass
class PSOResults:
    """Container for PSO optimization results"""
    optimal_prv_settings: List[np.ndarray]
    optimal_pressures: List[np.ndarray]
    demands: List[np.ndarray]
    critical_point_predictions: np.ndarray
    score: float
    convergence_history: List[float]
    optimization_time: float = 0.0
    final_parameters: Dict[str, Any] = None

# Abstract Base Classes
class DataProcessor(ABC):
    """Abstract base class for data processing"""
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data"""
        pass
    
    @abstractmethod
    def validate_columns(self, data: pd.DataFrame) -> bool:
        """Validate required columns"""
        pass

class ModelTrainer(ABC):
    """Abstract base class for model training"""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelResults:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

class Optimizer(ABC):
    """Abstract base class for optimization algorithms"""
    
    @abstractmethod
    def optimize(self, *args, **kwargs) -> PSOResults:
        """Run optimization algorithm"""
        pass

# Concrete Implementations
class WaterNetworkDataProcessor(DataProcessor):
    """Concrete implementation for water network data processing"""
    
    def __init__(self, constants: Constants):
        self.constants = constants
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data: impute missing values and remove outliers"""
        logger.info("Starting data preprocessing")
        
        # Impute missing values
        imputer = KNNImputer(n_neighbors=self.constants.N_NEIGHBORS_IMPUTER)
        data_imputed = pd.DataFrame(
            imputer.fit_transform(data), 
            columns=data.columns
        )
        
        # Remove outliers using IQR
        Q1 = data_imputed.quantile(0.25)
        Q3 = data_imputed.quantile(0.75)
        IQR = Q3 - Q1
        
        outlier_mask = ~(
            (data_imputed < (Q1 - self.constants.OUTLIER_THRESHOLD * IQR)) | 
            (data_imputed > (Q3 + self.constants.OUTLIER_THRESHOLD * IQR))
        ).any(axis=1)
        
        data_clean = data_imputed[outlier_mask]
        logger.info(f"Data preprocessing completed. Removed {len(data) - len(data_clean)} outliers")
        
        return data_clean
    
    def validate_columns(self, data: pd.DataFrame) -> bool:
        """Validate required columns exist"""
        required_columns = {
            'PRV': lambda col: "prv" in col.lower(),
            'Critical Point': lambda col: col.lower().startswith("j-"),
            'Point After Valve': lambda col: col.endswith("-B"),
            'Deby': lambda col: col == 'P-676'
        }
        
        for category, validator in required_columns.items():
            if not any(validator(col) for col in data.columns):
                logger.error(f"Missing required column for {category}")
                return False
        
        return True

class XGBoostTrainer(ModelTrainer):
    """Concrete implementation for XGBoost model training"""
    
    def __init__(self, constants: Constants):
        self.constants = constants
        self.model = None
        self.scaler = None
        self.feature_names = None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> ModelResults:
        """Train XGBoost model with hyperparameter tuning"""
        import time
        start_time = time.time()
        
        logger.info("Starting XGBoost model training")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.constants.TEST_SIZE, 
            random_state=self.constants.RANDOM_STATE
        )
        
        # Hyperparameter tuning
        base_model = xgb.XGBRegressor(
            random_state=self.constants.RANDOM_STATE, 
            n_jobs=-1
        )
        
        # Handle multi-output case
        if y.ndim > 1 and y.shape[1] > 1:
            model = MultiOutputRegressor(base_model)
            param_dist = {
                f'estimator__{key}': value 
                for key, value in self.constants.XGBOOST_PARAMS.items()
            }
        else:
            model = base_model
            param_dist = self.constants.XGBOOST_PARAMS
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            random_state=self.constants.RANDOM_STATE
        )
        
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        training_time = time.time() - start_time
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': [f'Feature_{i}' for i in range(X.shape[1])],
                'importance': self.model.feature_importances_
            })
        
        logger.info(f"Model training completed in {training_time:.2f} seconds. MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return ModelResults(
            model=self.model,
            scaler=self.scaler,
            predictions={},
            metrics={'mae': mae, 'r2': r2, 'rmse': rmse},
            feature_importance=feature_importance,
            training_time=training_time,
            hyperparameters=random_search.best_params_
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class PSOOptimizer(Optimizer):
    """Concrete implementation of PSO optimization"""
    
    def __init__(self, constants: Constants):
        self.constants = constants
    
    def optimize(self, demands: np.ndarray, prev_prv_settings: Optional[np.ndarray] = None) -> PSOResults:
        """Run PSO optimization for PRV settings"""
        import time
        start_time = time.time()
        
        logger.info("Starting PSO optimization")
        
        num_prvs = len(self.constants.PSO_PARAMS)
        num_particles = self.constants.PSO_PARAMS["num_particles"]
        max_iterations = self.constants.PSO_PARAMS["max_iterations"]
        w_max = self.constants.PSO_PARAMS["w_max"]
        w_min = self.constants.PSO_PARAMS["w_min"]
        c1 = self.constants.PSO_PARAMS["c1"]
        c2 = self.constants.PSO_PARAMS["c2"]
        
        # Initialize particles and velocities
        particles = np.random.uniform(
            self.constants.MIN_PRV, 
            self.constants.MAX_PRV, 
            (num_particles, num_prvs)
        )
        v_max = (self.constants.MAX_PRV - self.constants.MIN_PRV) * 0.2
        velocities = np.random.uniform(-v_max, v_max, (num_particles, num_prvs))
        
        # Initialize personal and global best
        pbest = particles.copy()
        pbest_scores = np.array([
            self._objective_function(p, demands, prev_prv_settings) 
            for p in particles
        ])
        gbest_idx = np.argmin(pbest_scores)
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]
        
        convergence_history = [gbest_score]
        
        # PSO main loop
        for iteration in range(max_iterations):
            w = w_max - (w_max - w_min) * (iteration / max_iterations)
            c1_t = c1 * (1.0 + 0.5 * (iteration / max_iterations))
            c2_t = c2 * (1.0 - 0.5 * (iteration / max_iterations))
            
            for i in range(num_particles):
                r1, r2 = np.random.random(), np.random.random()
                velocities[i] = (
                    w * velocities[i] + 
                    c1_t * r1 * (pbest[i] - particles[i]) + 
                    c2_t * r2 * (gbest - particles[i])
                )
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.constants.MIN_PRV, self.constants.MAX_PRV)
                
                score = self._objective_function(particles[i], demands, prev_prv_settings)
                if score < pbest_scores[i]:
                    pbest[i] = particles[i].copy()
                    pbest_scores[i] = score
                    if score < gbest_score:
                        gbest = particles[i].copy()
                        gbest_score = score
            
            convergence_history.append(gbest_score)
        
        pressures = self._calculate_pressures(gbest, demands)
        optimization_time = time.time() - start_time
        
        final_parameters = {
            'num_particles': num_particles,
            'max_iterations': max_iterations,
            'w_max': w_max,
            'w_min': w_min,
            'c1': c1,
            'c2': c2,
            'final_score': gbest_score,
            'convergence_rate': (convergence_history[0] - convergence_history[-1]) / len(convergence_history)
        }
        
        logger.info(f"PSO optimization completed in {optimization_time:.2f} seconds. Best score: {gbest_score:.4f}")
        
        return PSOResults(
            optimal_prv_settings=[gbest],
            optimal_pressures=[pressures],
            demands=[demands],
            critical_point_predictions=np.array([]),
            score=gbest_score,
            convergence_history=convergence_history,
            optimization_time=optimization_time,
            final_parameters=final_parameters
        )
    
    def _objective_function(self, prv_settings: np.ndarray, demands: np.ndarray, 
                          prev_prv_settings: Optional[np.ndarray] = None) -> float:
        """Calculate objective function for PSO"""
        pressures = self._calculate_pressures(prv_settings, demands)
        penalty = 0.0
        
        # Pressure constraint penalty
        for p in pressures:
            if p < self.constants.MIN_PRESSURE:
                penalty += (self.constants.MIN_PRESSURE - p) ** 2
            elif p > self.constants.MAX_PRESSURE:
                penalty += (p - self.constants.MAX_PRESSURE) ** 2
        
        # Stability penalty
        if prev_prv_settings is not None:
            stability_penalty = np.sum((prv_settings - prev_prv_settings) ** 2) * 0.1
            penalty += stability_penalty
        
        return penalty
    
    def _calculate_pressures(self, prv_settings: np.ndarray, demands: np.ndarray, 
                           elevations: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate pressures based on PRV settings and demands"""
        if elevations is None:
            elevations = np.zeros(len(demands))
        
        pressures = np.zeros(len(demands))
        for i, demand in enumerate(demands):
            head_loss = 0.01 * demand + 0.0005 * (np.mean(prv_settings) - demand) ** 2
            pressures[i] = np.mean(prv_settings) - head_loss - elevations[i]
            pressures[i] = np.clip(pressures[i], self.constants.MIN_PRESSURE, self.constants.MAX_PRESSURE)
        
        return pressures

# Utility Classes
class PlotGenerator:
    """Generate and display high-quality plots using matplotlib and seaborn"""
    
    def __init__(self):
        # Set professional style for plots
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.0)  # تغییر از "talk" به "notebook" برای اندازه کوچکتر
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'figure.dpi': 150,  # کاهش DPI برای اندازه کوچکتر
            'savefig.dpi': 300,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'font.size': 10,  # کاهش اندازه فونت
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 1.5,  # کاهش ضخامت خطوط
            'patch.edgecolor': 'white',
            'patch.force_edgecolor': True,
            'figure.autolayout': True,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.5,  # کاهش شفافیت خطوط شبکه
        })
    
    def create_boxplot(self, data: pd.DataFrame, title: str) -> plt.Figure:
        """Create a high-quality boxplot from dataframe"""
        # تنظیم اندازه بر اساس تعداد ستون‌ها
        n_cols = len(data.columns)
        fig_width = max(8, min(n_cols * 0.8, 15))  # کاهش ضریب عرض
        fig_height = 6  # کاهش ارتفاع
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create boxplot with enhanced styling
        boxplot = sns.boxplot(
            data=data, ax=ax, palette="Set2", linewidth=1.5, fliersize=4,
            boxprops=dict(alpha=0.9), whiskerprops=dict(linestyle='--')
        )
        
        # Add title and labels
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_ylabel("Values", fontsize=12, labelpad=8)
        ax.set_xlabel("Categories", fontsize=12, labelpad=8)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_scatter_plot(self, actual: np.ndarray, predicted: np.ndarray, 
                          title: str, x_label: str, y_label: str) -> plt.Figure:
        """Create a high-quality scatter plot with regression line"""
        fig, ax = plt.subplots(figsize=(8, 6))  # کاهش اندازه
        
        # Create scatter plot
        scatter = ax.scatter(actual, predicted, alpha=0.7, edgecolors='w', s=60, 
                           color='#1f77b4', label='Predictions', linewidth=1)
        
        # Add regression line
        sns.regplot(x=actual, y=predicted, scatter=False, ax=ax, 
                   color='#ff7f0e', line_kws={"linewidth": 2, "linestyle": "--"})
        
        # Add perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.7, 
                linewidth=2, label='Perfect Prediction')
        
        # Add title and labels
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12, labelpad=8)
        ax.set_ylabel(y_label, fontsize=12, labelpad=8)
        
        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add legend
        ax.legend(loc='upper left', frameon=True, fontsize=10)
        
        # Calculate and display R²
        r2 = np.corrcoef(actual, predicted)[0, 1]**2
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_line_plot(self, data: Dict[str, List[float]], title: str, 
                        x_label: str, y_label: str) -> plt.Figure:
        """Create a high-quality line plot with multiple series"""
        n_series = len(data)
        fig_width = max(8, min(n_series * 1.2, 14))  # کاهش ضریب عرض
        fig, ax = plt.subplots(figsize=(fig_width, 6))  # کاهش ارتفاع
        
        # Color palette
        colors = sns.color_palette("husl", n_series)
        
        # Create line plot for each series
        for i, (label, values) in enumerate(data.items()):
            ax.plot(range(1, len(values)+1), values, 
                   linewidth=2, marker='o', markersize=5,  # کاهش اندازه نشانگرها
                   markeredgecolor='w', markeredgewidth=1, 
                   color=colors[i], label=label)
        
        # Add title and labels
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=12, labelpad=8)
        ax.set_ylabel(y_label, fontsize=12, labelpad=8)
        
        # Set x-axis ticks
        ax.set_xticks(range(1, len(list(data.values())[0])+1))
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add legend
        ax.legend(loc='best', frameon=True, fontsize=9, ncol=min(n_series, 3))
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_comparison_plot(self, simulated: np.ndarray, predicted: np.ndarray, 
                            node_names: List[str], title: str) -> plt.Figure:
        """Create a comparison plot for simulation vs predicted values"""
        num_nodes = len(node_names)
        fig_height = max(6, num_nodes * 1.5)  # کاهش ضریب ارتفاع
        fig = plt.figure(figsize=(10, fig_height))
        gs = gridspec.GridSpec(num_nodes, 1, figure=fig, hspace=0.4)  # کاهش فاصله
        
        for i, node in enumerate(node_names):
            ax = fig.add_subplot(gs[i])
            
            # Plot simulated values
            ax.plot(range(1, simulated.shape[0]+1), simulated[:, i], 
                   linewidth=2, label='Simulated', color='#1f77b4')
            
            # Plot predicted values
            ax.plot(range(1, simulated.shape[0]+1), predicted[:, i], 
                   linewidth=2, label='Predicted', color='#ff7f0e', linestyle='--')
            
            # Add title and labels
            ax.set_title(f'Node: {node}', fontsize=12, pad=8, fontweight='bold')
            ax.set_xlabel('Hour', fontsize=10, labelpad=6)
            ax.set_ylabel('Pressure (m)', fontsize=10, labelpad=6)
            
            # Set tick sizes
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Add legend
            ax.legend(loc='upper right', frameon=True, fontsize=9)
            
            # Set x-axis ticks
            ax.set_xticks(range(1, simulated.shape[0]+1, 2))
        
        # Add main title
        fig.suptitle(title, fontsize=16, y=0.99, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return fig
    
    def create_pressure_heatmap(self, pressures: np.ndarray, node_names: List[str], 
                             title: str) -> plt.Figure:
        """Create a heatmap of pressure values over time"""
        num_nodes = len(node_names)
        fig_width = max(10, min(pressures.shape[0] * 0.4, 15))  # کاهش ضریب عرض
        fig_height = max(6, num_nodes * 0.4)  # کاهش ضریب ارتفاع
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Create heatmap with annotations
        sns.heatmap(
            pressures.T, annot=True, fmt=".1f", cmap="YlOrRd", 
            linewidths=0.3, ax=ax, cbar_kws={'label': 'Pressure (m)', 'shrink': 0.8},
            annot_kws={"fontsize": min(9, max(7, 60 // len(node_names)))}  # تنظیم فونت متناسب با تعداد گره‌ها
        )
        
        # Set labels
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Hour', fontsize=12, labelpad=8)
        ax.set_ylabel('Node', fontsize=12, labelpad=8)
        
        # Set ticks
        ax.set_xticks(np.arange(0.5, len(pressures)+0.5, 1))
        ax.set_xticklabels(range(1, len(pressures)+1))
        ax.set_yticks(np.arange(0.5, len(node_names)+0.5, 1))
        ax.set_yticklabels(node_names, rotation=0)
        
        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_convergence_plot(self, convergence_history: List[float], title: str) -> plt.Figure:
        """Create convergence plot for PSO optimization"""
        fig, ax = plt.subplots(figsize=(8, 6))  # کاهش اندازه
        
        ax.plot(range(1, len(convergence_history)+1), convergence_history, 
               linewidth=2, marker='o', markersize=5, color='#2ca02c')
        
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12, labelpad=8)
        ax.set_ylabel('Objective Function Value', fontsize=12, labelpad=8)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        plt.tight_layout()
        
        return fig
    
    def create_feature_importance_plot(self, feature_importance: pd.DataFrame, title: str) -> plt.Figure:
        """Create feature importance plot"""
        n_features = len(feature_importance)
        fig_height = max(5, min(n_features * 0.3, 8))  # کاهش ضریب ارتفاع
        fig, ax = plt.subplots(figsize=(8, fig_height))
        
        # Sort features by importance
        feature_importance = feature_importance.sort_values('importance', ascending=True)
        
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                      color='#d62728', alpha=0.8)
        
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12, labelpad=8)
        ax.set_ylabel('Features', fontsize=12, labelpad=8)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        return fig
    
    def show_plot(self, fig: plt.Figure, title: str) -> None:
        """Display plot in a new window"""
        # Create new window
        plot_window = tk.Toplevel()
        plot_window.title(title)
        # تنظیم اندازه پنجره متناسب با رزولوشن
        plot_window.geometry("1000x700")  # کاهش اندازه پنجره
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add save button with high-quality export
        def save_plot():
            file_path = filedialog.asksaveasfilename(
                title="Save Plot",
                defaultextension=".png",
                filetypes=[("PNG Files", "*.png"), ("SVG Files", "*.svg"), ("All Files", "*.*")]
            )
            if file_path:
                fig.savefig(file_path, bbox_inches='tight', dpi=300, facecolor='white', format=file_path.split('.')[-1])
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
        
        save_button = ttk.Button(plot_window, text="Save Plot", command=save_plot)
        save_button.pack(pady=10)

class FileHandler:
    """Handle file I/O operations"""
    
    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load CSV file with validation"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    @staticmethod
    def save_csv(data: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to CSV file"""
        try:
            data.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving CSV file: {str(e)}")
            raise

class SimpleSimulator:
    """Simple simulator for demonstration purposes"""
    
    @staticmethod
    def run_simulation(inp_file: str, critical_point_ids: List[str], hours: List[int] = None) -> np.ndarray:
        """Run a simple simulation (placeholder for WNTR)"""
        logger.info("Running simple simulation (WNTR placeholder)")
        
        if hours is None:
            hours = list(range(1, 25))  # Default to all 24 hours
        
        num_hours = len(hours)
        num_points = len(critical_point_ids)
        
        # Generate random but realistic pressure values
        base_pressures = np.random.uniform(20, 50, num_points)
        
        # Create hourly variations for all 24 hours
        all_hourly_variations = np.sin(np.linspace(0, 2*np.pi, 24)) * 5
        
        # Generate pressure data for selected hours
        simulated_pressures = []
        for h in hours:
            idx = h - 1  # 0-based index
            variation = all_hourly_variations[idx]
            hour_pressures = base_pressures + variation + np.random.normal(0, 2, num_points)
            hour_pressures = np.clip(hour_pressures, 10, 60)  # Keep within realistic range
            simulated_pressures.append(hour_pressures)
        
        logger.info(f"Simple simulation completed for {num_hours} selected hours and {num_points} points")
        return np.array(simulated_pressures)

class INPGenerator:
    """Generate INP files for water network simulation"""
    
    @staticmethod
    def generate_inp_file(results: PSOResults, file_path: str, 
                         critical_point_ids: List[str], 
                         point_after_valve_ids: List[str]) -> None:
        """Generate WNTR INP file with SI units"""
        try:
            with open(file_path, 'w') as f:
                # Title section
                f.write("[TITLE]\n")
                f.write("Water Network Simulation\n")
                f.write("Generated by Water Network Analyzer\n\n")
                
                # Junctions section
                f.write("[JUNCTIONS]\n")
                f.write(";ID\tElev\tDemand\tPattern\n")
                for node in critical_point_ids + point_after_valve_ids:
                    elevation = Constants.DEFAULT_ELEVATION
                    base_demand = Constants.DEFAULT_BASE_DEMAND
                    f.write(f"{node}\t{elevation:.2f}\t{base_demand:.4f}\t;\n")
                f.write("\n")
                
                # Reservoirs section
                f.write("[RESERVOIRS]\n")
                f.write(";ID\tHead\n")
                f.write(f"RES\t{Constants.DEFAULT_RESERVOIR_HEAD:.2f}\n\n")
                
                # Pipes section
                f.write("[PIPES]\n")
                f.write(";ID\tNode1\tNode2\tLength\tDiameter\tRoughness\n")
                all_nodes = critical_point_ids + point_after_valve_ids
                for i in range(len(all_nodes) - 1):
                    length = Constants.DEFAULT_PIPE_LENGTH
                    diameter = Constants.DEFAULT_PIPE_DIAMETER
                    roughness = Constants.DEFAULT_PIPE_ROUGHNESS
                    f.write(f"P{i+1}\t{all_nodes[i]}\t{all_nodes[i+1]}\t{length:.2f}\t{diameter:.3f}\t{roughness:.1f}\n")
                f.write("\n")
                
                # Valves section
                f.write("[VALVES]\n")
                f.write(";ID\tNode1\tNode2\tDiameter\tType\tSetting\tMinorLoss\n")
                num_prvs = len(results.optimal_prv_settings[0])
                for i in range(num_prvs):
                    prv_id = f"PRV{i+1}"
                    node1 = all_nodes[i % len(all_nodes)]
                    node2 = all_nodes[(i + 1) % len(all_nodes)]
                    avg_setting = np.mean([settings[i] for settings in results.optimal_prv_settings])
                    diameter = Constants.DEFAULT_PIPE_DIAMETER
                    minor_loss = 0.0
                    f.write(f"{prv_id}\t{node1}\t{node2}\t{diameter:.3f}\tPRV\t{avg_setting:.2f}\t{minor_loss:.2f}\n")
                f.write("\n")
                
                # Patterns section
                f.write("[PATTERNS]\n")
                f.write(";ID\tMultipliers\n")
                f.write("DEMAND\t" + " ".join(["1.0"] * Constants.NUM_HOURS) + "\n")
                
                for i in range(num_prvs):
                    f.write(f"PRV{i+1}\t")
                    multipliers = [f"{settings[i]:.2f}" for settings in results.optimal_prv_settings]
                    f.write(" ".join(multipliers) + "\n")
                f.write("\n")
                
                # Controls section
                f.write("[CONTROLS]\n")
                f.write(";ID\tCondition\tAction\n")
                for i in range(num_prvs):
                    prv_id = f"PRV{i+1}"
                    for hour in range(Constants.NUM_HOURS):
                        f.write(f"C{prv_id}_{hour}\tAT TIME {hour} CLOCKTIME\tLINK {prv_id} SETTING PRV{i+1}\n")
                f.write("\n")
                
                # Options section with SI units
                f.write("[OPTIONS]\n")
                f.write("UNITS\tSI\n")  # Use SI units
                f.write("HEADLOSS\tH-W\n")
                f.write("QUALITY\tNONE\n")
                f.write("VISCOSITY\t1.0\n")
                f.write("TRIALS\t40\n")
                f.write("ACCURACY\t0.001\n")
                f.write("TOLERANCE\t0.01\n")
                f.write("EMITTER EXPONENT\t0.5\n")
                f.write("DEMAND MULTIPLIER\t1.0\n")
                f.write("\n")
                
                # Times section
                f.write("[TIMES]\n")
                f.write("DURATION\t24:00\n")
                f.write("HYDRAULIC TIMESTEP\t1:00\n")
                f.write("QUALITY TIMESTEP\t0:05\n")
                f.write("REPORT TIMESTEP\t1:00\n")
                f.write("REPORT START\t0:00\n")
                f.write("PATTERN TIMESTEP\t1:00\n")
                f.write("\n")
                
                # Report section
                f.write("[REPORT]\n")
                f.write("PAGESIZE\t60\n")
                f.write("FILE\tYES\n")
                f.write("\n")
                
                # End section
                f.write("[END]\n")
            
            logger.info(f"INP file generated successfully: {file_path}")
            
        except Exception as e:
            logger.error(f"Error generating INP file: {str(e)}")
            raise

# GUI Components
class WaterNetworkGUI:
    """Main GUI application for water network analysis"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Water Network Analyzer - Professional Edition v4.0")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize components
        self.constants = Constants()
        self.data_processor = WaterNetworkDataProcessor(self.constants)
        self.model_trainer = XGBoostTrainer(self.constants)
        self.optimizer = PSOOptimizer(self.constants)
        self.plot_generator = PlotGenerator()
        self.file_handler = FileHandler()
        self.simulator = SimpleSimulator()
        self.inp_generator = INPGenerator()
        
        # Data storage
        self.water_network_data: Optional[WaterNetworkData] = None
        self.model_results: Optional[ModelResults] = None
        self.pso_results: Optional[PSOResults] = None
        
        # GUI elements
        self.treeviews: Dict[str, ttk.Treeview] = {}
        self.search_vars: Dict[str, tk.StringVar] = {}
        self.progress_bar: Optional[ttk.Progressbar] = None
        
        # Setup GUI
        self._setup_gui()
        self._setup_menu()
        self._setup_controls()
        
        # Configure styles
        self._configure_styles()
        
        logger.info("GUI initialized successfully")
    
    def _setup_gui(self):
        """Setup main GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Notebook for tabs
        self.tabs = ttk.Notebook(main_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_menu(self):
        """Setup application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open CSV File", command=self.load_file)
        file_menu.add_command(label="Save CSV File", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Extract Deby Data", command=self.extract_deby)
        analysis_menu.add_command(label="Analyze Point After Valve", command=self.analyze_point_after_valve)
        analysis_menu.add_command(label="Predict Pressures", command=self.predict_pressures)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Train XGBoost Model", command=self.train_xgboost_model)
        model_menu.add_command(label="Predict Critical Points", command=self.predict_critical_points)
        
        # Optimization menu
        optimization_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Optimization", menu=optimization_menu)
        optimization_menu.add_command(label="Optimize with PSO", command=self.optimize_with_pso)
        optimization_menu.add_command(label="Run Simulation", command=self.run_simulation)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _setup_controls(self):
        """Setup control buttons"""
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        buttons = [
            ("Load Data", self.load_file, "Load water network data from CSV"),
            ("Save Data", self.save_file, "Save current data to CSV"),
            ("Undo Changes", self.undo_changes, "Revert all changes to original data"),
            ("Extract Deby", self.extract_deby, "Extract Deby data to CSV file"),
            ("Analyze Valve", self.analyze_point_after_valve, "Analyze Point After Valve data"),
            ("Predict Pressures", self.predict_pressures, "Predict pressures using XGBoost"),
            ("Train Model", self.train_xgboost_model, "Train XGBoost model"),
            ("PSO Optimize", self.optimize_with_pso, "Optimize PRV settings with PSO"),
            ("Run Simulation", self.run_simulation, "Run hydraulic simulation")
        ]
        
        for text, command, tooltip in buttons:
            btn = ttk.Button(control_frame, text=text, command=command)
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            self._add_tooltip(btn, tooltip)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.root, 
            mode='indeterminate', 
            length=300
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
    
    def _configure_styles(self):
        """Configure GUI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure fonts
        self.custom_font = font.Font(family="Helvetica", size=10)
        self.title_font = font.Font(family="Helvetica", size=12, weight="bold")
        
        # Configure styles
        style.configure("TNotebook.Tab", padding=[12, 8], font=('Helvetica', 11))
        style.configure("TButton", padding=8, font=self.custom_font)
        style.configure("TLabel", font=self.custom_font, background="#f0f0f0")
        style.configure("Treeview", font=self.custom_font, rowheight=25)
        style.configure("Treeview.Heading", font=self.custom_font, weight="bold")
        style.configure("Title.TLabel", font=self.title_font, background="#f0f0f0")
    
    def _add_tooltip(self, widget: tk.Widget, text: str):
        """Add tooltip to a widget"""
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.withdraw()
        
        label = ttk.Label(
            tooltip, 
            text=text, 
            background="#ffffe0", 
            relief='solid', 
            borderwidth=1, 
            font=('Helvetica', 9)
        )
        label.pack()
        
        def show_tooltip(event):
            tooltip.wm_geometry(
                f"+{widget.winfo_rootx()+20}+{widget.winfo_rooty()+30}"
            )
            tooltip.deiconify()
        
        def hide_tooltip(event):
            tooltip.withdraw()
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
    
    def _update_status(self, message: str):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def _start_progress(self, message: str):
        """Start progress bar with message"""
        if self.progress_bar:
            self.progress_bar.start()
            self._update_status(message)
    
    def _stop_progress(self):
        """Stop progress bar"""
        if self.progress_bar:
            self.progress_bar.stop()
            self._update_status("Ready")
    
    def _show_accuracy_message(self, title: str, metrics: Dict[str, float], 
                              additional_info: Optional[str] = None):
        """Display a message box with accuracy metrics"""
        message = f"Accuracy Metrics for {title}:\n\n"
        
        for metric, value in metrics.items():
            if metric == 'mae':
                message += f"Mean Absolute Error (MAE): {value:.4f}\n"
            elif metric == 'r2':
                message += f"R-squared (R²): {value:.4f}\n"
            elif metric == 'rmse':
                message += f"Root Mean Square Error (RMSE): {value:.4f}\n"
            elif metric == 'mape':
                message += f"Mean Absolute Percentage Error (MAPE): {value:.2f}%\n"
            elif metric == 'penalty':
                message += f"Optimization Penalty: {value:.4f}\n"
            elif metric == 'avg_pressure':
                message += f"Average Pressure: {value:.4f} meters\n"
            elif metric == 'pressure_std':
                message += f"Pressure Standard Deviation: {value:.4f}\n"
            elif metric == 'training_time':
                message += f"Training Time: {value:.2f} seconds\n"
            elif metric == 'optimization_time':
                message += f"Optimization Time: {value:.2f} seconds\n"
            elif metric == 'convergence_rate':
                message += f"Convergence Rate: {value:.4f}\n"
            else:
                message += f"{metric}: {value:.4f}\n"
        
        if additional_info:
            message += f"\nAdditional Information:\n{additional_info}"
        
        messagebox.showinfo("Accuracy Report", message)
    
    def load_file(self):
        """Load and process CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self._start_progress("Loading data...")
            
            # Load data
            raw_data = self.file_handler.load_csv(file_path)
            
            # Validate data
            if len(raw_data) < self.constants.NUM_HOURS:
                raise ValueError(
                    f"CSV file must contain at least {self.constants.NUM_HOURS} rows for 24-hour analysis"
                )
            
            if not self.data_processor.validate_columns(raw_data):
                raise ValueError("CSV must contain PRV, Point After Valve, Critical Point, and Deby (P-676) columns")
            
            # Preprocess data
            processed_data = self.data_processor.preprocess(raw_data)
            
            # Identify columns
            prv_cols = [col for col in processed_data.columns if "prv" in col.lower()]
            critical_point_cols = [col for col in processed_data.columns if col.lower().startswith("j-")]
            point_after_valve_cols = [col for col in processed_data.columns if col.endswith("-B")]
            deby_col = ['P-676'] if 'P-676' in processed_data.columns else []
            
            # Create data model
            self.water_network_data = WaterNetworkData(
                prv_data=processed_data[prv_cols],
                critical_point_data=processed_data[critical_point_cols],
                point_after_valve_data=processed_data[point_after_valve_cols],
                deby_data=processed_data[deby_col],
                original_data={
                    "PRV": processed_data[prv_cols].copy(),
                    "Critical Point": processed_data[critical_point_cols].copy(),
                    "Point After Valve": processed_data[point_after_valve_cols].copy(),
                    "Deby": processed_data[deby_col].copy()
                }
            )
            
            # Create tabs
            self._create_data_tabs()
            
            self._stop_progress()
            messagebox.showinfo("Success", "CSV file loaded successfully")
            self._update_status(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            self._stop_progress()
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            logger.error(f"Load file error: {str(e)}")
    
    def _create_data_tabs(self):
        """Create tabs for each data category"""
        # Clear existing tabs
        for tab in self.tabs.tabs():
            self.tabs.forget(tab)
        
        self.treeviews.clear()
        self.search_vars.clear()
        
        # Create tabs for each data category
        for name, df in self.water_network_data.original_data.items():
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text=name)
            
            # Create search frame
            search_frame = ttk.Frame(tab)
            search_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(
                search_frame, 
                text=f"Search {name}:", 
                font=self.custom_font
            ).pack(side=tk.LEFT)
            
            search_var = tk.StringVar()
            self.search_vars[name] = search_var
            
            search_entry = ttk.Entry(
                search_frame, 
                textvariable=search_var, 
                font=self.custom_font
            )
            search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            search_var.trace_add('write', lambda *args, n=name: self.filter_treeview(n))
            
            # Create treeview
            tree_frame = ttk.Frame(tab)
            tree_frame.pack(fill=tk.BOTH, expand=True)
            
            x_scroll = ttk.Scrollbar(tree_frame, orient="horizontal")
            y_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
            
            tree = ttk.Treeview(
                tree_frame, 
                columns=list(df.columns), 
                show='headings', 
                selectmode='browse',
                xscrollcommand=x_scroll.set, 
                yscrollcommand=y_scroll.set
            )
            
            self.treeviews[name] = tree
            
            x_scroll.config(command=tree.xview)
            y_scroll.config(command=tree.yview)
            
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Configure columns
            for col in df.columns:
                tree.heading(col, text=col)
                max_width = max([len(str(x)) for x in df[col]] + [len(col)]) * 10
                tree.column(col, width=max_width, anchor='center', stretch=True)
            
            # Insert data
            for _, row in df.iterrows():
                tree.insert("", "end", values=list(row))
            
            # Bind double-click for editing
            tree.bind('<Double-1>', self.on_double_click)
    
    def filter_treeview(self, element_type: str):
        """Filter treeview based on search text"""
        search_text = self.search_vars[element_type].get().lower()
        tree = self.treeviews[element_type]
        
        tree.delete(*tree.get_children())
        
        for _, row in self.water_network_data.original_data[element_type].iterrows():
            if any(search_text in str(item).lower() for item in row):
                tree.insert("", "end", values=list(row))
    
    def on_double_click(self, event):
        """Handle double-click on treeview for editing"""
        tree = event.widget
        region = tree.identify("region", event.x, event.y)
        
        if region != "cell":
            return
        
        row_id = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        
        x, y, width, height = tree.bbox(row_id, column)
        value = tree.set(row_id, column)
        
        edit_window = tk.Toplevel(self.root)
        edit_window.geometry(
            f"{width+10}x{height+10}+{self.root.winfo_rootx()+x}+{self.root.winfo_rooty()+y}"
        )
        edit_window.overrideredirect(True)
        
        entry = ttk.Entry(edit_window, font=self.custom_font)
        entry.insert(0, value)
        entry.focus()
        entry.select_range(0, tk.END)
        entry.pack(padx=2, pady=2)
        
        def save_edit(event=None):
            new_value = entry.get()
            col = int(column.replace('#', '')) - 1
            active_tab = self.tabs.tab(self.tabs.select(), "text")
            col_name = self.water_network_data.original_data[active_tab].columns[col]
            
            try:
                value = float(new_value)
                
                if col_name.lower().startswith("prv") and not (
                    self.constants.MIN_PRESSURE <= value <= self.constants.MAX_PRESSURE
                ):
                    messagebox.showerror(
                        "Error", 
                        f"Value in '{col_name}' must be between {self.constants.MIN_PRESSURE} and {self.constants.MAX_PRESSURE}"
                    )
                    entry.focus()
                    return
                
                if col_name.lower().startswith("p-") or col_name.lower().startswith("j-"):
                    try:
                        value = float(new_value)
                    except ValueError:
                        messagebox.showerror("Error", f"Value in '{col_name}' must be numeric")
                        entry.focus()
                        return
                
            except ValueError:
                messagebox.showerror("Error", f"Value in '{col_name}' must be numeric")
                entry.focus()
                return
            
            values = list(tree.item(row_id, 'values'))
            values[col] = new_value
            tree.item(row_id, values=values)
            edit_window.destroy()
        
        entry.bind('<Return>', save_edit)
        entry.bind('<FocusOut>', save_edit)
    
    def undo_changes(self):
        """Revert all changes to original data"""
        if not self.water_network_data:
            messagebox.showerror("Error", "No data loaded")
            return
        
        for name, tree in self.treeviews.items():
            tree.delete(*tree.get_children())
            for _, row in self.water_network_data.original_data[name].iterrows():
                tree.insert("", "end", values=list(row))
        
        messagebox.showinfo("Success", "All changes have been reverted")
        self._update_status("Changes reverted to original data")
    
    def save_file(self):
        """Save all data to a CSV file"""
        if not self.water_network_data:
            messagebox.showerror("Error", "No data loaded")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save CSV File",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Combine all dataframes
            combined_df = pd.concat([
                df for df in self.water_network_data.original_data.values() 
                if not df.empty
            ], axis=1)
            
            self.file_handler.save_csv(combined_df, file_path)
            messagebox.showinfo("Success", f"File saved to {file_path}")
            self._update_status(f"Saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")
            logger.error(f"Save file error: {str(e)}")
    
    def extract_deby(self):
        """Extract Deby data to CSV file"""
        if not self.water_network_data or self.water_network_data.deby_data.empty:
            messagebox.showerror("Error", "Deby data not found")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Deby Data",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.file_handler.save_csv(self.water_network_data.deby_data, file_path)
            messagebox.showinfo("Success", f"Deby data saved to {file_path}")
            self._update_status(f"Deby data saved: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Deby data: {str(e)}")
            logger.error(f"Extract Deby error: {str(e)}")
    
    def analyze_point_after_valve(self):
        """Analyze Point After Valve data and show statistics"""
        if not self.water_network_data or self.water_network_data.point_after_valve_data.empty:
            messagebox.showerror("Error", "No Point After Valve data found")
            return
        
        try:
            data = self.water_network_data.point_after_valve_data
            
            # Create statistics window
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Point After Valve Analysis")
            stats_window.geometry("800x600")
            stats_window.configure(bg="#f0f0f0")
            
            # Title
            ttk.Label(
                stats_window, 
                text="Point After Valve Analysis", 
                font=self.title_font
            ).pack(pady=10)
            
            # Statistics text
            stats_text = tk.Text(stats_window, wrap=tk.WORD, font=self.custom_font)
            stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            stats_text.insert(tk.END, "Descriptive Statistics:\n\n")
            stats_text.insert(tk.END, data.describe().to_string())
            
            # Create and show boxplot
            box_plot = self.plot_generator.create_boxplot(
                data, 
                "Point After Valve Data Distribution"
            )
            self.plot_generator.show_plot(box_plot, "Point After Valve Boxplot")
            
            # Save data button
            def save_data():
                file_path = filedialog.asksaveasfilename(
                    title="Save Point After Valve Data",
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
                )
                
                if file_path:
                    try:
                        self.file_handler.save_csv(data, file_path)
                        messagebox.showinfo("Success", f"Data saved to {file_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            
            ttk.Button(
                stats_window, 
                text="Save Data", 
                command=save_data
            ).pack(pady=10)
            
            self._update_status("Point After Valve analysis completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze Point After Valve: {str(e)}")
            logger.error(f"Analyze Point After Valve error: {str(e)}")
    
    def predict_pressures(self):
        """Predict pressures using XGBoost model and show accuracy"""
        if not self.water_network_data:
            messagebox.showerror("Error", "No data loaded")
            return
        
        try:
            self._start_progress("Predicting pressures...")
            
            # Prepare data
            prv_data = self.water_network_data.prv_data.apply(pd.to_numeric, errors='coerce')
            deby_data = self.water_network_data.deby_data.apply(pd.to_numeric, errors='coerce')
            
            # Preprocess data
            prv_data = pd.DataFrame(
                KNNImputer(n_neighbors=self.constants.N_NEIGHBORS_IMPUTER).fit_transform(prv_data),
                columns=prv_data.columns
            )
            deby_data = pd.DataFrame(
                KNNImputer(n_neighbors=self.constants.N_NEIGHBORS_IMPUTER).fit_transform(deby_data),
                columns=deby_data.columns
            )
            
            data = pd.concat([prv_data, deby_data], axis=1, join='inner')
            
            if data.empty:
                raise ValueError("No overlapping data between PRV and Deby")
            
            # Remove outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            data_no_outliers = data[~(
                (data < (Q1 - self.constants.OUTLIER_THRESHOLD * IQR)) | 
                (data > (Q3 + self.constants.OUTLIER_THRESHOLD * IQR))
            ).any(axis=1)]
            
            if data_no_outliers.empty or data_no_outliers.shape[1] < 2:
                raise ValueError("Insufficient data for pressure prediction")
            
            X = data_no_outliers.drop(columns=['P-676']).to_numpy(dtype=float)
            y = data_no_outliers['P-676'].to_numpy(dtype=float)
            
            # Train model
            self.model_results = self.model_trainer.train(X, y)
            
            # Make predictions
            y_pred = self.model_results.model.predict(X)
            
            # Calculate additional accuracy metrics
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            # Create and show scatter plot
            scatter_plot = self.plot_generator.create_scatter_plot(
                y, y_pred,
                "Actual vs Predicted Pressures",
                "Actual Pressure",
                "Predicted Pressure"
            )
            self.plot_generator.show_plot(scatter_plot, "Pressure Predictions")
            
            # Show accuracy metrics
            metrics = {
                'mae': self.model_results.metrics['mae'],
                'r2': self.model_results.metrics['r2'],
                'rmse': rmse,
                'mape': mape,
                'training_time': self.model_results.training_time
            }
            
            additional_info = (
                f"Data Points: {len(y)}\n"
                f"Pressure Range: {y.min():.2f} - {y.max():.2f} meters\n"
                f"Prediction Range: {y_pred.min():.2f} - {y_pred.max():.2f} meters\n"
                f"Best Hyperparameters: {self.model_results.hyperparameters}"
            )
            
            self._show_accuracy_message("XGBoost Model Performance", metrics, additional_info)
            
            # Show feature importance if available
            if self.model_results.feature_importance is not None:
                importance_plot = self.plot_generator.create_feature_importance_plot(
                    self.model_results.feature_importance,
                    "Feature Importance"
                )
                self.plot_generator.show_plot(importance_plot, "Feature Importance")
            
            self._stop_progress()
            self._update_status("Pressure prediction completed")
            
        except Exception as e:
            self._stop_progress()
            messagebox.showerror("Error", f"Failed to predict pressures: {str(e)}")
            logger.error(f"Pressure prediction error: {str(e)}")
    
    def train_xgboost_model(self):
        """Train XGBoost model with Point After Valve features and Critical Point target"""
        if not self.water_network_data:
            messagebox.showerror("Error", "No data loaded")
            return
        
        # Create input window
        input_window = tk.Toplevel(self.root)
        input_window.title("Train XGBoost Model")
        input_window.geometry("500x700")
        input_window.configure(bg="#f0f0f0")
        
        ttk.Label(
            input_window, 
            text="Train XGBoost Model", 
            font=self.title_font
        ).pack(pady=10)
        
        # Point After Valve inputs
        ttk.Label(
            input_window, 
            text="Enter values for Point After Valve:", 
            font=self.custom_font
        ).pack(pady=10)
        
        point_after_valve_inputs = {}
        for col in self.water_network_data.point_after_valve_data.columns:
            frame = ttk.Frame(input_window)
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            ttk.Label(
                frame, 
                text=f"{col} ({self.constants.MIN_PRESSURE}-{self.constants.MAX_PRESSURE}):", 
                font=self.custom_font
            ).pack(side=tk.LEFT)
            
            entry_var = tk.StringVar(
                value=str(self.water_network_data.point_after_valve_data[col].mean())
            )
            ttk.Entry(
                frame, 
                textvariable=entry_var, 
                font=self.custom_font
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            point_after_valve_inputs[col] = entry_var
        
        # Deby input
        ttk.Label(
            input_window, 
            text="Enter Deby:", 
            font=self.custom_font
        ).pack(pady=10)
        
        deby_var = tk.StringVar(
            value=str(self.water_network_data.deby_data['P-676'].mean())
        )
        ttk.Entry(
            input_window, 
            textvariable=deby_var, 
            font=self.custom_font
        ).pack(pady=5, padx=10, fill=tk.X)
        
        # Target selection
        ttk.Label(
            input_window, 
            text="Select Target from Critical Point:", 
            font=self.custom_font
        ).pack(pady=10)
        
        target_var = tk.StringVar()
        critical_point_cols = list(self.water_network_data.critical_point_data.columns)
        target_dropdown = ttk.Combobox(
            input_window, 
            textvariable=target_var, 
            values=critical_point_cols, 
            font=self.custom_font
        )
        target_dropdown.set(critical_point_cols[0])
        target_dropdown.pack(pady=5, padx=10)
        
        def train_model():
            try:
                # Validate inputs
                point_after_valve_values = {}
                for col, var in point_after_valve_inputs.items():
                    value = var.get()
                    try:
                        value = float(value)
                        if not self.constants.MIN_PRESSURE <= value <= self.constants.MAX_PRESSURE:
                            messagebox.showerror(
                                "Error", 
                                f"Value for {col} must be between {self.constants.MIN_PRESSURE} and {self.constants.MAX_PRESSURE}"
                            )
                            return
                        point_after_valve_values[col] = value
                    except ValueError:
                        messagebox.showerror("Error", f"Value for {col} must be numeric")
                        return
                
                try:
                    deby_value = float(deby_var.get())
                except ValueError:
                    messagebox.showerror("Error", "Deby value must be numeric")
                    return
                
                selected_target = target_var.get()
                if not selected_target or selected_target not in critical_point_cols:
                    messagebox.showerror("Error", "Select a valid target column")
                    return
                
                input_window.destroy()
                self._run_xgb_training(
                    list(point_after_valve_values.keys()), 
                    selected_target, 
                    point_after_valve_values, 
                    deby_value
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to train model: {str(e)}")
                logger.error(f"Train model error: {str(e)}")
        
        ttk.Button(
            input_window, 
            text="Train Model", 
            command=train_model
        ).pack(pady=15)
    
    def _run_xgb_training(self, selected_features: List[str], selected_target: str, 
                         point_after_valve_values: Dict[str, float], deby_value: float):
        """Run XGBoost training with specified parameters"""
        try:
            self._start_progress("Training XGBoost model...")
            
            # Prepare data
            point_after_valve_data = self.water_network_data.point_after_valve_data[selected_features].copy()
            deby_data = self.water_network_data.deby_data.copy()
            critical_point_data = self.water_network_data.critical_point_data.copy()
            
            # Reset index and merge data
            point_after_valve_data = point_after_valve_data.reset_index().rename(columns={'index': 'id'})
            deby_data = deby_data.reset_index().rename(columns={'index': 'id'})
            critical_point_data = critical_point_data.reset_index().rename(columns={'index': 'id'})
            
            merged_data = pd.merge(point_after_valve_data, deby_data, on='id', how='inner')
            merged_data = pd.merge(merged_data, critical_point_data, on='id', how='inner')
            
            if merged_data.empty:
                raise ValueError("No overlapping data found after merging")
            
            features = selected_features + ['P-676']
            targets = list(self.water_network_data.critical_point_data.columns)
            
            X = merged_data[features].to_numpy(dtype=float)
            y = merged_data[targets].to_numpy(dtype=float)
            
            if X.shape[0] < 10 or y.shape[0] < 10:
                raise ValueError("Insufficient data for training (minimum 10 samples required)")
            
            # Train model
            self.model_results = self.model_trainer.train(X, y)
            
            # Make predictions
            input_data = [point_after_valve_values[col] for col in selected_features] + [deby_value]
            predictions = self.model_trainer.predict(np.array([input_data]))[0]
            
            self.model_results.predictions = {
                target: pred for target, pred in zip(targets, predictions)
            }
            
            # Display predictions
            self._display_predictions(
                self.model_results.predictions, 
                "Predicted Critical Point Values"
            )
            
            self._stop_progress()
            messagebox.showinfo(
                "Success", 
                f"XGBoost training completed in {self.model_results.training_time:.2f} seconds"
            )
            self._update_status("XGBoost model training completed")
            
        except Exception as e:
            self._stop_progress()
            messagebox.showerror("Error", f"Failed to train XGBoost: {str(e)}")
            logger.error(f"Error in run_xgb_training: {str(e)}")
    
    def _display_predictions(self, predictions: Dict[str, float], title: str):
        """Display predictions in a new window"""
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title(title)
        prediction_window.geometry("500x400")
        prediction_window.configure(bg="#f0f0f0")
        
        ttk.Label(
            prediction_window, 
            text=title, 
            font=self.title_font
        ).pack(pady=10)
        
        # Create treeview
        tree_frame = ttk.Frame(prediction_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ["Critical Point", "Predicted Value"]
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=200, anchor='center')
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Insert data
        for target, pred in predictions.items():
            tree.insert("", "end", values=[target, f"{pred:.2f}"])
        
        # Save button
        def save_predictions():
            file_path = filedialog.asksaveasfilename(
                title="Save Predictions",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if file_path:
                try:
                    prediction_df = pd.DataFrame([
                        {"Critical Point": target, "Predicted Value": pred}
                        for target, pred in predictions.items()
                    ])
                    self.file_handler.save_csv(prediction_df, file_path)
                    messagebox.showinfo("Success", f"Predictions saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save predictions: {str(e)}")
        
        ttk.Button(
            prediction_window, 
            text="Save Predictions to CSV", 
            command=save_predictions
        ).pack(pady=10)
    
    def optimize_with_pso(self):
        """Optimize PRV settings using PSO and show accuracy"""
        if not self.water_network_data:
            messagebox.showerror("Error", "No data loaded")
            return
        
        try:
            self._start_progress("Optimizing with PSO...")
            
            # Prepare data
            prv_data = self.water_network_data.prv_data.copy()
            point_after_valve_data = self.water_network_data.point_after_valve_data.copy()
            critical_point_data = self.water_network_data.critical_point_data.copy()
            
            prv_data = prv_data.apply(pd.to_numeric, errors='coerce').fillna(prv_data.mean())
            point_after_valve_data = point_after_valve_data.apply(pd.to_numeric, errors='coerce').fillna(point_after_valve_data.mean())
            critical_point_data = critical_point_data.apply(pd.to_numeric, errors='coerce').fillna(critical_point_data.mean())
            
            if len(point_after_valve_data) < self.constants.NUM_HOURS:
                raise ValueError(
                    f"Data must contain at least {self.constants.NUM_HOURS} hours"
                )
            
            # Initialize results storage
            optimal_prv_settings = []
            optimal_pressures = []
            demands_list = []
            penalties = []
            
            # Run PSO for each hour
            for hour in range(self.constants.NUM_HOURS):
                demands = point_after_valve_data.iloc[hour].values
                prev_settings = optimal_prv_settings[-1] if hour > 0 else None
                
                results = self.optimizer.optimize(demands, prev_settings)
                optimal_prv_settings.append(results.optimal_prv_settings[0])
                optimal_pressures.append(results.optimal_pressures[0])
                demands_list.append(demands)
                penalties.append(results.score)
            
            # Predict critical point pressures using XGBoost
            critical_point_predictions = self._predict_critical_points(
                optimal_pressures, 
                point_after_valve_data, 
                critical_point_data
            )
            
            # Store results
            self.pso_results = PSOResults(
                optimal_prv_settings=optimal_prv_settings,
                optimal_pressures=optimal_pressures,
                demands=demands_list,
                critical_point_predictions=critical_point_predictions,
                score=np.mean(penalties),
                convergence_history=results.convergence_history,
                optimization_time=results.optimization_time,
                final_parameters=results.final_parameters
            )
            
            # Calculate optimization metrics
            avg_pressure = np.mean([np.mean(p) for p in optimal_pressures])
            pressure_std = np.mean([np.std(p) for p in optimal_pressures])
            avg_penalty = np.mean(penalties)
            
            # Display results
            self._display_pso_results()
            
            # Show accuracy metrics
            metrics = {
                'penalty': avg_penalty,
                'avg_pressure': avg_pressure,
                'pressure_std': pressure_std,
                'optimization_time': self.pso_results.optimization_time,
                'convergence_rate': self.pso_results.final_parameters['convergence_rate']
            }
            
            additional_info = (
                f"Optimization Hours: {self.constants.NUM_HOURS}\n"
                f"Number of PRVs: {len(optimal_prv_settings[0])}\n"
                f"Pressure Range: {self.constants.MIN_PRESSURE} - {self.constants.MAX_PRESSURE} meters\n"
                f"PRV Settings Range: {np.min(optimal_prv_settings):.2f} - {np.max(optimal_prv_settings):.2f}\n"
                f"PSO Parameters: {self.pso_results.final_parameters}"
            )
            
            self._show_accuracy_message("PSO Optimization Performance", metrics, additional_info)
            
            # Show convergence plot
            convergence_plot = self.plot_generator.create_convergence_plot(
                self.pso_results.convergence_history,
                "PSO Convergence History"
            )
            self.plot_generator.show_plot(convergence_plot, "PSO Convergence")
            
            self._stop_progress()
            messagebox.showinfo("Success", "PSO optimization completed")
            self._update_status("PSO optimization completed")
            
        except Exception as e:
            self._stop_progress()
            messagebox.showerror("Error", f"Failed to optimize with PSO: {str(e)}")
            logger.error(f"PSO optimization error: {str(e)}")
    
    def _predict_critical_points(self, optimal_pressures: List[np.ndarray], 
                               point_after_valve_data: pd.DataFrame, 
                               critical_point_data: pd.DataFrame) -> np.ndarray:
        """Predict critical point pressures using XGBoost"""
        deby_data = self.water_network_data.deby_data.copy()
        deby_data = deby_data.apply(pd.to_numeric, errors='coerce').fillna(deby_data.mean())
        deby_values = deby_data['P-676'].values[:self.constants.NUM_HOURS]
        
        features = list(point_after_valve_data.columns) + ['P-676']
        targets = list(critical_point_data.columns)
        
        data = pd.concat([
            point_after_valve_data, 
            deby_data, 
            critical_point_data
        ], axis=1, join='inner')
        
        if data.empty:
            raise ValueError("No overlapping data found")
        
        X = data[features].to_numpy(dtype=float)
        y = data[targets].to_numpy(dtype=float)
        
        if X.shape[0] < 10 or y.shape[0] < 10:
            raise ValueError("Insufficient data for training")
        
        # Train model
        model_results = self.model_trainer.train(X, y)
        
        # Prepare input data for prediction
        input_data = []
        for hour in range(self.constants.NUM_HOURS):
            pressures = optimal_pressures[hour]
            deby = deby_values[hour]
            input_data.append(list(pressures) + [deby])
        
        input_data = np.array(input_data)
        return model_results.model.predict(input_data)
    
    def _display_pso_results(self):
        """Display PSO optimization results"""
        result_window = tk.Toplevel(self.root)
        result_window.title("PSO Optimization Results")
        result_window.geometry("1000x700")
        result_window.configure(bg="#f0f0f0")
        
        ttk.Label(
            result_window, 
            text=f"PSO Optimization Results ({self.constants.NUM_HOURS} Hours)", 
            font=self.title_font
        ).pack(pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(result_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Results table
        table_tab = ttk.Frame(notebook)
        notebook.add(table_tab, text="Results Table")
        
        # Create treeview for results
        tree_frame = ttk.Frame(table_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ["Hour"] + \
                  [f"PRV_{i+1}" for i in range(len(self.pso_results.optimal_prv_settings[0]))] + \
                  [f"Pressure_{i+1}" for i in range(len(self.pso_results.optimal_pressures[0]))] + \
                  [f"Demand_{i+1}" for i in range(len(self.pso_results.demands[0]))] + \
                  [f"Critical_{col}" for col in self.water_network_data.critical_point_data.columns]
        
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor='center')
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Insert data
        for hour in range(self.constants.NUM_HOURS):
            values = [hour + 1] + \
                     [f"{x:.2f}" for x in self.pso_results.optimal_prv_settings[hour]] + \
                     [f"{x:.2f}" for x in self.pso_results.optimal_pressures[hour]] + \
                     [f"{x:.2f}" for x in self.pso_results.demands[hour]] + \
                     [f"{self.pso_results.critical_point_predictions[hour][i]:.2f}" 
                      for i in range(len(self.water_network_data.critical_point_data.columns))]
            
            tree.insert("", "end", values=values)
        
        # Tab 2: Plots
        plots_tab = ttk.Frame(notebook)
        notebook.add(plots_tab, text="Plots")
        
        # Create plots
        self._create_pso_plots(plots_tab)
        
        # Save results button
        def save_results():
            file_path = filedialog.asksaveasfilename(
                title="Save PSO Results",
                defaultextension=".inp",
                filetypes=[("INP Files", "*.inp"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
            
            try:
                self.inp_generator.generate_inp_file(
                    self.pso_results,
                    file_path,
                    list(self.water_network_data.critical_point_data.columns),
                    [col.split('-B')[0] for col in self.water_network_data.point_after_valve_data.columns if '-B' in col]
                )
                messagebox.showinfo("Success", f"PSO results saved to {file_path}")
                
                # Prompt user to run simulation
                if messagebox.askyesno("Run Simulation", "Do you want to run simulation?"):
                    self.run_simulation(file_path)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save PSO results: {str(e)}")
        
        ttk.Button(
            result_window, 
            text="Save Results to INP", 
            command=save_results
        ).pack(pady=10)
    
    def _create_pso_plots(self, parent: tk.Widget):
        """Create and save PSO optimization plots"""
        # Pressure plot
        pressure_data = {
            f"Point_{i+1}": [self.pso_results.optimal_pressures[hour][i] 
                             for hour in range(self.constants.NUM_HOURS)]
            for i in range(len(self.pso_results.optimal_pressures[0]))
        }
        
        pressure_plot = self.plot_generator.create_line_plot(
            pressure_data,
            "Optimized Pressures at Point After Valve",
            "Hour",
            "Pressure (meters)"
        )
        self.plot_generator.show_plot(pressure_plot, "Optimized Pressures")
        
        # Demand plot
        demand_data = {
            f"Demand_{i+1}": [self.pso_results.demands[hour][i] 
                             for hour in range(self.constants.NUM_HOURS)]
            for i in range(len(self.pso_results.demands[0]))
        }
        
        demand_plot = self.plot_generator.create_line_plot(
            demand_data,
            "Demands at Point After Valve",
            "Hour",
            "Demand"
        )
        self.plot_generator.show_plot(demand_plot, "Demands")
        
        # Critical point pressures plot
        critical_data = {
            f"Critical_{col}": [self.pso_results.critical_point_predictions[hour][i] 
                               for hour in range(self.constants.NUM_HOURS)]
            for i, col in enumerate(self.water_network_data.critical_point_data.columns)
        }
        
        critical_plot = self.plot_generator.create_line_plot(
            critical_data,
            "Predicted Critical Point Pressures",
            "Hour",
            "Pressure (meters)"
        )
        self.plot_generator.show_plot(critical_plot, "Critical Point Pressures")
        
        # Create heatmap
        heatmap_fig = self.plot_generator.create_pressure_heatmap(
            np.array(self.pso_results.critical_point_predictions).T,
            list(self.water_network_data.critical_point_data.columns),
            "Critical Point Pressures Heatmap"
        )
        self.plot_generator.show_plot(heatmap_fig, "Pressure Heatmap")
        
        # Display plot buttons in GUI
        ttk.Label(
            parent, 
            text="Generated Plots:", 
            font=self.custom_font
        ).pack(pady=10)
        
        plot_buttons = [
            ("Pressure Plot", lambda: self.plot_generator.show_plot(pressure_plot, "Optimized Pressures")),
            ("Demand Plot", lambda: self.plot_generator.show_plot(demand_plot, "Demands")),
            ("Critical Point Plot", lambda: self.plot_generator.show_plot(critical_plot, "Critical Point Pressures")),
            ("Heatmap", lambda: self.plot_generator.show_plot(heatmap_fig, "Pressure Heatmap"))
        ]
        
        for text, command in plot_buttons:
            ttk.Button(
                parent, 
                text=text, 
                command=command
            ).pack(side=tk.LEFT, padx=5, pady=5)
    
    def _parse_hours(self, hours_str: str) -> List[int]:
        """Parse hours string like '1,3,5-7' into list of integers"""
        hours = []
        parts = hours_str.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                if start > end or start < 1 or end > self.constants.NUM_HOURS:
                    raise ValueError(f"Invalid hour range: {part}")
                hours.extend(range(start, end + 1))
            else:
                h = int(part)
                if h < 1 or h > self.constants.NUM_HOURS:
                    raise ValueError(f"Invalid hour: {h}")
                hours.append(h)
        return sorted(set(hours))
    
    def run_simulation(self, inp_file: Optional[str] = None):
        """Run hydraulic simulation"""
        if not self.pso_results and not inp_file:
            messagebox.showerror("Error", "No PSO results or INP file provided")
            return
        
        if not inp_file:
            file_path = filedialog.askopenfilename(
                title="Select INP File",
                filetypes=[("INP Files", "*.inp"), ("All Files", "*.*")]
            )
            
            if not file_path:
                return
        else:
            file_path = inp_file
        
        # Prompt user for hours
        hours_str = simpledialog.askstring(
            "Select Hours", 
            "Enter hours (comma separated, e.g., 1,3,5-7) or leave blank for all:"
        )
        
        hours = None
        if hours_str and hours_str.strip():
            try:
                hours = self._parse_hours(hours_str)
            except ValueError as e:
                messagebox.showerror("Error", str(e))
                return
        
        try:
            self._start_progress("Running simulation...")
            
            # Run simulation
            simulated_pressures = self.simulator.run_simulation(
                file_path,
                list(self.water_network_data.critical_point_data.columns),
                hours=hours
            )
            
            # Compare results if PSO results available
            self._compare_simulation_results(simulated_pressures, hours)
            
            self._stop_progress()
            self._update_status("Simulation completed")
            
        except Exception as e:
            self._stop_progress()
            error_msg = f"Failed to run simulation: {str(e)}"
            logger.error(f"Simulation error: {str(e)}")
            
            error_detail = f"""
Simulation Failed
Error: {str(e)}
Possible solutions:
1. Verify the INP file format
2. Ensure all node IDs exist in the network
3. Check hydraulic parameters (pressures, demands)
Technical Details:
- File: {file_path}
- Critical Points: {list(self.water_network_data.critical_point_data.columns)}
- Error Type: {type(e).__name__}
"""
            
            messagebox.showerror("Simulation Error", error_detail)
    
    def _compare_simulation_results(self, simulated_pressures: np.ndarray, hours: Optional[List[int]] = None):
        """Compare simulation results with predictions and show accuracy"""
        if hours is None:
            hours = list(range(1, self.constants.NUM_HOURS + 1))
        
        if self.pso_results:
            predicted_pressures = self.pso_results.critical_point_predictions[[h - 1 for h in hours]]
        else:
            predicted_pressures = None  # No comparison if no PSO results
        
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Simulation vs Predictions")
        comparison_window.geometry("1000x700")
        comparison_window.configure(bg="#f0f0f0")
        
        ttk.Label(
            comparison_window, 
            text="Simulation vs Predicted Pressures", 
            font=self.title_font
        ).pack(pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(comparison_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Comparison table
        table_tab = ttk.Frame(notebook)
        notebook.add(table_tab, text="Comparison Table")
        
        # Create treeview for comparison
        tree_frame = ttk.Frame(table_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ["Hour"] + \
                  [f"{node}_Sim" for node in self.water_network_data.critical_point_data.columns]
        
        if predicted_pressures is not None:
            columns += [f"{node}_Pred" for node in self.water_network_data.critical_point_data.columns] + \
                       [f"{node}_Diff" for node in self.water_network_data.critical_point_data.columns]
        
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor='center')
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Insert comparison data and calculate accuracy
        total_mae = 0
        total_rmse = 0
        total_mape = 0
        count = 0
        
        for i, hour in enumerate(hours):
            values = [hour]
            
            for j, node in enumerate(self.water_network_data.critical_point_data.columns):
                sim = simulated_pressures[i][j]
                values.append(f"{sim:.2f}")
                
                if predicted_pressures is not None:
                    pred = predicted_pressures[i][j]
                    diff = sim - pred
                    values.extend([f"{pred:.2f}", f"{diff:.2f}"])
                    
                    # Calculate accuracy metrics
                    if sim != 0:  # Avoid division by zero
                        total_mae += abs(diff)
                        total_rmse += diff ** 2
                        total_mape += abs(diff / sim) * 100
                        count += 1
            
            tree.insert("", "end", values=values)
        
        # Calculate average accuracy metrics if comparison available
        if predicted_pressures is not None:
            avg_mae = total_mae / count if count > 0 else 0
            avg_rmse = np.sqrt(total_rmse / count) if count > 0 else 0
            avg_mape = total_mape / count if count > 0 else 0
        
        # Tab 2: Comparison plot (if comparison available)
        if predicted_pressures is not None:
            plot_tab = ttk.Frame(notebook)
            notebook.add(plot_tab, text="Comparison Plot")
            
            # Create comparison plot
            comparison_fig = self.plot_generator.create_comparison_plot(
                simulated_pressures,
                predicted_pressures,
                list(self.water_network_data.critical_point_data.columns),
                "Simulation vs Predicted Pressures"
            )
            
            # Add plot to window
            canvas = FigureCanvasTkAgg(comparison_fig, master=plot_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Show accuracy metrics
            metrics = {
                'mae': avg_mae,
                'rmse': avg_rmse,
                'mape': avg_mape
            }
            
            additional_info = (
                f"Comparison Hours: {len(hours)}\n"
                f"Critical Points: {len(self.water_network_data.critical_point_data.columns)}\n"
                f"Simulated Pressure Range: {simulated_pressures.min():.2f} - {simulated_pressures.max():.2f}\n"
                f"Predicted Pressure Range: {predicted_pressures.min():.2f} - {predicted_pressures.max():.2f}"
            )
            
            self._show_accuracy_message("Simulation vs Prediction Accuracy", metrics, additional_info)
        
        # Add save button
        def save_results():
            file_path = filedialog.asksaveasfilename(
                title="Save Comparison Results",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if file_path:
                try:
                    # Create DataFrame for saving
                    data = []
                    for i, h in enumerate(hours):
                        row = {'Hour': h}
                        for j, node in enumerate(self.water_network_data.critical_point_data.columns):
                            row[f"{node}_Sim"] = simulated_pressures[i][j]
                            if predicted_pressures is not None:
                                row[f"{node}_Pred"] = predicted_pressures[i][j]
                                row[f"{node}_Diff"] = simulated_pressures[i][j] - predicted_pressures[i][j]
                        data.append(row)
                    
                    df = pd.DataFrame(data)
                    self.file_handler.save_csv(df, file_path)
                    messagebox.showinfo("Success", f"Comparison results saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save results: {str(e)}")
        
        ttk.Button(
            comparison_window, 
            text="Save Results", 
            command=save_results
        ).pack(pady=10)
    
    def predict_critical_points(self):
        """Predict Critical Point pressures using PSO outputs and Deby"""
        if not self.pso_results:
            messagebox.showerror("Error", "Run PSO optimization first")
            return
        
        if not self.model_results or not self.model_results.model:
            messagebox.showerror("Error", "Train XGBoost model first")
            return
        
        try:
            # Create input window for Deby values
            deby_window = tk.Toplevel(self.root)
            deby_window.title("Input Deby Values")
            deby_window.geometry("400x600")
            deby_window.configure(bg="#f0f0f0")
            
            ttk.Label(
                deby_window, 
                text="Enter Deby values for each hour:", 
                font=self.custom_font
            ).pack(pady=10)
            
            deby_inputs = []
            for hour in range(self.constants.NUM_HOURS):
                frame = ttk.Frame(deby_window)
                frame.pack(fill=tk.X, padx=10, pady=2)
                
                ttk.Label(
                    frame, 
                    text=f"Hour {hour+1}:", 
                    font=self.custom_font
                ).pack(side=tk.LEFT)
                
                default_value = str(self.water_network_data.deby_data['P-676'].iloc[hour % len(self.water_network_data.deby_data)])
                
                entry_var = tk.StringVar(value=default_value)
                ttk.Entry(
                    frame, 
                    textvariable=entry_var, 
                    font=self.custom_font
                ).pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                deby_inputs.append(entry_var)
            
            def predict():
                try:
                    deby_values = [float(var.get()) for var in deby_inputs]
                    
                    # Get the features used during training
                    point_after_valve_cols = list(self.water_network_data.point_after_valve_data.columns)
                    
                    # Prepare input data for prediction
                    input_data = []
                    for hour in range(self.constants.NUM_HOURS):
                        # Use optimal pressures from PSO as Point After Valve values
                        pressures = self.pso_results.optimal_pressures[hour]
                        deby = deby_values[hour]
                        
                        # Ensure the number of pressures matches the number of Point After Valve columns
                        if len(pressures) != len(point_after_valve_cols):
                            raise ValueError(
                                f"Number of pressures ({len(pressures)}) does not match "
                                f"number of Point After Valve columns ({len(point_after_valve_cols)})"
                            )
                        
                        # Combine pressures and Deby value
                        input_data.append(list(pressures) + [deby])
                    
                    input_data = np.array(input_data)
                    
                    # Log input shape for debugging
                    logger.info(f"Prediction input shape: {input_data.shape}")
                    
                    # Make predictions using the trained model
                    predictions = self.model_trainer.predict(input_data)
                    
                    # Create a dictionary to store predictions for each critical point
                    prediction_dict = {}
                    for i, col in enumerate(self.water_network_data.critical_point_data.columns):
                        prediction_dict[col] = [predictions[hour][i] for hour in range(self.constants.NUM_HOURS)]
                    
                    deby_window.destroy()
                    
                    # Display predictions in a table and plot
                    self._display_critical_point_predictions(prediction_dict, "Critical Point Predictions")
                    
                except ValueError as e:
                    messagebox.showerror("Error", f"Invalid input: {str(e)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to predict critical points: {str(e)}")
                    logger.error(f"Critical point prediction error: {str(e)}")
            
            ttk.Button(
                deby_window, 
                text="Predict", 
                command=predict
            ).pack(pady=15)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create prediction window: {str(e)}")
            logger.error(f"Critical point prediction window error: {str(e)}")
    
    def _display_critical_point_predictions(self, predictions: Dict[str, List[float]], title: str):
        """Display critical point predictions in a new window with table and plot"""
        prediction_window = tk.Toplevel(self.root)
        prediction_window.title(title)
        prediction_window.geometry("1000x700")
        prediction_window.configure(bg="#f0f0f0")
        
        ttk.Label(
            prediction_window, 
            text=title, 
            font=self.title_font
        ).pack(pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(prediction_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Tab 1: Table view
        table_tab = ttk.Frame(notebook)
        notebook.add(table_tab, text="Table View")
        
        # Create treeview for predictions
        tree_frame = ttk.Frame(table_tab)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create columns: Hour + each critical point
        columns = ["Hour"] + list(predictions.keys())
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings')
        
        # Configure columns
        tree.column("Hour", width=50, anchor='center')
        for col in predictions.keys():
            tree.column(col, width=100, anchor='center')
            tree.heading(col, text=col)
        
        y_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=y_scroll.set)
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Insert data
        for hour in range(self.constants.NUM_HOURS):
            values = [hour + 1]
            for col in predictions.keys():
                values.append(f"{predictions[col][hour]:.2f}")
            tree.insert("", "end", values=values)
        
        # Tab 2: Plot view
        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="Plot View")
        
        # Create line plot
        line_plot = self.plot_generator.create_line_plot(
            predictions,
            "Critical Point Pressure Predictions",
            "Hour",
            "Pressure (m)"
        )
        
        # Add plot to window
        canvas = FigureCanvasTkAgg(line_plot, master=plot_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Save button
        def save_predictions():
            file_path = filedialog.asksaveasfilename(
                title="Save Predictions",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if file_path:
                try:
                    # Create DataFrame for saving
                    data = {'Hour': list(range(1, self.constants.NUM_HOURS + 1))}
                    for col, values in predictions.items():
                        data[col] = values
                    
                    df = pd.DataFrame(data)
                    self.file_handler.save_csv(df, file_path)
                    messagebox.showinfo("Success", f"Predictions saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save predictions: {str(e)}")
        
        ttk.Button(
            prediction_window, 
            text="Save Predictions to CSV", 
            command=save_predictions
        ).pack(pady=10)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Water Network Analyzer - Professional Edition v4.0
        
A comprehensive application for water network analysis using:
- Machine Learning (XGBoost)
- Particle Swarm Optimization (PSO)
- Hydraulic Simulation
Features:
- Advanced data preprocessing and validation
- Hyperparameter optimization
- Hydraulic simulation with INP file generation
- Comprehensive visualization
Version: 4.0
Author: Water Network Team
License: MIT"""
        
        messagebox.showinfo("About", about_text)

def main():
    """Main entry point for the application"""
    root = tk.Tk()
    app = WaterNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()