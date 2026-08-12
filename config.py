"""
Central configuration for Water Network AI Analyzer.

This module contains project-wide settings for:
- Machine learning
- Cross-validation
- XGBoost hyperparameter search
- Pressure constraints
- PRV operating limits
- Particle Swarm Optimization
- Reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass(frozen=True, slots=True)
class AppConfig:
    """
    Immutable application configuration.

    Using a frozen dataclass prevents accidental modification of important
    experiment settings while the application is running.
    """

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    app_name: str = "Water Network AI Analyzer"
    version: str = "5.0.0"
    log_file: str = "water_network_analyzer.log"

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    random_state: int = 42

    # ------------------------------------------------------------------
    # Dataset / training
    # ------------------------------------------------------------------

    test_size: float = 0.20

    # Minimum number of valid observations required for model training.
    min_training_rows: int = 12

    # Prevent extremely large accidental GUI training workloads.
    max_training_rows: int = 50_000

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    knn_neighbors: int = 5

    # IQR clipping:
    # lower = Q1 - factor * IQR
    # upper = Q3 + factor * IQR
    iqr_factor: float = 1.5

    # ------------------------------------------------------------------
    # Cross-validation / hyperparameter tuning
    # ------------------------------------------------------------------

    cv_folds: int = 5

    # Number of parameter combinations sampled by RandomizedSearchCV.
    search_iterations: int = 14

    # ------------------------------------------------------------------
    # Hydraulic / engineering pressure constraints
    # ------------------------------------------------------------------

    min_pressure: float = 10.0
    max_pressure: float = 60.0

    # Preferred operating pressure used by the PSO objective.
    target_pressure: float = 30.0

    # ------------------------------------------------------------------
    # PRV safety bounds
    # ------------------------------------------------------------------

    min_prv: float = 10.0
    max_prv: float = 60.0

    # Default number of sequential operating periods.
    optimization_hours: int = 24

    # ------------------------------------------------------------------
    # Particle Swarm Optimization
    # ------------------------------------------------------------------

    pso_particles: int = 36
    pso_iterations: int = 70

    # Inertia decreases from w_max to w_min during optimization.
    pso_w_max: float = 0.90
    pso_w_min: float = 0.40

    # Cognitive coefficient.
    pso_c1: float = 1.70

    # Social coefficient.
    pso_c2: float = 1.90

    # Maximum velocity relative to each PRV operating range.
    pso_velocity_fraction: float = 0.20

    # ------------------------------------------------------------------
    # PSO objective weights
    # ------------------------------------------------------------------

    # Large penalty for pressure outside engineering limits.
    pressure_violation_weight: float = 50.0

    # Encourages pressure toward target_pressure.
    pressure_target_weight: float = 0.15

    # Penalizes large setting changes between sequential periods.
    stability_weight: float = 0.10

    # Penalizes unnecessary deviation from historical PRV settings.
    reference_weight: float = 0.03

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate configuration immediately after creation."""

        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")

        if self.min_training_rows < 4:
            raise ValueError("min_training_rows must be at least 4.")

        if self.max_training_rows < self.min_training_rows:
            raise ValueError(
                "max_training_rows must be greater than or equal to "
                "min_training_rows."
            )

        if self.knn_neighbors < 1:
            raise ValueError("knn_neighbors must be at least 1.")

        if self.iqr_factor <= 0:
            raise ValueError("iqr_factor must be greater than 0.")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")

        if self.search_iterations < 1:
            raise ValueError("search_iterations must be at least 1.")

        if self.min_pressure >= self.max_pressure:
            raise ValueError(
                "min_pressure must be smaller than max_pressure."
            )

        if not self.min_pressure <= self.target_pressure <= self.max_pressure:
            raise ValueError(
                "target_pressure must be inside the allowed pressure range."
            )

        if self.min_prv >= self.max_prv:
            raise ValueError("min_prv must be smaller than max_prv.")

        if self.optimization_hours < 1:
            raise ValueError("optimization_hours must be at least 1.")

        if self.pso_particles < 2:
            raise ValueError("pso_particles must be at least 2.")

        if self.pso_iterations < 1:
            raise ValueError("pso_iterations must be at least 1.")

        if self.pso_w_min <= 0 or self.pso_w_max <= 0:
            raise ValueError("PSO inertia weights must be positive.")

        if self.pso_w_min > self.pso_w_max:
            raise ValueError(
                "pso_w_min cannot be greater than pso_w_max."
            )

        if self.pso_c1 < 0 or self.pso_c2 < 0:
            raise ValueError(
                "PSO acceleration coefficients cannot be negative."
            )

        if not 0.0 < self.pso_velocity_fraction <= 1.0:
            raise ValueError(
                "pso_velocity_fraction must be between 0 and 1."
            )

        weights = (
            self.pressure_violation_weight,
            self.pressure_target_weight,
            self.stability_weight,
            self.reference_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Optimization objective weights cannot be negative."
            )

    # ------------------------------------------------------------------
    # XGBoost search space
    # ------------------------------------------------------------------

    def xgb_param_distributions(
        self,
        multi_output: bool = False,
    ) -> Dict[str, Sequence[Any]]:
        """
        Return the RandomizedSearchCV parameter search space.

        MultiOutputRegressor introduces an additional ``estimator`` level,
        so parameter names must be adjusted accordingly.
        """

        prefix = (
            "model__estimator__"
            if multi_output
            else "model__"
        )

        return {
            f"{prefix}n_estimators": [
                100,
                160,
                240,
                320,
                450,
            ],
            f"{prefix}max_depth": [
                2,
                3,
                4,
                5,
                6,
                8,
            ],
            f"{prefix}learning_rate": [
                0.015,
                0.03,
                0.05,
                0.08,
                0.12,
            ],
            f"{prefix}subsample": [
                0.70,
                0.80,
                0.90,
                1.00,
            ],
            f"{prefix}colsample_bytree": [
                0.70,
                0.80,
                0.90,
                1.00,
            ],
            f"{prefix}min_child_weight": [
                1,
                2,
                4,
                6,
            ],
            f"{prefix}reg_alpha": [
                0.0,
                0.001,
                0.01,
                0.10,
            ],
            f"{prefix}reg_lambda": [
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
            ],
        }


# ----------------------------------------------------------------------
# Default project configuration
# ----------------------------------------------------------------------

DEFAULT_CONFIG = AppConfig()
