from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
import xgboost as xgb
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    import wntr  # Optional dependency for real EPANET/WNTR simulation.
except Exception:  # pragma: no cover - optional dependency
    wntr = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG_FILE = "water_network_analyzer.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("water-network-ai-analyzer")


# -----------------------------------------------------------------------------
# Configuration and data containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Central configuration for ML, PSO, and UI behavior."""

    random_state: int = 42
    test_size: float = 0.20
    min_training_rows: int = 12
    max_training_rows: int = 50_000
    knn_neighbors: int = 5
    iqr_factor: float = 1.5
    cv_folds: int = 5
    search_iterations: int = 14

    min_pressure: float = 10.0
    max_pressure: float = 60.0
    target_pressure: float = 30.0
    min_prv: float = 10.0
    max_prv: float = 60.0
    optimization_hours: int = 24

    pso_particles: int = 36
    pso_iterations: int = 70
    pso_w_max: float = 0.90
    pso_w_min: float = 0.40
    pso_c1: float = 1.7
    pso_c2: float = 1.9
    pso_velocity_fraction: float = 0.20

    pressure_violation_weight: float = 50.0
    pressure_target_weight: float = 0.15
    stability_weight: float = 0.10
    reference_weight: float = 0.03


@dataclass
class Schema:
    """Detected semantic groups in a water-network CSV."""

    prv_columns: List[str] = field(default_factory=list)
    point_after_valve_columns: List[str] = field(default_factory=list)
    critical_point_columns: List[str] = field(default_factory=list)
    demand_column: Optional[str] = None
    ignored_columns: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prv_columns": self.prv_columns,
            "point_after_valve_columns": self.point_after_valve_columns,
            "critical_point_columns": self.critical_point_columns,
            "demand_column": self.demand_column,
            "ignored_columns": self.ignored_columns,
        }


@dataclass
class WaterNetworkData:
    """Raw data plus detected schema."""

    raw: pd.DataFrame
    schema: Schema
    source_path: Optional[Path] = None

    @property
    def prv_data(self) -> pd.DataFrame:
        return self.raw.loc[:, self.schema.prv_columns].copy()

    @property
    def point_after_valve_data(self) -> pd.DataFrame:
        return self.raw.loc[:, self.schema.point_after_valve_columns].copy()

    @property
    def critical_point_data(self) -> pd.DataFrame:
        return self.raw.loc[:, self.schema.critical_point_columns].copy()

    @property
    def demand_data(self) -> pd.DataFrame:
        if not self.schema.demand_column:
            return pd.DataFrame(index=self.raw.index)
        return self.raw.loc[:, [self.schema.demand_column]].copy()


@dataclass
class RegressionResult:
    pipeline: Pipeline
    feature_names: List[str]
    target_names: List[str]
    metrics: Dict[str, float]
    per_target_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    best_params: Dict[str, Any]
    best_cv_score: Optional[float]
    training_seconds: float
    y_test: np.ndarray
    y_pred: np.ndarray
    test_indices: np.ndarray


@dataclass
class PSOHourResult:
    hour: int
    demand: float
    prv_settings: np.ndarray
    downstream_pressures: np.ndarray
    critical_pressures: Optional[np.ndarray]
    objective: float
    convergence: List[float]


@dataclass
class OptimizationResult:
    hours: List[PSOHourResult]
    prv_names: List[str]
    downstream_names: List[str]
    critical_names: List[str]
    total_seconds: float

    def to_dataframe(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for item in self.hours:
            row: Dict[str, Any] = {
                "Hour": item.hour,
                "Demand": item.demand,
                "Objective": item.objective,
                "Downstream_Min": float(np.min(item.downstream_pressures)),
                "Downstream_Mean": float(np.mean(item.downstream_pressures)),
                "Downstream_Max": float(np.max(item.downstream_pressures)),
            }
            for name, value in zip(self.prv_names, item.prv_settings):
                row[f"PRV::{name}"] = float(value)
            for name, value in zip(self.downstream_names, item.downstream_pressures):
                row[f"Downstream::{name}"] = float(value)
            if item.critical_pressures is not None:
                for name, value in zip(self.critical_names, item.critical_pressures):
                    row[f"Critical::{name}"] = float(value)
            rows.append(row)
        return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# File I/O and schema detection
# -----------------------------------------------------------------------------

class CSVService:
    """CSV reader/writer with encoding fallbacks and clean validation."""

    ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1252", "latin1")

    @classmethod
    def load(cls, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")

        errors: List[str] = []
        for encoding in cls.ENCODINGS:
            try:
                df = pd.read_csv(path, encoding=encoding)
                if df.empty:
                    raise ValueError("CSV contains no rows.")
                if len(df.columns) < 2:
                    raise ValueError("CSV must contain at least two columns.")
                df.columns = [str(c).strip() for c in df.columns]
                logger.info("Loaded %s: %d rows x %d cols (%s)", path, len(df), len(df.columns), encoding)
                return df
            except UnicodeDecodeError as exc:
                errors.append(f"{encoding}: {exc}")
            except pd.errors.ParserError as exc:
                errors.append(f"{encoding}: {exc}")

        raise ValueError("Could not decode/parse CSV. " + " | ".join(errors[-3:]))

    @staticmethod
    def save(df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("Saved CSV: %s", path)


class SchemaDetector:
    """Detect repository-compatible water-network column naming patterns."""

    DEMAND_EXACT_NAMES = ("p-676", "deby", "demand", "flow", "total_demand")

    @classmethod
    def detect(cls, df: pd.DataFrame) -> Schema:
        columns = [str(c).strip() for c in df.columns]
        lower = {c: c.lower().strip() for c in columns}

        prv = [
            c for c in columns
            if "prv" in lower[c]
            and not any(token in lower[c] for token in ("status", "id", "name"))
        ]

        point_after = [
            c for c in columns
            if (
                lower[c].endswith("-b")
                or lower[c].endswith("_b")
                or "after_valve" in lower[c]
                or "after valve" in lower[c]
                or "downstream" in lower[c]
            )
            and c not in prv
        ]

        demand: Optional[str] = None
        # Repository legacy convention has first priority.
        for candidate in ("P-676", "p-676"):
            match = next((c for c in columns if c.lower() == candidate.lower()), None)
            if match:
                demand = match
                break
        if demand is None:
            for name in cls.DEMAND_EXACT_NAMES:
                match = next((c for c in columns if lower[c] == name), None)
                if match:
                    demand = match
                    break
        if demand is None:
            match = next(
                (
                    c for c in columns
                    if any(token in lower[c] for token in ("demand", "deby", "flow"))
                ),
                None,
            )
            demand = match

        critical = [
            c for c in columns
            if (
                lower[c].startswith("j-")
                or lower[c].startswith("critical")
                or "critical_point" in lower[c]
                or "critical point" in lower[c]
            )
            and c not in point_after
            and c not in prv
            and c != demand
        ]

        used = set(prv + point_after + critical + ([demand] if demand else []))
        ignored = [c for c in columns if c not in used]

        return Schema(
            prv_columns=prv,
            point_after_valve_columns=point_after,
            critical_point_columns=critical,
            demand_column=demand,
            ignored_columns=ignored,
        )

    @staticmethod
    def validate_for_critical_model(schema: Schema) -> None:
        problems: List[str] = []
        if not schema.point_after_valve_columns:
            problems.append("Point After Valve columns (e.g. *-B or downstream*)")
        if not schema.critical_point_columns:
            problems.append("Critical Point columns (e.g. J-*)")
        if not schema.demand_column:
            problems.append("Demand/Deby column (legacy: P-676)")
        if problems:
            raise ValueError("Missing required columns: " + ", ".join(problems))

    @staticmethod
    def validate_for_pso(schema: Schema) -> None:
        SchemaDetector.validate_for_critical_model(schema)
        if not schema.prv_columns:
            raise ValueError("Missing PRV columns. PSO needs historical PRV settings to train its surrogate model.")


# -----------------------------------------------------------------------------
# Leakage-safe ML transformers and regression service
# -----------------------------------------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Fold-local IQR winsorizer.

    Unlike row deletion, clipping is compatible with sklearn Pipeline and
    therefore is refit independently inside each cross-validation fold. This
    avoids the leakage caused by calculating outlier thresholds on the full
    dataset before splitting.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IQRClipper":
        arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(arr, 25, axis=0)
        q3 = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        # Constant columns should not be widened artificially.
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "lower_"):
            raise RuntimeError("IQRClipper must be fitted before transform().")
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lower_, self.upper_)


class XGBoostRegressionService:
    """Leakage-safe XGBoost training for single or multi-output regression."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.result: Optional[RegressionResult] = None

    def _base_regressor(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=self.config.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    def _build_pipeline(self, multi_output: bool) -> Pipeline:
        model: Any = self._base_regressor()
        if multi_output:
            model = MultiOutputRegressor(model, n_jobs=1)

        return Pipeline(
            steps=[
                ("imputer", KNNImputer(n_neighbors=self.config.knn_neighbors, weights="distance")),
                ("iqr", IQRClipper(self.config.iqr_factor)),
                ("model", model),
            ]
        )

    def _param_distributions(self, multi_output: bool) -> Dict[str, Sequence[Any]]:
        prefix = "model__estimator__" if multi_output else "model__"
        return {
            f"{prefix}n_estimators": [100, 160, 240, 320, 450],
            f"{prefix}max_depth": [2, 3, 4, 5, 6, 8],
            f"{prefix}learning_rate": [0.015, 0.03, 0.05, 0.08, 0.12],
            f"{prefix}subsample": [0.70, 0.80, 0.90, 1.0],
            f"{prefix}colsample_bytree": [0.70, 0.80, 0.90, 1.0],
            f"{prefix}min_child_weight": [1, 2, 4, 6],
            f"{prefix}reg_alpha": [0.0, 0.001, 0.01, 0.1],
            f"{prefix}reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        }

    @staticmethod
    def _clean_numeric_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        out = df.loc[:, list(columns)].copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def train(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        *,
        tune: bool = True,
    ) -> RegressionResult:
        feature_names = list(feature_names)
        target_names = list(target_names)
        if not feature_names:
            raise ValueError("At least one feature is required.")
        if not target_names:
            raise ValueError("At least one target is required.")

        missing = [c for c in feature_names + target_names if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in dataset: {missing}")

        X_df = self._clean_numeric_frame(df, feature_names)
        y_df = self._clean_numeric_frame(df, target_names)

        # Targets must be observed; imputing target values would fabricate labels.
        valid_target_mask = ~y_df.isna().any(axis=1)
        X_df = X_df.loc[valid_target_mask]
        y_df = y_df.loc[valid_target_mask]

        # Features with no observed value cannot be imputed meaningfully.
        all_nan_features = [c for c in feature_names if X_df[c].isna().all()]
        if all_nan_features:
            raise ValueError(f"Feature columns contain only missing values: {all_nan_features}")

        if len(X_df) < self.config.min_training_rows:
            raise ValueError(
                f"Insufficient labeled rows: {len(X_df)}. "
                f"Need at least {self.config.min_training_rows}."
            )

        if len(X_df) > self.config.max_training_rows:
            sampled = X_df.sample(
                self.config.max_training_rows,
                random_state=self.config.random_state,
            ).index
            X_df = X_df.loc[sampled]
            y_df = y_df.loc[sampled]

        X = X_df.to_numpy(dtype=float)
        y = y_df.to_numpy(dtype=float)
        if len(target_names) == 1:
            y = y.ravel()

        indices = X_df.index.to_numpy()
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            indices,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
        )

        multi_output = len(target_names) > 1
        pipeline = self._build_pipeline(multi_output)
        start = time.perf_counter()
        best_params: Dict[str, Any] = {}
        best_cv_score: Optional[float] = None

        max_folds = min(self.config.cv_folds, len(X_train))
        # R² is undefined for validation folds with fewer than 2 observations.
        cv_folds = min(max_folds, max(2, len(X_train) // 4))

        if tune and len(X_train) >= 16 and cv_folds >= 2:
            cv = KFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=self._param_distributions(multi_output),
                n_iter=self.config.search_iterations,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=self.config.random_state,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = dict(search.best_params_)
            best_cv_score = float(search.best_score_)
        else:
            fitted = pipeline
            fitted.fit(X_train, y_train)

        y_pred = fitted.predict(X_test)
        elapsed = time.perf_counter() - start

        y_test_2d = np.asarray(y_test).reshape(-1, len(target_names))
        y_pred_2d = np.asarray(y_pred).reshape(-1, len(target_names))

        overall_mae = float(mean_absolute_error(y_test_2d, y_pred_2d))
        overall_rmse = float(math.sqrt(mean_squared_error(y_test_2d, y_pred_2d)))
        try:
            overall_r2 = float(r2_score(y_test_2d, y_pred_2d, multioutput="uniform_average"))
        except ValueError:
            overall_r2 = float("nan")

        nonzero = np.abs(y_test_2d) > 1e-9
        if np.any(nonzero):
            mape = float(np.mean(np.abs((y_test_2d[nonzero] - y_pred_2d[nonzero]) / y_test_2d[nonzero])) * 100)
        else:
            mape = float("nan")

        per_target_rows: List[Dict[str, Any]] = []
        for i, target in enumerate(target_names):
            true = y_test_2d[:, i]
            pred = y_pred_2d[:, i]
            try:
                target_r2 = float(r2_score(true, pred))
            except ValueError:
                target_r2 = float("nan")
            per_target_rows.append(
                {
                    "Target": target,
                    "MAE": float(mean_absolute_error(true, pred)),
                    "RMSE": float(math.sqrt(mean_squared_error(true, pred))),
                    "R2": target_r2,
                }
            )

        feature_importance = self._feature_importance(fitted, feature_names)
        result = RegressionResult(
            pipeline=fitted,
            feature_names=feature_names,
            target_names=target_names,
            metrics={"MAE": overall_mae, "RMSE": overall_rmse, "R2": overall_r2, "MAPE": mape},
            per_target_metrics=pd.DataFrame(per_target_rows),
            feature_importance=feature_importance,
            best_params=best_params,
            best_cv_score=best_cv_score,
            training_seconds=elapsed,
            y_test=y_test_2d,
            y_pred=y_pred_2d,
            test_indices=np.asarray(idx_test),
        )
        self.result = result
        logger.info(
            "Model trained | rows=%d | features=%d | targets=%d | RMSE=%.4f | R2=%.4f",
            len(X_df),
            len(feature_names),
            len(target_names),
            overall_rmse,
            overall_r2,
        )
        return result

    @staticmethod
    def _feature_importance(pipeline: Pipeline, feature_names: Sequence[str]) -> pd.DataFrame:
        model = pipeline.named_steps["model"]
        values: Optional[np.ndarray] = None
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
        elif isinstance(model, MultiOutputRegressor) and getattr(model, "estimators_", None):
            importances = [
                np.asarray(est.feature_importances_, dtype=float)
                for est in model.estimators_
                if hasattr(est, "feature_importances_")
            ]
            if importances:
                values = np.mean(np.vstack(importances), axis=0)

        if values is None or len(values) != len(feature_names):
            return pd.DataFrame(columns=["Feature", "Importance"])

        frame = pd.DataFrame({"Feature": list(feature_names), "Importance": values})
        return frame.sort_values("Importance", ascending=False).reset_index(drop=True)

    def predict_frame(self, values: pd.DataFrame) -> pd.DataFrame:
        if self.result is None:
            raise RuntimeError("Model has not been trained.")
        missing = [c for c in self.result.feature_names if c not in values.columns]
        if missing:
            raise ValueError(f"Prediction input is missing features: {missing}")
        X = values.loc[:, self.result.feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        pred = self.result.pipeline.predict(X)
        pred = np.asarray(pred).reshape(-1, len(self.result.target_names))
        return pd.DataFrame(pred, columns=self.result.target_names, index=values.index)

    def save(self, path: str | Path) -> None:
        if self.result is None:
            raise RuntimeError("No trained model to save.")
        payload = {
            "pipeline": self.result.pipeline,
            "feature_names": self.result.feature_names,
            "target_names": self.result.target_names,
            "metrics": self.result.metrics,
        }
        joblib.dump(payload, path)

    def load(self, path: str | Path) -> None:
        payload = joblib.load(path)
        required = {"pipeline", "feature_names", "target_names"}
        if not required.issubset(payload):
            raise ValueError("Invalid model file.")
        self.result = RegressionResult(
            pipeline=payload["pipeline"],
            feature_names=list(payload["feature_names"]),
            target_names=list(payload["target_names"]),
            metrics=dict(payload.get("metrics", {})),
            per_target_metrics=pd.DataFrame(),
            feature_importance=pd.DataFrame(),
            best_params={},
            best_cv_score=None,
            training_seconds=0.0,
            y_test=np.empty((0, len(payload["target_names"]))),
            y_pred=np.empty((0, len(payload["target_names"]))),
            test_indices=np.array([], dtype=int),
        )


# -----------------------------------------------------------------------------
# Data-driven PSO optimizer
# -----------------------------------------------------------------------------

class SurrogatePSOOptimizer:
    """
    Optimize PRV settings against learned pressure surrogate models.

    The downstream predictor is vectorized: it receives an ``(n, n_prv)``
    matrix and returns an ``(n, n_downstream)`` matrix. An optional critical
    predictor can extend the objective so pressure constraints are enforced at
    critical points as well as downstream valve points.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

    def optimize_hour(
        self,
        *,
        demand: float,
        prv_bounds: Sequence[Tuple[float, float]],
        downstream_predictor: Callable[[np.ndarray, float], np.ndarray],
        critical_predictor: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        previous_settings: Optional[np.ndarray] = None,
        reference_settings: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
        n_prv = len(prv_bounds)
        if n_prv < 1:
            raise ValueError("PSO requires at least one PRV variable.")

        lower = np.array([b[0] for b in prv_bounds], dtype=float)
        upper = np.array([b[1] for b in prv_bounds], dtype=float)
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("PRV bounds must be finite.")
        if np.any(upper <= lower):
            raise ValueError("Every PRV upper bound must be greater than lower bound.")

        particles = self.rng.uniform(lower, upper, size=(self.config.pso_particles, n_prv))
        if reference_settings is not None and len(reference_settings) == n_prv:
            particles[0] = np.clip(reference_settings, lower, upper)
        if previous_settings is not None and len(previous_settings) == n_prv and self.config.pso_particles > 1:
            particles[1] = np.clip(previous_settings, lower, upper)

        span = upper - lower
        vmax = span * self.config.pso_velocity_fraction
        velocities = self.rng.uniform(-vmax, vmax, size=particles.shape)

        def evaluate(settings_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            settings_matrix = np.atleast_2d(np.asarray(settings_matrix, dtype=float))
            downstream = np.asarray(downstream_predictor(settings_matrix, demand), dtype=float)
            if downstream.ndim == 1:
                downstream = downstream.reshape(1, -1)
            if downstream.shape[0] != settings_matrix.shape[0]:
                raise ValueError("Downstream surrogate returned an unexpected number of rows.")
            if downstream.shape[1] == 0 or np.any(~np.isfinite(downstream)):
                return np.full(settings_matrix.shape[0], np.inf), downstream

            constrained = downstream
            if critical_predictor is not None:
                critical = np.asarray(critical_predictor(downstream, demand), dtype=float)
                if critical.ndim == 1:
                    critical = critical.reshape(1, -1)
                if critical.shape[0] != settings_matrix.shape[0]:
                    raise ValueError("Critical-point surrogate returned an unexpected number of rows.")
                if critical.shape[1] and np.all(np.isfinite(critical)):
                    constrained = np.concatenate([downstream, critical], axis=1)

            low_violation = np.maximum(self.config.min_pressure - constrained, 0.0)
            high_violation = np.maximum(constrained - self.config.max_pressure, 0.0)
            violation_penalty = self.config.pressure_violation_weight * np.mean(
                low_violation ** 2 + high_violation ** 2, axis=1
            )
            target_penalty = self.config.pressure_target_weight * np.mean(
                (constrained - self.config.target_pressure) ** 2, axis=1
            )

            stability_penalty = np.zeros(settings_matrix.shape[0], dtype=float)
            if previous_settings is not None:
                stability_penalty = self.config.stability_weight * np.mean(
                    (settings_matrix - previous_settings) ** 2, axis=1
                )

            reference_penalty = np.zeros(settings_matrix.shape[0], dtype=float)
            if reference_settings is not None:
                reference_penalty = self.config.reference_weight * np.mean(
                    (settings_matrix - reference_settings) ** 2, axis=1
                )

            scores = violation_penalty + target_penalty + stability_penalty + reference_penalty
            return scores.astype(float), downstream

        pbest = particles.copy()
        pbest_scores, _ = evaluate(particles)
        g_idx = int(np.argmin(pbest_scores))
        gbest = pbest[g_idx].copy()
        gbest_score = float(pbest_scores[g_idx])
        history = [gbest_score]

        for iteration in range(self.config.pso_iterations):
            progress = iteration / max(1, self.config.pso_iterations - 1)
            inertia = self.config.pso_w_max - (
                self.config.pso_w_max - self.config.pso_w_min
            ) * progress

            r1 = self.rng.random(size=particles.shape)
            r2 = self.rng.random(size=particles.shape)
            velocities = (
                inertia * velocities
                + self.config.pso_c1 * r1 * (pbest - particles)
                + self.config.pso_c2 * r2 * (gbest - particles)
            )
            velocities = np.clip(velocities, -vmax, vmax)
            particles = np.clip(particles + velocities, lower, upper)

            scores, _ = evaluate(particles)
            improved = scores < pbest_scores
            if np.any(improved):
                pbest[improved] = particles[improved]
                pbest_scores[improved] = scores[improved]
                candidate_idx = int(np.argmin(pbest_scores))
                candidate_score = float(pbest_scores[candidate_idx])
                if candidate_score < gbest_score:
                    gbest_score = candidate_score
                    gbest = pbest[candidate_idx].copy()
            history.append(gbest_score)

        _, downstream_final = evaluate(gbest.reshape(1, -1))
        return gbest, downstream_final[0], gbest_score, history


# -----------------------------------------------------------------------------
# Optional WNTR integration for real INP files
# -----------------------------------------------------------------------------

class WNTRService:
    @staticmethod
    def available() -> bool:
        return wntr is not None

    @staticmethod
    def simulate_pressures(
        inp_path: str | Path,
        node_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if wntr is None:
            raise RuntimeError("WNTR is not installed. Install it with: pip install wntr")
        inp_path = Path(inp_path)
        if not inp_path.is_file():
            raise FileNotFoundError(inp_path)

        network = wntr.network.WaterNetworkModel(str(inp_path))
        simulator = wntr.sim.EpanetSimulator(network)
        results = simulator.run_sim()
        pressure = results.node["pressure"].copy()

        if node_names:
            existing = [n for n in node_names if n in pressure.columns]
            if not existing:
                raise ValueError("None of the requested critical-point IDs exist in the INP network.")
            pressure = pressure.loc[:, existing]

        # WNTR index is simulation time in seconds; expose hours for readability.
        pressure.index = pressure.index.astype(float) / 3600.0
        pressure.index.name = "Hour"
        return pressure


# -----------------------------------------------------------------------------
# Plot utilities
# -----------------------------------------------------------------------------

class PlotFactory:
    @staticmethod
    def actual_vs_predicted(result: RegressionResult) -> plt.Figure:
        n_targets = len(result.target_names)
        cols = min(2, n_targets)
        rows = math.ceil(n_targets / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
        axes_flat = axes.ravel()

        for i, target in enumerate(result.target_names):
            ax = axes_flat[i]
            actual = result.y_test[:, i]
            predicted = result.y_pred[:, i]
            ax.scatter(actual, predicted, alpha=0.70)
            min_val = float(np.nanmin([actual.min(), predicted.min()]))
            max_val = float(np.nanmax([actual.max(), predicted.max()]))
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            metric_row = result.per_target_metrics[result.per_target_metrics["Target"] == target]
            r2 = float(metric_row["R2"].iloc[0]) if not metric_row.empty else float("nan")
            ax.set_title(f"{target} | R²={r2:.3f}")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.grid(alpha=0.25)

        for i in range(n_targets, len(axes_flat)):
            axes_flat[i].axis("off")
        fig.suptitle("Hold-out Evaluation: Actual vs Predicted", fontsize=14)
        fig.tight_layout()
        return fig

    @staticmethod
    def feature_importance(result: RegressionResult) -> plt.Figure:
        data = result.feature_importance.head(20).sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4.5, len(data) * 0.38)))
        if data.empty:
            ax.text(0.5, 0.5, "Feature importance unavailable", ha="center", va="center")
            ax.axis("off")
        else:
            ax.barh(data["Feature"], data["Importance"])
            ax.set_xlabel("Importance")
            ax.set_title("XGBoost Feature Importance")
            ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        return fig

    @staticmethod
    def pso_summary(result: OptimizationResult, config: AppConfig) -> plt.Figure:
        hours = [x.hour for x in result.hours]
        mean_down = [float(np.mean(x.downstream_pressures)) for x in result.hours]
        min_down = [float(np.min(x.downstream_pressures)) for x in result.hours]
        max_down = [float(np.max(x.downstream_pressures)) for x in result.hours]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(hours, mean_down, marker="o", label="Mean downstream pressure")
        ax.plot(hours, min_down, linestyle="--", label="Minimum downstream pressure")
        ax.plot(hours, max_down, linestyle="--", label="Maximum downstream pressure")
        ax.axhline(config.min_pressure, linestyle=":", label="Minimum allowed")
        ax.axhline(config.max_pressure, linestyle=":", label="Maximum allowed")
        ax.axhline(config.target_pressure, linestyle="-.", label="Target pressure")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Pressure")
        ax.set_title("PSO Surrogate Optimization Summary")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def pso_convergence(hour_result: PSOHourResult) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(len(hour_result.convergence)), hour_result.convergence)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective")
        ax.set_title(f"PSO Convergence - Hour {hour_result.hour}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig

    @staticmethod
    def descriptive_boxplot(df: pd.DataFrame, title: str) -> plt.Figure:
        numeric = df.apply(pd.to_numeric, errors="coerce")
        fig, ax = plt.subplots(figsize=(max(9, len(numeric.columns) * 0.8), 5.5))
        numeric.boxplot(ax=ax, rot=45)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class WaterNetworkApp:
    """Tkinter desktop application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = AppConfig()
        self.data: Optional[WaterNetworkData] = None

        # Separate model roles prevent accidental feature-space mismatch.
        self.critical_model = XGBoostRegressionService(self.config)
        self.downstream_model = XGBoostRegressionService(self.config)
        self.pso = SurrogatePSOOptimizer(self.config)
        self.optimization_result: Optional[OptimizationResult] = None

        self._current_plot: Optional[plt.Figure] = None
        self._working = False
        self._table_column_order: List[str] = []

        self._configure_root()
        self._configure_style()
        self._build_menu()
        self._build_layout()
        self._set_status("Ready. Load a CSV dataset to begin.")

    # --------------------------- UI construction ---------------------------

    def _configure_root(self) -> None:
        self.root.title("Water Network AI Analyzer - Professional Edition v5.0")
        self.root.geometry("1380x860")
        self.root.minsize(1080, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=25)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load CSV", command=self.load_csv)
        file_menu.add_command(label="Save Dataset", command=self.save_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Save Critical Model", command=self.save_critical_model)
        file_menu.add_command(label="Load Critical Model", command=self.load_critical_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        model_menu = tk.Menu(menubar, tearoff=False)
        model_menu.add_command(label="Train Critical-Point Model", command=self.train_critical_model)
        model_menu.add_command(label="Predict Critical Points", command=self.predict_critical_points_dialog)
        model_menu.add_command(label="Show Model Evaluation", command=self.show_model_evaluation)
        menubar.add_cascade(label="Machine Learning", menu=model_menu)

        optimization_menu = tk.Menu(menubar, tearoff=False)
        optimization_menu.add_command(label="Run PRV Optimization", command=self.run_pso)
        optimization_menu.add_command(label="Export Optimization CSV", command=self.export_optimization)
        menubar.add_cascade(label="Optimization", menu=optimization_menu)

        simulation_menu = tk.Menu(menubar, tearoff=False)
        simulation_menu.add_command(label="Run WNTR / EPANET INP", command=self.run_wntr_dialog)
        menubar.add_cascade(label="Hydraulics", menu=simulation_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Water Network AI Analyzer", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="Leakage-safe ML · Surrogate PSO · Optional WNTR",
            style="Subtitle.TLabel",
        ).pack(side=tk.LEFT, padx=(18, 0), pady=(6, 0))

        toolbar = ttk.Frame(self.root, padding=(16, 0, 16, 10))
        toolbar.pack(fill=tk.X)
        for label, command in [
            ("Load CSV", self.load_csv),
            ("Save CSV", self.save_dataset),
            ("Train Critical Model", self.train_critical_model),
            ("Predict", self.predict_critical_points_dialog),
            ("Run PSO", self.run_pso),
            ("Analyze Downstream", self.analyze_downstream),
            ("WNTR INP", self.run_wntr_dialog),
        ]:
            ttk.Button(toolbar, text=label, command=command).pack(side=tk.LEFT, padx=(0, 8))

        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=180)
        self.progress.pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 10))

        self.overview_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.optimization_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.overview_tab, text="Overview")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.optimization_tab, text="Optimization")
        self.notebook.add(self.log_tab, text="Log")

        self._build_overview_tab()
        self._build_data_tab()
        self._build_model_tab()
        self._build_optimization_tab()
        self._build_log_tab()

        status_frame = ttk.Frame(self.root, padding=(16, 0, 16, 10))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar()
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT)

    def _build_overview_tab(self) -> None:
        container = ttk.Frame(self.overview_tab, padding=16)
        container.pack(fill=tk.BOTH, expand=True)
        ttk.Label(container, text="Dataset & Schema", style="Section.TLabel").pack(anchor="w")
        self.overview_text = tk.Text(container, wrap=tk.WORD, height=20, font=("Consolas", 10))
        self.overview_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.overview_text.configure(state=tk.DISABLED)

    def _build_data_tab(self) -> None:
        controls = ttk.Frame(self.data_tab, padding=(10, 10, 10, 4))
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="Filter rows:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        entry = ttk.Entry(controls, textvariable=self.filter_var, width=35)
        entry.pack(side=tk.LEFT, padx=(8, 8))
        entry.bind("<KeyRelease>", lambda _event: self._populate_data_table())
        ttk.Label(controls, text="Double-click a cell to edit.").pack(side=tk.LEFT, padx=(12, 0))

        frame = ttk.Frame(self.data_tab, padding=(10, 4, 10, 10))
        frame.pack(fill=tk.BOTH, expand=True)
        self.data_tree = ttk.Treeview(frame, show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.data_tree.bind("<Double-1>", self._edit_cell)

    def _build_model_tab(self) -> None:
        top = ttk.Frame(self.model_tab, padding=12)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Train / Retrain", command=self.train_critical_model).pack(side=tk.LEFT)
        ttk.Button(top, text="Evaluation Plot", command=self.show_model_evaluation).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Feature Importance", command=self.show_feature_importance).pack(side=tk.LEFT)

        body = ttk.Panedwindow(self.model_tab, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)

        ttk.Label(left, text="Metrics", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.metrics_tree = ttk.Treeview(left, columns=("Metric", "Value"), show="headings", height=12)
        self.metrics_tree.heading("Metric", text="Metric")
        self.metrics_tree.heading("Value", text="Value")
        self.metrics_tree.column("Metric", width=160)
        self.metrics_tree.column("Value", width=140)
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right, text="Per-target Evaluation / Model Details", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.model_text = tk.Text(right, wrap=tk.WORD, font=("Consolas", 10))
        self.model_text.pack(fill=tk.BOTH, expand=True)
        self.model_text.configure(state=tk.DISABLED)

    def _build_optimization_tab(self) -> None:
        top = ttk.Frame(self.optimization_tab, padding=12)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Run PSO", command=self.run_pso).pack(side=tk.LEFT)
        ttk.Button(top, text="Optimization Plot", command=self.show_optimization_plot).pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Convergence", command=self.show_convergence_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Export CSV", command=self.export_optimization).pack(side=tk.LEFT, padx=8)

        frame = ttk.Frame(self.optimization_tab, padding=(12, 0, 12, 12))
        frame.pack(fill=tk.BOTH, expand=True)
        cols = ("Hour", "Demand", "Objective", "DownMin", "DownMean", "DownMax", "CriticalMin")
        self.optim_tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col in cols:
            self.optim_tree.heading(col, text=col)
            self.optim_tree.column(col, width=120, anchor="center")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.optim_tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.optim_tree.xview)
        self.optim_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.optim_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _build_log_tab(self) -> None:
        frame = ttk.Frame(self.log_tab, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(frame, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, f"Log file: {Path(LOG_FILE).resolve()}\n")
        self.log_text.configure(state=tk.DISABLED)

    # --------------------------- generic helpers ---------------------------

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self._append_log(text)

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log_text"):
            return
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {text}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_working(self, working: bool, status: Optional[str] = None) -> None:
        self._working = working
        if working:
            self.progress.start(12)
        else:
            self.progress.stop()
        if status:
            self._set_status(status)

    def _run_task(
        self,
        label: str,
        function: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        if self._working:
            messagebox.showinfo("Busy", "Another operation is still running.")
            return
        self._set_working(True, label)

        def worker() -> None:
            try:
                value = function()
            except Exception as exc:  # keep Tk calls on main thread
                logger.exception("Background task failed: %s", label)
                self.root.after(0, lambda exc=exc, label=label: self._task_error(label, exc))
                return
            self.root.after(0, lambda value=value, cb=on_success: self._task_success(cb, value))

        threading.Thread(target=worker, daemon=True).start()

    def _task_error(self, label: str, exc: Exception) -> None:
        self._set_working(False, f"Failed: {label}")
        messagebox.showerror("Error", f"{label}\n\n{type(exc).__name__}: {exc}")

    def _task_success(self, callback: Callable[[Any], None], value: Any) -> None:
        try:
            callback(value)
        finally:
            self._set_working(False)

    def _require_data(self) -> WaterNetworkData:
        if self.data is None:
            raise RuntimeError("Load a CSV dataset first.")
        return self.data

    @staticmethod
    def _numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        out = df.loc[:, list(columns)].copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.replace([np.inf, -np.inf], np.nan)

    # --------------------------- data actions ---------------------------

    def load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Water Network CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            raw = CSVService.load(path)
            schema = SchemaDetector.detect(raw)
            self.data = WaterNetworkData(raw=raw, schema=schema, source_path=Path(path))
            self.critical_model = XGBoostRegressionService(self.config)
            self.downstream_model = XGBoostRegressionService(self.config)
            self.optimization_result = None
            self._refresh_all()
            self._set_status(f"Loaded {Path(path).name}: {len(raw)} rows x {len(raw.columns)} columns")
        except Exception as exc:
            logger.exception("Load CSV failed")
            messagebox.showerror("Load Error", str(exc))

    def save_dataset(self) -> None:
        if self.data is None:
            messagebox.showerror("Error", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Dataset",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        try:
            CSVService.save(self.data.raw, path)
            self._set_status(f"Saved dataset: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _refresh_all(self) -> None:
        self._refresh_overview()
        self._populate_data_table()
        self._refresh_model_view()
        self._refresh_optimization_view()

    def _refresh_overview(self) -> None:
        self.overview_text.configure(state=tk.NORMAL)
        self.overview_text.delete("1.0", tk.END)
        if self.data is None:
            self.overview_text.insert(tk.END, "No dataset loaded.")
        else:
            raw = self.data.raw
            schema = self.data.schema
            numeric = raw.apply(pd.to_numeric, errors="coerce")
            missing = int(numeric.isna().sum().sum())
            content = [
                f"Source: {self.data.source_path or 'In-memory'}",
                f"Shape: {len(raw)} rows x {len(raw.columns)} columns",
                f"Numeric/missing cells after coercion: {missing}",
                "",
                "Detected schema",
                "---------------",
                f"PRV columns ({len(schema.prv_columns)}): {schema.prv_columns}",
                f"Point After Valve ({len(schema.point_after_valve_columns)}): {schema.point_after_valve_columns}",
                f"Critical Point ({len(schema.critical_point_columns)}): {schema.critical_point_columns}",
                f"Demand/Deby: {schema.demand_column}",
                f"Other columns ({len(schema.ignored_columns)}): {schema.ignored_columns}",
                "",
                "Model roles",
                "-----------",
                "Critical model: [Point After Valve + Demand] -> [Critical Points]",
                "PSO surrogate: [PRV settings + Demand] -> [Point After Valve pressures]",
                "PSO uses the learned surrogate; it is not a hydraulic solver.",
                "WNTR/EPANET is used only when a real INP file is supplied.",
            ]
            self.overview_text.insert(tk.END, "\n".join(content))
        self.overview_text.configure(state=tk.DISABLED)

    def _populate_data_table(self) -> None:
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        if self.data is None:
            return

        df = self.data.raw
        query = self.filter_var.get().strip().lower() if hasattr(self, "filter_var") else ""
        display = df
        if query:
            mask = df.astype(str).apply(lambda col: col.str.lower().str.contains(query, na=False)).any(axis=1)
            display = df.loc[mask]

        self._table_column_order = list(df.columns)
        self.data_tree.configure(columns=self._table_column_order)
        for col in self._table_column_order:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120, minwidth=80, anchor="center", stretch=False)

        for idx, row in display.head(5000).iterrows():
            values = ["" if pd.isna(row[c]) else row[c] for c in self._table_column_order]
            self.data_tree.insert("", "end", iid=str(idx), values=values)

    def _edit_cell(self, event: tk.Event) -> None:
        if self.data is None:
            return
        region = self.data_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        item_id = self.data_tree.identify_row(event.y)
        column_id = self.data_tree.identify_column(event.x)
        if not item_id or not column_id:
            return
        col_index = int(column_id[1:]) - 1
        if col_index >= len(self._table_column_order):
            return
        column_name = self._table_column_order[col_index]
        try:
            df_index: Any = int(item_id) if item_id.isdigit() and int(item_id) in self.data.raw.index else item_id
            current = self.data.raw.at[df_index, column_name]
        except Exception:
            return

        value = simpledialog.askstring("Edit Cell", f"{column_name}", initialvalue=str(current))
        if value is None:
            return
        self.data.raw.at[df_index, column_name] = value
        # Data changes invalidate trained models and optimization results.
        self.critical_model = XGBoostRegressionService(self.config)
        self.downstream_model = XGBoostRegressionService(self.config)
        self.optimization_result = None
        self._refresh_all()
        self._set_status(f"Edited row {df_index}, column {column_name}. Models invalidated.")

    # --------------------------- ML actions ---------------------------

    def train_critical_model(self) -> None:
        try:
            data = self._require_data()
            SchemaDetector.validate_for_critical_model(data.schema)
            features = data.schema.point_after_valve_columns + [data.schema.demand_column]  # type: ignore[list-item]
            targets = data.schema.critical_point_columns
        except Exception as exc:
            messagebox.showerror("Cannot Train", str(exc))
            return

        def task() -> RegressionResult:
            return self.critical_model.train(data.raw, features, targets, tune=True)

        def done(result: RegressionResult) -> None:
            self._refresh_model_view()
            self.notebook.select(self.model_tab)
            self._set_status(
                f"Critical model trained | R²={result.metrics['R2']:.3f} | RMSE={result.metrics['RMSE']:.3f}"
            )
            messagebox.showinfo(
                "Training Complete",
                f"Hold-out R²: {result.metrics['R2']:.4f}\n"
                f"Hold-out RMSE: {result.metrics['RMSE']:.4f}\n"
                f"MAE: {result.metrics['MAE']:.4f}\n"
                f"Training time: {result.training_seconds:.2f} s",
            )

        self._run_task("Training leakage-safe critical-point XGBoost model...", task, done)

    def _refresh_model_view(self) -> None:
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        self.model_text.configure(state=tk.NORMAL)
        self.model_text.delete("1.0", tk.END)

        result = self.critical_model.result
        if result is None:
            self.model_text.insert(tk.END, "No critical-point model trained yet.")
            self.model_text.configure(state=tk.DISABLED)
            return

        for name, value in result.metrics.items():
            shown = "nan" if not np.isfinite(value) else f"{value:.6f}"
            self.metrics_tree.insert("", "end", values=(name, shown))
        self.metrics_tree.insert("", "end", values=("Training seconds", f"{result.training_seconds:.2f}"))
        if result.best_cv_score is not None:
            self.metrics_tree.insert("", "end", values=("Best CV neg-RMSE", f"{result.best_cv_score:.6f}"))

        lines = [
            f"Features: {result.feature_names}",
            f"Targets: {result.target_names}",
            "",
            "Per-target metrics:",
            result.per_target_metrics.to_string(index=False),
            "",
            "Best hyperparameters:",
            json.dumps(result.best_params, indent=2, default=str) if result.best_params else "Tuning skipped (small dataset).",
            "",
            "Leakage controls:",
            "- Train/test split occurs before pipeline fitting.",
            "- KNNImputer is fitted inside the sklearn Pipeline.",
            "- IQRClipper is fitted inside each CV fold.",
            "- Targets are never imputed.",
            "- No StandardScaler is used because XGBoost does not require feature scaling.",
        ]
        self.model_text.insert(tk.END, "\n".join(lines))
        self.model_text.configure(state=tk.DISABLED)

    def predict_critical_points_dialog(self) -> None:
        result = self.critical_model.result
        if result is None:
            messagebox.showerror("Error", "Train or load the critical-point model first.")
            return
        if self.data is None:
            messagebox.showerror("Error", "Load a dataset first.")
            return

        window = tk.Toplevel(self.root)
        window.title("Predict Critical Points")
        window.geometry("560x650")

        outer = ttk.Frame(window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="Enter feature values", style="Section.TLabel").pack(anchor="w")

        canvas = tk.Canvas(outer, highlightthickness=0)
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        form = ttk.Frame(canvas)
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=8)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        entries: Dict[str, tk.StringVar] = {}
        for feature in result.feature_names:
            row = ttk.Frame(form)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=feature, width=28).pack(side=tk.LEFT)
            numeric = pd.to_numeric(self.data.raw[feature], errors="coerce") if feature in self.data.raw else pd.Series(dtype=float)
            default = float(numeric.median()) if numeric.notna().any() else 0.0
            var = tk.StringVar(value=f"{default:.6g}")
            ttk.Entry(row, textvariable=var, width=22).pack(side=tk.LEFT)
            entries[feature] = var

        def do_predict() -> None:
            try:
                frame = pd.DataFrame([{name: float(var.get()) for name, var in entries.items()}])
                pred = self.critical_model.predict_frame(frame).iloc[0]
                text = "\n".join(f"{name}: {value:.4f}" for name, value in pred.items())
                messagebox.showinfo("Critical-Point Prediction", text, parent=window)
            except Exception as exc:
                messagebox.showerror("Prediction Error", str(exc), parent=window)

        ttk.Button(window, text="Predict", command=do_predict).pack(pady=(0, 12))

    def show_model_evaluation(self) -> None:
        result = self.critical_model.result
        if result is None or result.y_test.size == 0:
            messagebox.showerror("Error", "Train the critical-point model first.")
            return
        self._show_figure(PlotFactory.actual_vs_predicted(result), "Model Evaluation")

    def show_feature_importance(self) -> None:
        result = self.critical_model.result
        if result is None:
            messagebox.showerror("Error", "Train the critical-point model first.")
            return
        self._show_figure(PlotFactory.feature_importance(result), "Feature Importance")

    def save_critical_model(self) -> None:
        if self.critical_model.result is None:
            messagebox.showerror("Error", "No trained critical model to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".joblib",
            filetypes=[("Joblib model", "*.joblib")],
        )
        if not path:
            return
        try:
            self.critical_model.save(path)
            self._set_status(f"Saved critical model: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save Model Error", str(exc))

    def load_critical_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Joblib model", "*.joblib"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.critical_model.load(path)
            self._refresh_model_view()
            self._set_status(f"Loaded critical model: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Load Model Error", str(exc))

    # --------------------------- PSO actions ---------------------------

    def _derive_prv_bounds(self, data: WaterNetworkData) -> List[Tuple[float, float]]:
        frame = self._numeric(data.raw, data.schema.prv_columns)
        bounds: List[Tuple[float, float]] = []
        for col in data.schema.prv_columns:
            values = frame[col].dropna().to_numpy(dtype=float)
            if values.size < 2:
                bounds.append((self.config.min_prv, self.config.max_prv))
                continue
            low = float(np.quantile(values, 0.01))
            high = float(np.quantile(values, 0.99))
            spread = max(high - low, 1.0)
            low = max(self.config.min_prv, low - 0.08 * spread)
            high = min(self.config.max_prv, high + 0.08 * spread)
            if high - low < 0.5:
                midpoint = (high + low) / 2.0
                low = max(self.config.min_prv, midpoint - 0.5)
                high = min(self.config.max_prv, midpoint + 0.5)
            bounds.append((low, high))
        return bounds

    def run_pso(self) -> None:
        try:
            data = self._require_data()
            SchemaDetector.validate_for_pso(data.schema)
        except Exception as exc:
            messagebox.showerror("Cannot Optimize", str(exc))
            return

        # Ask how many rows/hours to optimize without demanding exactly 24.
        max_hours = min(self.config.optimization_hours, len(data.raw))
        hours = simpledialog.askinteger(
            "Optimization Hours",
            f"Number of sequential rows/hours to optimize (1-{max_hours}):",
            initialvalue=max_hours,
            minvalue=1,
            maxvalue=max_hours,
            parent=self.root,
        )
        if not hours:
            return

        def task() -> OptimizationResult:
            schema = data.schema
            demand_col = schema.demand_column
            assert demand_col is not None

            # Surrogate A: historical PRV + demand -> downstream/point-after-valve pressure.
            down_features = schema.prv_columns + [demand_col]
            self.downstream_model.train(
                data.raw,
                down_features,
                schema.point_after_valve_columns,
                tune=True,
            )

            # Surrogate B: downstream + demand -> critical pressure.
            critical_features = schema.point_after_valve_columns + [demand_col]
            self.critical_model.train(
                data.raw,
                critical_features,
                schema.critical_point_columns,
                tune=True,
            )

            bounds = self._derive_prv_bounds(data)
            demand_series = pd.to_numeric(data.raw[demand_col], errors="coerce")
            if demand_series.notna().sum() == 0:
                raise ValueError(f"Demand column '{demand_col}' has no numeric values.")
            demand_series = demand_series.interpolate(limit_direction="both").fillna(demand_series.median())

            prv_history = self._numeric(data.raw, schema.prv_columns)
            # For reference settings only; never used to fit test evaluation.
            prv_history = prv_history.interpolate(limit_direction="both")
            for col in prv_history.columns:
                if prv_history[col].isna().all():
                    prv_history[col] = (self.config.min_prv + self.config.max_prv) / 2.0
                else:
                    prv_history[col] = prv_history[col].fillna(prv_history[col].median())

            def downstream_predict(settings_matrix: np.ndarray, demand: float) -> np.ndarray:
                settings_matrix = np.atleast_2d(np.asarray(settings_matrix, dtype=float))
                frame = pd.DataFrame(settings_matrix, columns=schema.prv_columns)
                frame[demand_col] = float(demand)
                return self.downstream_model.predict_frame(frame).to_numpy(dtype=float)

            def critical_predict(downstream_matrix: np.ndarray, demand: float) -> np.ndarray:
                downstream_matrix = np.atleast_2d(np.asarray(downstream_matrix, dtype=float))
                frame = pd.DataFrame(downstream_matrix, columns=schema.point_after_valve_columns)
                frame[demand_col] = float(demand)
                return self.critical_model.predict_frame(frame).to_numpy(dtype=float)

            hour_results: List[PSOHourResult] = []
            previous: Optional[np.ndarray] = None
            start = time.perf_counter()

            for h in range(hours):
                demand = float(demand_series.iloc[h])
                reference = prv_history.iloc[h].to_numpy(dtype=float)
                settings, downstream, score, convergence = self.pso.optimize_hour(
                    demand=demand,
                    prv_bounds=bounds,
                    downstream_predictor=downstream_predict,
                    critical_predictor=critical_predict,
                    previous_settings=previous,
                    reference_settings=reference,
                )

                critical_input = {
                    name: float(value)
                    for name, value in zip(schema.point_after_valve_columns, downstream)
                }
                critical_input[demand_col] = demand
                critical_pred = self.critical_model.predict_frame(pd.DataFrame([critical_input]))
                critical = critical_pred.iloc[0].to_numpy(dtype=float)

                hour_results.append(
                    PSOHourResult(
                        hour=h + 1,
                        demand=demand,
                        prv_settings=settings,
                        downstream_pressures=downstream,
                        critical_pressures=critical,
                        objective=float(score),
                        convergence=list(convergence),
                    )
                )
                previous = settings.copy()

            return OptimizationResult(
                hours=hour_results,
                prv_names=schema.prv_columns,
                downstream_names=schema.point_after_valve_columns,
                critical_names=schema.critical_point_columns,
                total_seconds=time.perf_counter() - start,
            )

        def done(result: OptimizationResult) -> None:
            self.optimization_result = result
            self._refresh_model_view()
            self._refresh_optimization_view()
            self.notebook.select(self.optimization_tab)
            violations = sum(
                int(np.sum((x.downstream_pressures < self.config.min_pressure) | (x.downstream_pressures > self.config.max_pressure)))
                for x in result.hours
            )
            self._set_status(
                f"PSO complete | {len(result.hours)} hours | downstream constraint violations={violations} | {result.total_seconds:.2f}s"
            )
            messagebox.showinfo(
                "Optimization Complete",
                f"Optimized hours: {len(result.hours)}\n"
                f"PRVs: {len(result.prv_names)}\n"
                f"Downstream constraint violations: {violations}\n"
                f"Optimization time: {result.total_seconds:.2f} s\n\n"
                "Method: data-driven surrogate PSO (not a hydraulic solver).",
            )

        self._run_task("Training surrogate models and running PSO...", task, done)

    def _refresh_optimization_view(self) -> None:
        for item in self.optim_tree.get_children():
            self.optim_tree.delete(item)
        if self.optimization_result is None:
            return
        for item in self.optimization_result.hours:
            critical_min = (
                float(np.min(item.critical_pressures))
                if item.critical_pressures is not None and len(item.critical_pressures)
                else float("nan")
            )
            self.optim_tree.insert(
                "",
                "end",
                values=(
                    item.hour,
                    f"{item.demand:.4f}",
                    f"{item.objective:.4f}",
                    f"{np.min(item.downstream_pressures):.3f}",
                    f"{np.mean(item.downstream_pressures):.3f}",
                    f"{np.max(item.downstream_pressures):.3f}",
                    f"{critical_min:.3f}",
                ),
            )

    def export_optimization(self) -> None:
        if self.optimization_result is None:
            messagebox.showerror("Error", "Run PSO optimization first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Export Optimization Results",
        )
        if not path:
            return
        try:
            CSVService.save(self.optimization_result.to_dataframe(), path)
            self._set_status(f"Exported optimization results: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def show_optimization_plot(self) -> None:
        if self.optimization_result is None:
            messagebox.showerror("Error", "Run PSO optimization first.")
            return
        self._show_figure(
            PlotFactory.pso_summary(self.optimization_result, self.config),
            "PSO Optimization Summary",
        )

    def show_convergence_dialog(self) -> None:
        if self.optimization_result is None:
            messagebox.showerror("Error", "Run PSO optimization first.")
            return
        max_hour = len(self.optimization_result.hours)
        hour = simpledialog.askinteger(
            "PSO Convergence",
            f"Hour to display (1-{max_hour}):",
            minvalue=1,
            maxvalue=max_hour,
            initialvalue=1,
            parent=self.root,
        )
        if not hour:
            return
        self._show_figure(
            PlotFactory.pso_convergence(self.optimization_result.hours[hour - 1]),
            f"PSO Convergence - Hour {hour}",
        )

    # --------------------------- analysis / hydraulics ---------------------------

    def analyze_downstream(self) -> None:
        if self.data is None or not self.data.schema.point_after_valve_columns:
            messagebox.showerror("Error", "No Point After Valve columns detected.")
            return
        frame = self.data.point_after_valve_data
        stats = frame.apply(pd.to_numeric, errors="coerce").describe().T

        window = tk.Toplevel(self.root)
        window.title("Point After Valve Analysis")
        window.geometry("900x600")
        text = tk.Text(window, font=("Consolas", 10), wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, stats.to_string())
        text.configure(state=tk.DISABLED)

        ttk.Button(
            window,
            text="Show Boxplot",
            command=lambda: self._show_figure(
                PlotFactory.descriptive_boxplot(frame, "Point After Valve Distribution"),
                "Downstream Distribution",
            ),
        ).pack(pady=8)

    def run_wntr_dialog(self) -> None:
        if not WNTRService.available():
            messagebox.showinfo(
                "WNTR Not Installed",
                "Real hydraulic simulation requires WNTR.\n\nInstall it with:\n\npip install wntr",
            )
            return

        path = filedialog.askopenfilename(
            title="Select EPANET INP File",
            filetypes=[("EPANET INP", "*.inp"), ("All files", "*.*")],
        )
        if not path:
            return

        nodes: Optional[List[str]] = None
        if self.data is not None and self.data.schema.critical_point_columns:
            nodes = self.data.schema.critical_point_columns

        def task() -> pd.DataFrame:
            return WNTRService.simulate_pressures(path, nodes)

        def done(frame: pd.DataFrame) -> None:
            self._set_status(f"WNTR simulation complete: {len(frame)} time steps x {len(frame.columns)} nodes")
            self._show_wntr_results(frame)

        self._run_task("Running real WNTR/EPANET hydraulic simulation...", task, done)

    def _show_wntr_results(self, pressure: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in pressure.columns[:12]:
            ax.plot(pressure.index, pressure[col], label=col)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Pressure")
        ax.set_title("WNTR / EPANET Pressure Simulation")
        ax.grid(alpha=0.25)
        if len(pressure.columns) <= 12:
            ax.legend(ncol=2)
        fig.tight_layout()
        self._show_figure(fig, "WNTR Simulation")

    # --------------------------- figure window ---------------------------

    def _show_figure(self, fig: plt.Figure, title: str) -> None:
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("1050x720")

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()

        def save_plot() -> None:
            path = filedialog.asksaveasfilename(
                parent=window,
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            )
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight")

        ttk.Button(window, text="Save Plot", command=save_plot).pack(pady=6)

        def close() -> None:
            plt.close(fig)
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close)

    # --------------------------- misc ---------------------------

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Water Network AI Analyzer v5.0\n\n"
            "Leakage-safe XGBoost regression\n"
            "Data-driven PRV optimization with PSO\n"
            "Critical-point prediction\n"
            "Optional real WNTR/EPANET simulation\n\n"
            "PSO is explicitly a learned surrogate optimizer and is not presented as a hydraulic solver.",
        )

    def _on_close(self) -> None:
        if self._working:
            if not messagebox.askyesno("Exit", "An operation is running. Exit anyway?"):
                return
        plt.close("all")
        self.root.destroy()


# -----------------------------------------------------------------------------
# CLI smoke-test helper (useful for CI and GitHub Actions)
# -----------------------------------------------------------------------------

def run_headless_smoke_test() -> int:
    """Train the core ML/PSO services on deterministic synthetic data."""
    rng = np.random.default_rng(42)
    n = 120
    prv1 = rng.uniform(22, 48, n)
    prv2 = rng.uniform(20, 45, n)
    demand = rng.uniform(5, 25, n)
    down1 = 0.55 * prv1 + 0.20 * prv2 - 0.22 * demand + rng.normal(0, 0.6, n)
    down2 = 0.25 * prv1 + 0.65 * prv2 - 0.18 * demand + rng.normal(0, 0.6, n)
    crit1 = 0.58 * down1 + 0.35 * down2 - 0.08 * demand + rng.normal(0, 0.45, n)
    crit2 = 0.30 * down1 + 0.62 * down2 - 0.05 * demand + rng.normal(0, 0.45, n)

    df = pd.DataFrame(
        {
            "PRV-1": prv1,
            "PRV-2": prv2,
            "J-101-B": down1,
            "J-102-B": down2,
            "P-676": demand,
            "J-201": crit1,
            "J-202": crit2,
        }
    )
    # Explicitly test missing values and an extreme feature outlier.
    df.loc[3, "J-101-B"] = np.nan
    df.loc[7, "P-676"] = np.nan
    df.loc[10, "PRV-1"] = 500.0

    config = AppConfig(search_iterations=2, cv_folds=3, pso_particles=8, pso_iterations=8)
    schema = SchemaDetector.detect(df)
    assert len(schema.prv_columns) == 2
    assert len(schema.point_after_valve_columns) == 2
    assert len(schema.critical_point_columns) == 2
    assert schema.demand_column == "P-676"

    critical = XGBoostRegressionService(config)
    result = critical.train(
        df,
        schema.point_after_valve_columns + [schema.demand_column],  # type: ignore[list-item]
        schema.critical_point_columns,
        tune=True,
    )
    assert np.isfinite(result.metrics["RMSE"])

    downstream = XGBoostRegressionService(config)
    downstream.train(
        df,
        schema.prv_columns + [schema.demand_column],  # type: ignore[list-item]
        schema.point_after_valve_columns,
        tune=True,
    )

    def predictor(settings_matrix: np.ndarray, d: float) -> np.ndarray:
        settings_matrix = np.atleast_2d(np.asarray(settings_matrix, dtype=float))
        frame = pd.DataFrame(settings_matrix, columns=schema.prv_columns)
        frame[schema.demand_column] = d  # type: ignore[index]
        return downstream.predict_frame(frame).to_numpy(dtype=float)

    def critical_predictor(downstream_matrix: np.ndarray, d: float) -> np.ndarray:
        downstream_matrix = np.atleast_2d(np.asarray(downstream_matrix, dtype=float))
        frame = pd.DataFrame(downstream_matrix, columns=schema.point_after_valve_columns)
        frame[schema.demand_column] = d  # type: ignore[index]
        return critical.predict_frame(frame).to_numpy(dtype=float)

    optimizer = SurrogatePSOOptimizer(config)
    settings, pressures, score, history = optimizer.optimize_hour(
        demand=float(np.nanmedian(demand)),
        prv_bounds=[(20.0, 50.0), (18.0, 48.0)],
        downstream_predictor=predictor,
        critical_predictor=critical_predictor,
    )
    assert settings.shape == (2,)
    assert pressures.shape == (2,)
    assert np.isfinite(score)
    assert len(history) == config.pso_iterations + 1

    print("SMOKE TEST PASSED")
    print(f"Critical model RMSE: {result.metrics['RMSE']:.4f}")
    print(f"Critical model R2:   {result.metrics['R2']:.4f}")
    print(f"PSO score:           {score:.4f}")
    return 0


def main() -> int:
    if "--smoke-test" in sys.argv:
        return run_headless_smoke_test()

    root = tk.Tk()
    WaterNetworkApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
