"""
Critical-point pressure regression for Water Network AI Analyzer.

This module owns the complete supervised-learning workflow for predicting
critical-point pressures from downstream pressure measurements and demand.

Key properties
--------------
- train/test split happens before any learned preprocessing is fitted
- KNN imputation and IQR clipping live inside an sklearn Pipeline
- hyperparameter search therefore remains leakage-safe inside CV folds
- single-target and multi-target regression are both supported
- trained models can be saved and loaded with joblib
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import math
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from ..config import AppConfig, DEFAULT_CONFIG
from ..data.preprocessing import (
    build_model_pipeline,
    coerce_numeric_features,
    remove_missing_target_rows,
)


MODEL_FILE_VERSION = 1
MODEL_TYPE = "critical_pressure"


@dataclass
class RegressionResult:
    """Artifacts and metrics produced by a critical-pressure training run."""

    pipeline: Pipeline
    feature_names: List[str]
    target_names: List[str]

    metrics: Dict[str, float]
    per_target_metrics: pd.DataFrame
    feature_importance: pd.DataFrame

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_cv_score: Optional[float] = None
    training_seconds: float = 0.0

    y_test: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    y_pred: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    test_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)


class CriticalPressureModel:
    """
    Leakage-safe XGBoost service for critical-point pressure prediction.

    Typical feature set
    -------------------
    - downstream / point-after-valve pressures
    - network demand

    Typical targets
    ---------------
    - J-101
    - J-205
    - Critical_Point_1
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.result: Optional[RegressionResult] = None

    # ------------------------------------------------------------------
    # Estimator construction
    # ------------------------------------------------------------------

    def _base_regressor(self) -> xgb.XGBRegressor:
        """Return the base XGBoost regressor used by the project."""

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=self.config.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    def _build_pipeline(
        self,
        multi_output: bool,
    ) -> Pipeline:
        """Create the complete preprocessing + estimator pipeline."""

        estimator: Any = self._base_regressor()

        if multi_output:
            estimator = MultiOutputRegressor(
                estimator,
                n_jobs=1,
            )

        return build_model_pipeline(
            estimator=estimator,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        *,
        tune: bool = True,
    ) -> RegressionResult:
        """
        Train the critical-pressure prediction model.

        Parameters
        ----------
        df:
            Source dataframe.
        feature_names:
            Input feature columns. Normally downstream pressures plus demand.
        target_names:
            One or more critical-point pressure columns.
        tune:
            Run RandomizedSearchCV when enough training data is available.

        Returns
        -------
        RegressionResult
            Trained pipeline, metrics, predictions and diagnostic metadata.
        """

        self._validate_training_request(
            df,
            feature_names,
            target_names,
        )

        feature_names = list(feature_names)
        target_names = list(target_names)

        X_df = coerce_numeric_features(
            df.loc[:, feature_names]
        )

        y_df = coerce_numeric_features(
            df.loc[:, target_names]
        )

        # Targets are never imputed. Any row with an incomplete label vector
        # is removed, while missing feature values remain for KNNImputer.
        X_df, y_df = remove_missing_target_rows(
            X_df,
            y_df,
        )

        if len(X_df) < self.config.min_training_rows:
            raise ValueError(
                f"Insufficient labeled rows: {len(X_df)}. "
                f"At least {self.config.min_training_rows} are required."
            )

        # An entirely missing feature cannot be recovered meaningfully.
        all_missing = [
            column
            for column in feature_names
            if X_df[column].isna().all()
        ]

        if all_missing:
            raise ValueError(
                "Feature columns contain only missing values: "
                + ", ".join(all_missing)
            )

        # Protect the desktop application from accidentally training on a
        # very large table. Sampling remains deterministic.
        if len(X_df) > self.config.max_training_rows:
            sampled_positions = X_df.sample(
                n=self.config.max_training_rows,
                random_state=self.config.random_state,
            ).index

            X_df = X_df.loc[sampled_positions].reset_index(drop=True)
            y_df = y_df.loc[sampled_positions].reset_index(drop=True)

        row_ids = np.arange(len(X_df), dtype=int)

        (
            X_train,
            X_test,
            y_train_df,
            y_test_df,
            _,
            test_indices,
        ) = train_test_split(
            X_df,
            y_df,
            row_ids,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            shuffle=True,
        )

        # It is possible for a sparse feature to have observations only in the
        # hold-out split. Catch that case before KNNImputer sees training data.
        train_all_missing = [
            column
            for column in feature_names
            if X_train[column].isna().all()
        ]

        if train_all_missing:
            raise ValueError(
                "After train/test splitting, these training features contain "
                "only missing values: "
                + ", ".join(train_all_missing)
                + ". Provide more complete data or remove those features."
            )

        multi_output = len(target_names) > 1
        pipeline = self._build_pipeline(
            multi_output=multi_output
        )

        y_train = self._prepare_target_array(
            y_train_df,
            multi_output=multi_output,
        )
        y_test = self._prepare_target_array(
            y_test_df,
            multi_output=multi_output,
        )

        start = time.perf_counter()

        fitted_pipeline: Pipeline
        best_params: Dict[str, Any] = {}
        best_cv_score: Optional[float] = None

        if tune and self._can_tune(len(X_train)):
            cv = KFold(
                n_splits=self._cv_folds(len(X_train)),
                shuffle=True,
                random_state=self.config.random_state,
            )

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=(
                    self.config.xgb_param_distributions(
                        multi_output=multi_output
                    )
                ),
                n_iter=self.config.search_iterations,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                random_state=self.config.random_state,
                n_jobs=-1,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )

            search.fit(
                X_train,
                y_train,
            )

            fitted_pipeline = search.best_estimator_
            best_params = dict(search.best_params_)
            best_cv_score = float(search.best_score_)

        else:
            fitted_pipeline = pipeline
            fitted_pipeline.fit(
                X_train,
                y_train,
            )

        predictions = fitted_pipeline.predict(
            X_test
        )

        elapsed = time.perf_counter() - start

        y_test_2d = self._as_2d_targets(
            y_test,
            len(target_names),
        )
        y_pred_2d = self._as_2d_targets(
            predictions,
            len(target_names),
        )

        metrics = self._overall_metrics(
            y_test_2d,
            y_pred_2d,
        )

        per_target = self._per_target_metrics(
            y_test_2d,
            y_pred_2d,
            target_names,
        )

        importance = self._feature_importance(
            fitted_pipeline,
            feature_names,
        )

        result = RegressionResult(
            pipeline=fitted_pipeline,
            feature_names=feature_names,
            target_names=target_names,
            metrics=metrics,
            per_target_metrics=per_target,
            feature_importance=importance,
            best_params=best_params,
            best_cv_score=best_cv_score,
            training_seconds=elapsed,
            y_test=y_test_2d,
            y_pred=y_pred_2d,
            test_indices=np.asarray(
                test_indices,
                dtype=int,
            ),
        )

        self.result = result
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_frame(
        self,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict critical-point pressures from a dataframe of features."""

        result = self._require_result()

        if not isinstance(values, pd.DataFrame):
            raise TypeError(
                "Prediction input must be a pandas DataFrame."
            )

        missing = [
            column
            for column in result.feature_names
            if column not in values.columns
        ]

        if missing:
            raise ValueError(
                "Prediction input is missing required features: "
                + ", ".join(missing)
            )

        X = coerce_numeric_features(
            values.loc[:, result.feature_names]
        )

        predictions = result.pipeline.predict(X)

        prediction_array = self._as_2d_targets(
            predictions,
            len(result.target_names),
        )

        return pd.DataFrame(
            prediction_array,
            columns=result.target_names,
            index=values.index,
        )

    def predict_one(
        self,
        values: Dict[str, Any],
    ) -> Dict[str, float]:
        """Convenience wrapper for predicting a single observation."""

        frame = pd.DataFrame([values])
        prediction = self.predict_frame(frame).iloc[0]

        return {
            str(name): float(value)
            for name, value in prediction.items()
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save the trained model and its metadata to a Joblib file."""

        result = self._require_result()
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_file_version": MODEL_FILE_VERSION,
            "model_type": MODEL_TYPE,
            "pipeline": result.pipeline,
            "feature_names": result.feature_names,
            "target_names": result.target_names,
            "metrics": result.metrics,
            "per_target_metrics": result.per_target_metrics,
            "feature_importance": result.feature_importance,
            "best_params": result.best_params,
            "best_cv_score": result.best_cv_score,
            "training_seconds": result.training_seconds,
        }

        joblib.dump(
            payload,
            path,
        )

    def load(
        self,
        path: str | Path,
    ) -> RegressionResult:
        """Load a previously saved critical-pressure model."""

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        payload = joblib.load(path)

        if not isinstance(payload, dict):
            raise ValueError(
                "Invalid model file: expected a dictionary payload."
            )

        required = {
            "pipeline",
            "feature_names",
            "target_names",
        }

        if not required.issubset(payload):
            raise ValueError(
                "Invalid model file: required fields are missing."
            )

        model_type = payload.get("model_type")
        if model_type not in (None, MODEL_TYPE):
            raise ValueError(
                f"Model file type is '{model_type}', expected '{MODEL_TYPE}'."
            )

        target_names = list(payload["target_names"])

        result = RegressionResult(
            pipeline=payload["pipeline"],
            feature_names=list(payload["feature_names"]),
            target_names=target_names,
            metrics=dict(payload.get("metrics", {})),
            per_target_metrics=self._restore_dataframe(
                payload.get("per_target_metrics")
            ),
            feature_importance=self._restore_dataframe(
                payload.get("feature_importance")
            ),
            best_params=dict(payload.get("best_params", {})),
            best_cv_score=payload.get("best_cv_score"),
            training_seconds=float(
                payload.get("training_seconds", 0.0)
            ),
            y_test=np.empty(
                (0, len(target_names)),
                dtype=float,
            ),
            y_pred=np.empty(
                (0, len(target_names)),
                dtype=float,
            ),
            test_indices=np.array([], dtype=int),
        )

        self.result = result
        return result

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_importance(
        pipeline: Pipeline,
        feature_names: Sequence[str],
    ) -> pd.DataFrame:
        """Extract or average XGBoost feature importance values."""

        model = pipeline.named_steps["model"]
        values: Optional[np.ndarray] = None

        if hasattr(model, "feature_importances_"):
            values = np.asarray(
                model.feature_importances_,
                dtype=float,
            )

        elif isinstance(model, MultiOutputRegressor):
            estimators = getattr(
                model,
                "estimators_",
                [],
            )

            importances = [
                np.asarray(
                    estimator.feature_importances_,
                    dtype=float,
                )
                for estimator in estimators
                if hasattr(
                    estimator,
                    "feature_importances_",
                )
            ]

            if importances:
                values = np.mean(
                    np.vstack(importances),
                    axis=0,
                )

        if (
            values is None
            or len(values) != len(feature_names)
        ):
            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Importance",
                ]
            )

        frame = pd.DataFrame(
            {
                "Feature": list(feature_names),
                "Importance": values,
            }
        )

        return frame.sort_values(
            "Importance",
            ascending=False,
        ).reset_index(drop=True)

    @staticmethod
    def _overall_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate aggregate regression metrics."""

        mae = float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        )

        rmse = float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        )

        try:
            r2 = float(
                r2_score(
                    y_true,
                    y_pred,
                    multioutput="uniform_average",
                )
            )
        except ValueError:
            r2 = float("nan")

        nonzero = np.abs(y_true) > 1e-9

        if np.any(nonzero):
            mape = float(
                np.mean(
                    np.abs(
                        (
                            y_true[nonzero]
                            - y_pred[nonzero]
                        )
                        / y_true[nonzero]
                    )
                )
                * 100.0
            )
        else:
            mape = float("nan")

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
        }

    @staticmethod
    def _per_target_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Sequence[str],
    ) -> pd.DataFrame:
        """Calculate metrics separately for every critical point."""

        rows: List[Dict[str, float | str]] = []

        for index, target in enumerate(target_names):
            true_values = y_true[:, index]
            predicted_values = y_pred[:, index]

            try:
                target_r2 = float(
                    r2_score(
                        true_values,
                        predicted_values,
                    )
                )
            except ValueError:
                target_r2 = float("nan")

            nonzero = np.abs(true_values) > 1e-9

            if np.any(nonzero):
                target_mape = float(
                    np.mean(
                        np.abs(
                            (
                                true_values[nonzero]
                                - predicted_values[nonzero]
                            )
                            / true_values[nonzero]
                        )
                    )
                    * 100.0
                )
            else:
                target_mape = float("nan")

            rows.append(
                {
                    "Target": str(target),
                    "MAE": float(
                        mean_absolute_error(
                            true_values,
                            predicted_values,
                        )
                    ),
                    "RMSE": float(
                        math.sqrt(
                            mean_squared_error(
                                true_values,
                                predicted_values,
                            )
                        )
                    ),
                    "R2": target_r2,
                    "MAPE": target_mape,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_training_request(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str],
        target_names: Sequence[str],
    ) -> None:
        """Validate dataframe and requested feature/target columns."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot train from an empty dataframe."
            )

        feature_names = list(feature_names)
        target_names = list(target_names)

        if not feature_names:
            raise ValueError(
                "At least one feature column is required."
            )

        if not target_names:
            raise ValueError(
                "At least one target column is required."
            )

        duplicate_features = sorted(
            {
                name
                for name in feature_names
                if feature_names.count(name) > 1
            }
        )

        duplicate_targets = sorted(
            {
                name
                for name in target_names
                if target_names.count(name) > 1
            }
        )

        if duplicate_features:
            raise ValueError(
                "Duplicate feature names are not allowed: "
                + ", ".join(duplicate_features)
            )

        if duplicate_targets:
            raise ValueError(
                "Duplicate target names are not allowed: "
                + ", ".join(duplicate_targets)
            )

        overlap = sorted(
            set(feature_names)
            & set(target_names)
        )

        if overlap:
            raise ValueError(
                "Columns cannot be both features and targets: "
                + ", ".join(overlap)
            )

        requested = feature_names + target_names

        missing = [
            name
            for name in requested
            if name not in df.columns
        ]

        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing)
            )

    def _can_tune(
        self,
        n_train_rows: int,
    ) -> bool:
        """Return whether the training split is large enough for tuning."""

        folds = self._cv_folds(
            n_train_rows
        )

        return (
            folds >= 2
            and n_train_rows >= max(
                16,
                folds * 3,
            )
        )

    def _cv_folds(
        self,
        n_train_rows: int,
    ) -> int:
        """Choose a valid number of CV folds for the available data."""

        if n_train_rows < 2:
            return 0

        return min(
            self.config.cv_folds,
            n_train_rows,
        )

    @staticmethod
    def _prepare_target_array(
        y: pd.DataFrame,
        *,
        multi_output: bool,
    ) -> np.ndarray:
        """Convert target dataframe into sklearn-compatible target shape."""

        array = y.to_numpy(
            dtype=float,
        )

        if not multi_output:
            return array.ravel()

        return array

    @staticmethod
    def _as_2d_targets(
        values,
        n_targets: int,
    ) -> np.ndarray:
        """Normalize model output to shape (rows, targets)."""

        array = np.asarray(
            values,
            dtype=float,
        )

        if n_targets == 1:
            return array.reshape(-1, 1)

        return array.reshape(
            -1,
            n_targets,
        )

    def _require_result(self) -> RegressionResult:
        """Return the current model result or fail clearly."""

        if self.result is None:
            raise RuntimeError(
                "Critical-pressure model has not been trained or loaded."
            )

        return self.result

    @staticmethod
    def _restore_dataframe(
        value,
    ) -> pd.DataFrame:
        """Restore dataframe metadata from a persisted model payload."""

        if isinstance(value, pd.DataFrame):
            return value.copy()

        if value is None:
            return pd.DataFrame()

        return pd.DataFrame(value)
