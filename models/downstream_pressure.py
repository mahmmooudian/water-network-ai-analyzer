Downstream-pressure surrogate model for Water Network AI Analyzer.

This module learns:

    PRV Settings + Demand -> Downstream Pressure

It intentionally reuses the tested regression engine from
``critical_pressure.py`` instead of duplicating the entire XGBoost,
cross-validation, preprocessing, metrics, and feature-importance logic.

The trained surrogate is later consumed by the PSO optimizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from ..config import AppConfig, DEFAULT_CONFIG
from .critical_pressure import (
    CriticalPressureModel,
    RegressionResult,
)


MODEL_FILE_VERSION = 1
MODEL_TYPE = "downstream_pressure"


# Semantic alias: the underlying result structure is identical.
DownstreamRegressionResult = RegressionResult


class DownstreamPressureModel:
    """
    Data-driven surrogate for downstream pressure prediction.

    Inputs
    ------
    PRV settings + demand

    Targets
    -------
    One or more downstream / point-after-valve pressure columns.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG

        # Reuse the already tested leakage-safe regression engine.
        self._core = CriticalPressureModel(
            config=self.config
        )

        self.prv_columns: list[str] = []
        self.demand_column: Optional[str] = None

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def result(
        self,
    ) -> Optional[DownstreamRegressionResult]:
        """Return the trained/loaded surrogate result."""

        return self._core.result

    @property
    def is_ready(
        self,
    ) -> bool:
        """Return True when a surrogate model is available."""

        return self.result is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        prv_columns: Sequence[str],
        demand_column: str,
        target_names: Sequence[str],
        *,
        tune: bool = True,
    ) -> DownstreamRegressionResult:
        """
        Train the downstream-pressure surrogate.

        Parameters
        ----------
        df:
            Source water-network dataframe.
        prv_columns:
            Controllable PRV setting columns.
        demand_column:
            Network demand / flow column.
        target_names:
            Downstream pressure columns.
        tune:
            If True, enable RandomizedSearchCV through the shared
            regression engine.
        """

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        prv_columns = [
            str(column)
            for column in prv_columns
        ]

        target_names = [
            str(column)
            for column in target_names
        ]

        demand_column = str(
            demand_column
        ).strip()

        self._validate_training_request(
            df=df,
            prv_columns=prv_columns,
            demand_column=demand_column,
            target_names=target_names,
        )

        # Deterministic feature ordering is important because PSO particle
        # dimensions must match the exact order used during training.
        feature_names = (
            prv_columns
            + [demand_column]
        )

        result = self._core.train(
            df=df,
            feature_names=feature_names,
            target_names=target_names,
            tune=tune,
        )

        self.prv_columns = list(
            prv_columns
        )

        self.demand_column = (
            demand_column
        )

        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_frame(
        self,
        values: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict downstream pressure for one or more operating states.
        """

        self._require_ready()

        return self._core.predict_frame(
            values
        )

    def predict_one(
        self,
        values: Dict[str, Any],
    ) -> Dict[str, float]:
        """Predict one operating state."""

        self._require_ready()

        return self._core.predict_one(
            values
        )

    def predict_particles(
        self,
        particles: np.ndarray,
        demand: float,
    ) -> np.ndarray:
        """
        Vectorized prediction for PSO particles.

        Parameters
        ----------
        particles:
            Matrix with shape ``(n_particles, n_prvs)``.
        demand:
            Demand value for the current optimization period.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_particles, n_downstream_targets)``.
        """

        result = self._require_ready()

        particles = np.asarray(
            particles,
            dtype=float,
        )

        if particles.ndim != 2:
            raise ValueError(
                "particles must be a 2D array."
            )

        if not self.prv_columns:
            raise RuntimeError(
                "PRV metadata is unavailable."
            )

        if not self.demand_column:
            raise RuntimeError(
                "Demand-column metadata is unavailable."
            )

        if (
            particles.shape[1]
            != len(self.prv_columns)
        ):
            raise ValueError(
                "Particle dimension does not match "
                "the number of trained PRV columns."
            )

        if not np.isfinite(
            float(demand)
        ):
            raise ValueError(
                "demand must be finite."
            )

        frame = pd.DataFrame(
            particles,
            columns=self.prv_columns,
        )

        frame[
            self.demand_column
        ] = float(demand)

        # Reorder exactly as training features.
        frame = frame.loc[
            :,
            result.feature_names,
        ]

        prediction = (
            self._core
            .predict_frame(frame)
            .to_numpy(dtype=float)
        )

        return prediction

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save surrogate model plus PRV/demand metadata.
        """

        result = self._require_ready()

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model_file_version": MODEL_FILE_VERSION,
            "model_type": MODEL_TYPE,

            "pipeline": result.pipeline,
            "feature_names": list(
                result.feature_names
            ),
            "target_names": list(
                result.target_names
            ),

            "metrics": dict(
                result.metrics
            ),
            "per_target_metrics": (
                result.per_target_metrics
            ),
            "feature_importance": (
                result.feature_importance
            ),

            "best_params": dict(
                result.best_params
            ),
            "best_cv_score": (
                result.best_cv_score
            ),
            "training_seconds": float(
                result.training_seconds
            ),

            "prv_columns": list(
                self.prv_columns
            ),
            "demand_column": (
                self.demand_column
            ),
        }

        joblib.dump(
            payload,
            path,
        )

    def load(
        self,
        path: str | Path,
    ) -> DownstreamRegressionResult:
        """
        Load a previously saved surrogate.
        """

        path = Path(
            path
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        payload = joblib.load(
            path
        )

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Invalid downstream model file."
            )

        if (
            payload.get("model_type")
            != MODEL_TYPE
        ):
            raise ValueError(
                "The selected file is not a "
                "downstream-pressure surrogate."
            )

        required = {
            "pipeline",
            "feature_names",
            "target_names",
            "prv_columns",
            "demand_column",
        }

        missing = (
            required
            - set(payload)
        )

        if missing:
            raise ValueError(
                "Model file is missing fields: "
                + ", ".join(
                    sorted(missing)
                )
            )

        per_target = payload.get(
            "per_target_metrics"
        )

        if not isinstance(
            per_target,
            pd.DataFrame,
        ):
            per_target = pd.DataFrame(
                per_target
                if per_target is not None
                else []
            )

        importance = payload.get(
            "feature_importance"
        )

        if not isinstance(
            importance,
            pd.DataFrame,
        ):
            importance = pd.DataFrame(
                importance
                if importance is not None
                else []
            )

        result = RegressionResult(
            pipeline=payload[
                "pipeline"
            ],
            feature_names=list(
                payload[
                    "feature_names"
                ]
            ),
            target_names=list(
                payload[
                    "target_names"
                ]
            ),
            metrics=dict(
                payload.get(
                    "metrics",
                    {},
                )
            ),
            per_target_metrics=(
                per_target
            ),
            feature_importance=(
                importance
            ),
            best_params=dict(
                payload.get(
                    "best_params",
                    {},
                )
            ),
            best_cv_score=(
                payload.get(
                    "best_cv_score"
                )
            ),
            training_seconds=float(
                payload.get(
                    "training_seconds",
                    0.0,
                )
            ),
        )

        self._core.result = result

        self.prv_columns = list(
            payload[
                "prv_columns"
            ]
        )

        self.demand_column = str(
            payload[
                "demand_column"
            ]
        )

        self._validate_loaded_metadata()

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_training_request(
        *,
        df: pd.DataFrame,
        prv_columns: Sequence[str],
        demand_column: str,
        target_names: Sequence[str],
    ) -> None:
        """Validate semantic inputs before delegating to the core model."""

        if df.empty:
            raise ValueError(
                "Cannot train on an empty dataframe."
            )

        if not prv_columns:
            raise ValueError(
                "At least one PRV setting column is required."
            )

        if not demand_column:
            raise ValueError(
                "A demand column is required."
            )

        if not target_names:
            raise ValueError(
                "At least one downstream pressure target is required."
            )

        if (
            len(set(prv_columns))
            != len(prv_columns)
        ):
            raise ValueError(
                "PRV column names must be unique."
            )

        if (
            len(set(target_names))
            != len(target_names)
        ):
            raise ValueError(
                "Target names must be unique."
            )

        if demand_column in prv_columns:
            raise ValueError(
                "Demand column cannot also be a PRV setting."
            )

        feature_names = (
            list(prv_columns)
            + [demand_column]
        )

        overlap = (
            set(feature_names)
            & set(target_names)
        )

        if overlap:
            raise ValueError(
                "Surrogate inputs and targets overlap: "
                + ", ".join(
                    sorted(overlap)
                )
            )

        required = (
            feature_names
            + list(target_names)
        )

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing)
            )

    def _validate_loaded_metadata(
        self,
    ) -> None:
        """Ensure persisted feature metadata is internally consistent."""

        result = self._require_ready()

        expected = (
            self.prv_columns
            + [self.demand_column]
        )

        if (
            result.feature_names
            != expected
        ):
            raise ValueError(
                "Saved surrogate metadata is inconsistent "
                "with its trained feature ordering."
            )

    def _require_ready(
        self,
    ) -> DownstreamRegressionResult:
        """Return trained result or raise a clear error."""

        if self._core.result is None:
            raise RuntimeError(
                "Downstream-pressure model has not been "
                "trained or loaded."
            )

        return self._core.result
