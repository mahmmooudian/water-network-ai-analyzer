"""
Leakage-safe preprocessing utilities for Water Network AI Analyzer.

This module contains preprocessing components used inside scikit-learn
pipelines so that all learned preprocessing statistics are fitted only on
training data or the current cross-validation fold.

Pipeline:

    Raw Features
        ↓
    KNN Imputation
        ↓
    IQR Outlier Clipping
        ↓
    Machine Learning Model

Feature scaling is intentionally omitted because XGBoost is tree-based.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from ..config import AppConfig, DEFAULT_CONFIG


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clip extreme numerical feature values using the Interquartile Range.

    Bounds are learned only during ``fit()``. When this transformer is used
    inside an sklearn Pipeline, each training fold receives its own bounds,
    preventing information leakage from validation or test data.

    Parameters
    ----------
    factor:
        IQR multiplier.

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
    """

    def __init__(self, factor: float = 1.5) -> None:
        self.factor = factor

    def fit(self, X, y=None):
        """
        Learn clipping bounds from training data.
        """

        if self.factor <= 0:
            raise ValueError(
                "IQR factor must be greater than zero."
            )

        array = self._to_2d_float_array(X)

        if array.shape[0] == 0:
            raise ValueError(
                "Cannot fit IQRClipper on an empty dataset."
            )

        if array.shape[1] == 0:
            raise ValueError(
                "Cannot fit IQRClipper without features."
            )

        # KNNImputer normally runs before this transformer,
        # but nan-aware quantiles make the class robust when
        # it is used independently.
        self.q1_ = np.nanpercentile(
            array,
            25,
            axis=0,
        )

        self.q3_ = np.nanpercentile(
            array,
            75,
            axis=0,
        )

        self.iqr_ = self.q3_ - self.q1_

        self.lower_bounds_ = (
            self.q1_
            - self.factor * self.iqr_
        )

        self.upper_bounds_ = (
            self.q3_
            + self.factor * self.iqr_
        )

        # Constant features have IQR = 0.
        # Their lower and upper bounds therefore equal
        # the constant value, which is acceptable.
        self.n_features_in_ = array.shape[1]

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(
                X.columns,
                dtype=object,
            )

        return self

    def transform(self, X):
        """
        Apply learned clipping bounds.
        """

        self._check_is_fitted()

        array = self._to_2d_float_array(X)

        if array.shape[1] != self.n_features_in_:
            raise ValueError(
                "Feature count does not match the data used "
                "to fit IQRClipper."
            )

        clipped = np.clip(
            array,
            self.lower_bounds_,
            self.upper_bounds_,
        )

        return clipped

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        """
        Preserve feature names for sklearn-compatible inspection.
        """

        self._check_is_fitted()

        if input_features is not None:
            return np.asarray(
                input_features,
                dtype=object,
            )

        if hasattr(
            self,
            "feature_names_in_",
        ):
            return self.feature_names_in_

        return np.asarray(
            [
                f"feature_{index}"
                for index in range(
                    self.n_features_in_
                )
            ],
            dtype=object,
        )

    def _check_is_fitted(self) -> None:
        """
        Ensure bounds have been learned before transformation.
        """

        required = (
            "lower_bounds_",
            "upper_bounds_",
            "n_features_in_",
        )

        if not all(
            hasattr(self, attr)
            for attr in required
        ):
            raise RuntimeError(
                "IQRClipper has not been fitted yet."
            )

    @staticmethod
    def _to_2d_float_array(X) -> np.ndarray:
        """
        Convert pandas/numpy input into a 2D floating-point array.
        """

        if isinstance(
            X,
            pd.DataFrame,
        ):
            array = X.to_numpy(
                dtype=float,
                copy=True,
            )

        elif isinstance(
            X,
            pd.Series,
        ):
            array = X.to_numpy(
                dtype=float,
                copy=True,
            ).reshape(-1, 1)

        else:
            array = np.asarray(
                X,
                dtype=float,
            )

            if array.ndim == 1:
                array = array.reshape(
                    -1,
                    1,
                )

            else:
                array = array.copy()

        if array.ndim != 2:
            raise ValueError(
                "Preprocessing input must be two-dimensional."
            )

        return array


def build_feature_preprocessor(
    config: Optional[AppConfig] = None,
) -> Pipeline:
    """
    Create the leakage-safe feature preprocessing pipeline.

    Important
    ---------
    The returned pipeline must be fitted only after train/test splitting
    or as part of a complete sklearn Pipeline used by cross-validation.

    Parameters
    ----------
    config:
        Optional application configuration.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline containing KNN imputation and IQR clipping.
    """

    config = (
        config
        if config is not None
        else DEFAULT_CONFIG
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                KNNImputer(
                    n_neighbors=config.knn_neighbors,
                    weights="distance",
                ),
            ),
            (
                "outlier_clipper",
                IQRClipper(
                    factor=config.iqr_factor
                ),
            ),
        ]
    )


def build_model_pipeline(
    estimator,
    config: Optional[AppConfig] = None,
) -> Pipeline:
    """
    Create the complete leakage-safe ML pipeline.

    The estimator is deliberately supplied by the model module so this
    preprocessing module does not depend on XGBoost directly.

    Structure
    ---------

        KNNImputer
            ↓
        IQRClipper
            ↓
        Estimator

    Because all components are inside one sklearn Pipeline,
    RandomizedSearchCV and cross-validation fit preprocessing only on
    each training fold.
    """

    if estimator is None:
        raise ValueError(
            "An estimator must be provided."
        )

    config = (
        config
        if config is not None
        else DEFAULT_CONFIG
    )

    preprocessing = build_feature_preprocessor(
        config
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                preprocessing.named_steps[
                    "imputer"
                ],
            ),
            (
                "outlier_clipper",
                preprocessing.named_steps[
                    "outlier_clipper"
                ],
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def coerce_numeric_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert model-facing dataframe columns to numeric values.

    Invalid strings become NaN and are subsequently handled by the
    KNNImputer inside the training pipeline.

    The original dataframe is never modified.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    converted = df.copy()

    for column in converted.columns:
        converted[column] = pd.to_numeric(
            converted[column],
            errors="coerce",
        )

    return converted


def remove_missing_target_rows(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
):
    """
    Remove observations where target values are missing.

    Targets must never be imputed because fabricating target values would
    compromise model training and evaluation.

    Missing values in X are preserved for KNNImputer.
    """

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of rows."
        )

    if isinstance(
        y,
        pd.Series,
    ):
        valid_mask = y.notna()

    elif isinstance(
        y,
        pd.DataFrame,
    ):
        valid_mask = y.notna().all(
            axis=1
        )

    else:
        raise TypeError(
            "y must be a pandas Series or DataFrame."
        )

    X_clean = X.loc[
        valid_mask
    ].copy()

    y_clean = y.loc[
        valid_mask
    ].copy()

    return (
        X_clean.reset_index(drop=True),
        y_clean.reset_index(drop=True),
    )
