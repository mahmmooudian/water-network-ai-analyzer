Visualization utilities for Water Network AI Analyzer.

This module contains reusable matplotlib figures for:
- Actual vs Predicted regression diagnostics
- Feature importance
- PSO convergence
- Optimized PRV settings
- Optimized pressure profiles
- Hydraulic pressure time series
- Hydraulic pressure summaries

The functions return matplotlib Figure objects and do not call ``plt.show()``.
This allows the same plots to be embedded inside Tkinter or saved to files.
"""

from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..models.critical_pressure import RegressionResult
    from ..optimization.pso import (
        PSOOptimizationResult,
        PSOPeriodResult,
    )
    from ..hydraulics.wntr_runner import (
        HydraulicSimulationResult,
    )


# ======================================================================
# Helpers
# ======================================================================


def _new_figure(
    *,
    width: float = 8.0,
    height: float = 5.0,
) -> tuple[Figure, object]:
    """
    Create a standalone figure with a single axes object.
    """

    fig = plt.figure(
        figsize=(width, height),
        constrained_layout=True,
    )

    ax = fig.add_axes(
        [0.10, 0.12, 0.86, 0.82]
    )

    return fig, ax


def _ensure_2d(
    values,
    *,
    n_targets: int,
) -> np.ndarray:
    """
    Normalize prediction/target arrays to ``(rows, targets)``.
    """

    array = np.asarray(
        values,
        dtype=float,
    )

    if n_targets == 1:
        return array.reshape(
            -1,
            1,
        )

    if array.ndim != 2:
        raise ValueError(
            "Multi-output values must be two-dimensional."
        )

    if array.shape[1] != n_targets:
        raise ValueError(
            "Target array width does not match target_names."
        )

    return array


def _validate_result_arrays(
    result,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract and validate hold-out arrays from a regression result.
    """

    target_names = list(
        result.target_names
    )

    if not target_names:
        raise ValueError(
            "Regression result has no target names."
        )

    y_test = _ensure_2d(
        result.y_test,
        n_targets=len(target_names),
    )

    y_pred = _ensure_2d(
        result.y_pred,
        n_targets=len(target_names),
    )

    if y_test.shape != y_pred.shape:
        raise ValueError(
            "y_test and y_pred shapes do not match."
        )

    if y_test.shape[0] == 0:
        raise ValueError(
            "Regression result contains no hold-out predictions."
        )

    return (
        y_test,
        y_pred,
        target_names,
    )


# ======================================================================
# Regression diagnostics
# ======================================================================


def plot_actual_vs_predicted(
    result: "RegressionResult",
    *,
    target: Optional[str] = None,
    max_points: int = 1500,
) -> Figure:
    """
    Plot actual vs predicted values for one regression target.

    Parameters
    ----------
    result:
        Trained regression result.
    target:
        Optional target name. Defaults to the first target.
    max_points:
        Maximum number of displayed hold-out points.
    """

    y_true, y_pred, target_names = (
        _validate_result_arrays(
            result
        )
    )

    if target is None:
        target = target_names[0]

    if target not in target_names:
        raise ValueError(
            f"Unknown target '{target}'. "
            f"Available targets: {target_names}"
        )

    if max_points < 1:
        raise ValueError(
            "max_points must be at least 1."
        )

    target_index = target_names.index(
        target
    )

    actual = y_true[
        :,
        target_index,
    ]

    predicted = y_pred[
        :,
        target_index,
    ]

    if len(actual) > max_points:
        indices = np.linspace(
            0,
            len(actual) - 1,
            max_points,
            dtype=int,
        )

        actual = actual[
            indices
        ]

        predicted = predicted[
            indices
        ]

    finite = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[
        finite
    ]

    predicted = predicted[
        finite
    ]

    if actual.size == 0:
        raise ValueError(
            "No finite actual/predicted values are available."
        )

    fig, ax = _new_figure()

    ax.scatter(
        actual,
        predicted,
        alpha=0.70,
    )

    minimum = float(
        min(
            np.min(actual),
            np.min(predicted),
        )
    )

    maximum = float(
        max(
            np.max(actual),
            np.max(predicted),
        )
    )

    if maximum == minimum:
        maximum = minimum + 1.0

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Ideal prediction",
    )

    ax.set_xlabel(
        "Actual Pressure"
    )

    ax.set_ylabel(
        "Predicted Pressure"
    )

    ax.set_title(
        f"Actual vs Predicted — {target}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    return fig


def plot_prediction_series(
    result: "RegressionResult",
    *,
    target: Optional[str] = None,
    max_points: int = 500,
) -> Figure:
    """
    Plot hold-out actual and predicted values across observation order.
    """

    y_true, y_pred, target_names = (
        _validate_result_arrays(
            result
        )
    )

    if target is None:
        target = target_names[0]

    if target not in target_names:
        raise ValueError(
            f"Unknown target '{target}'."
        )

    target_index = target_names.index(
        target
    )

    actual = y_true[
        :,
        target_index,
    ]

    predicted = y_pred[
        :,
        target_index,
    ]

    count = min(
        len(actual),
        int(max_points),
    )

    if count < 1:
        raise ValueError(
            "No points are available for plotting."
        )

    x = np.arange(
        count
    )

    fig, ax = _new_figure(
        width=9.0,
        height=5.0,
    )

    ax.plot(
        x,
        actual[:count],
        label="Actual",
    )

    ax.plot(
        x,
        predicted[:count],
        label="Predicted",
    )

    ax.set_xlabel(
        "Hold-out Observation"
    )

    ax.set_ylabel(
        "Pressure"
    )

    ax.set_title(
        f"Prediction Series — {target}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    return fig


def plot_feature_importance(
    result: "RegressionResult",
    *,
    top_n: int = 15,
) -> Figure:
    """
    Plot the most important model features.
    """

    importance = (
        result.feature_importance
        .copy()
    )

    if importance.empty:
        raise ValueError(
            "Feature importance is unavailable."
        )

    required = {
        "Feature",
        "Importance",
    }

    if not required.issubset(
        importance.columns
    ):
        raise ValueError(
            "Feature-importance table must contain "
            "'Feature' and 'Importance' columns."
        )

    importance[
        "Importance"
    ] = pd.to_numeric(
        importance[
            "Importance"
        ],
        errors="coerce",
    )

    importance = (
        importance
        .dropna(
            subset=["Importance"]
        )
        .sort_values(
            "Importance",
            ascending=False,
        )
    )

    if importance.empty:
        raise ValueError(
            "No numeric feature-importance values are available."
        )

    top_n = max(
        1,
        int(top_n),
    )

    subset = (
        importance
        .head(top_n)
        .sort_values(
            "Importance",
            ascending=True,
        )
    )

    fig, ax = _new_figure(
        width=8.5,
        height=max(
            4.5,
            0.38 * len(subset) + 2.0,
        ),
    )

    ax.barh(
        subset[
            "Feature"
        ],
        subset[
            "Importance"
        ],
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Feature Importance"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    return fig


# ======================================================================
# PSO visualization
# ======================================================================


def plot_pso_convergence(
    period_result: "PSOPeriodResult",
) -> Figure:
    """
    Plot best objective score across PSO iterations.
    """

    convergence = np.asarray(
        period_result.convergence,
        dtype=float,
    )

    if convergence.size == 0:
        raise ValueError(
            "PSO convergence history is empty."
        )

    fig, ax = _new_figure()

    ax.plot(
        np.arange(
            len(convergence)
        ),
        convergence,
    )

    ax.set_xlabel(
        "Iteration"
    )

    ax.set_ylabel(
        "Best Objective"
    )

    ax.set_title(
        f"PSO Convergence — Period {period_result.period}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    return fig


def plot_prv_settings(
    result: "PSOOptimizationResult",
) -> Figure:
    """
    Plot optimized PRV settings across all periods.
    """

    if not result.periods:
        raise ValueError(
            "Optimization result has no periods."
        )

    frame = result.to_frame()

    prv_columns = [
        column
        for column in frame.columns
        if column.startswith(
            "PRV::"
        )
    ]

    if not prv_columns:
        raise ValueError(
            "Optimization result contains no PRV-setting columns."
        )

    fig, ax = _new_figure(
        width=9.0,
        height=5.2,
    )

    x = frame[
        "Period"
    ].to_numpy()

    for column in prv_columns:
        ax.plot(
            x,
            frame[column],
            marker="o",
            label=column.replace(
                "PRV::",
                "",
            ),
        )

    ax.set_xlabel(
        "Period"
    )

    ax.set_ylabel(
        "Optimized PRV Setting"
    )

    ax.set_title(
        "Optimized PRV Settings"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    return fig


def plot_optimized_pressure_summary(
    result: "PSOOptimizationResult",
) -> Figure:
    """
    Plot minimum, mean and maximum optimized pressure by period.
    """

    if not result.periods:
        raise ValueError(
            "Optimization result has no periods."
        )

    frame = result.to_frame()

    fig, ax = _new_figure(
        width=9.0,
        height=5.2,
    )

    x = frame[
        "Period"
    ].to_numpy()

    ax.plot(
        x,
        frame[
            "MinPressure"
        ],
        marker="o",
        label="Minimum",
    )

    ax.plot(
        x,
        frame[
            "MeanPressure"
        ],
        marker="o",
        label="Mean",
    )

    ax.plot(
        x,
        frame[
            "MaxPressure"
        ],
        marker="o",
        label="Maximum",
    )

    ax.set_xlabel(
        "Period"
    )

    ax.set_ylabel(
        "Predicted Pressure"
    )

    ax.set_title(
        "Optimized Pressure Summary"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    return fig


def plot_objective_components(
    result: "PSOOptimizationResult",
) -> Figure:
    """
    Plot weighted PSO objective components across periods.
    """

    if not result.periods:
        raise ValueError(
            "Optimization result has no periods."
        )

    frame = result.to_frame()

    components = [
        "PressureViolationPenalty",
        "PressureTargetPenalty",
        "StabilityPenalty",
        "ReferencePenalty",
    ]

    fig, ax = _new_figure(
        width=9.0,
        height=5.2,
    )

    x = frame[
        "Period"
    ].to_numpy()

    for column in components:
        ax.plot(
            x,
            frame[column],
            marker="o",
            label=column.replace(
                "Penalty",
                "",
            ),
        )

    ax.set_xlabel(
        "Period"
    )

    ax.set_ylabel(
        "Weighted Objective Component"
    )

    ax.set_title(
        "PSO Objective Components"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    return fig


# ======================================================================
# Hydraulic visualization
# ======================================================================


def plot_hydraulic_pressure(
    result: "HydraulicSimulationResult",
    *,
    node_names: Optional[
        Sequence[str]
    ] = None,
    max_nodes: int = 10,
) -> Figure:
    """
    Plot hydraulic pressure time series for selected nodes.
    """

    pressure = (
        result.pressure.copy()
    )

    if pressure.empty:
        raise ValueError(
            "Hydraulic pressure result is empty."
        )

    if node_names is None:
        selected = list(
            pressure.columns[
                :max(
                    1,
                    int(max_nodes),
                )
            ]
        )

    else:
        selected = [
            str(name)
            for name in node_names
        ]

        missing = [
            name
            for name in selected
            if name not in pressure.columns
        ]

        if missing:
            raise ValueError(
                "Unknown pressure-result nodes: "
                + ", ".join(missing)
            )

    if not selected:
        raise ValueError(
            "No nodes selected for hydraulic-pressure plotting."
        )

    fig, ax = _new_figure(
        width=9.0,
        height=5.2,
    )

    time_hours = (
        pressure.index.to_numpy(
            dtype=float
        )
        / 3600.0
    )

    for node in selected:
        ax.plot(
            time_hours,
            pressure[
                node
            ].to_numpy(
                dtype=float
            ),
            label=node,
        )

    ax.set_xlabel(
        "Simulation Time (hours)"
    )

    ax.set_ylabel(
        "Pressure"
    )

    ax.set_title(
        "EPANET / WNTR Hydraulic Pressure"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    return fig


def plot_hydraulic_pressure_summary(
    result: "HydraulicSimulationResult",
    *,
    top_n: int = 15,
) -> Figure:
    """
    Plot mean hydraulic pressure by node.
    """

    summary = (
        result.pressure_summary()
        .copy()
    )

    if summary.empty:
        raise ValueError(
            "Hydraulic pressure summary is empty."
        )

    summary[
        "MeanPressure"
    ] = pd.to_numeric(
        summary[
            "MeanPressure"
        ],
        errors="coerce",
    )

    summary = (
        summary
        .dropna(
            subset=[
                "MeanPressure"
            ]
        )
        .sort_values(
            "MeanPressure",
            ascending=False,
        )
        .head(
            max(
                1,
                int(top_n),
            )
        )
        .sort_values(
            "MeanPressure",
            ascending=True,
        )
    )

    fig, ax = _new_figure(
        width=8.5,
        height=max(
            4.5,
            0.38 * len(summary) + 2.0,
        ),
    )

    ax.barh(
        summary[
            "Node"
        ],
        summary[
            "MeanPressure"
        ],
    )

    ax.set_xlabel(
        "Mean Pressure"
    )

    ax.set_ylabel(
        "Node"
    )

    ax.set_title(
        "Hydraulic Mean Pressure by Node"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.25,
    )

    return fig


# ======================================================================
# Figure persistence
# ======================================================================


def save_figure(
    figure: Figure,
    path: str,
    *,
    dpi: int = 150,
) -> str:
    """
    Save a matplotlib Figure and return the output path.
    """

    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib Figure."
        )

    output = str(
        Path(path)
    )

    Path(output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output,
        dpi=int(dpi),
        bbox_inches="tight",
    )

    return output


def close_figure(
    figure: Figure,
) -> None:
    """
    Explicitly release a matplotlib figure.
    """

    plt.close(
        figure
    )
