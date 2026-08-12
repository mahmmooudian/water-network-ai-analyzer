"""
Particle Swarm Optimization for Water Network AI Analyzer.

The optimizer is intentionally data-driven:

    Candidate PRV Settings
            +
          Demand
            |
            v
    DownstreamPressureModel
            |
            v
    Predicted Downstream Pressure
            +
          Demand
            |
            v
    CriticalPressureModel
            |
            v
    Predicted Critical Pressure
            |
            v
     Engineering Objective
            |
            v
           PSO

Important
---------
This module does NOT claim to be a hydraulic solver. It optimizes against
trained surrogate models. Physics-based WNTR / EPANET simulation remains a
separate validation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import AppConfig, DEFAULT_CONFIG
from ..models.critical_pressure import CriticalPressureModel
from ..models.downstream_pressure import DownstreamPressureModel


# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class PRVBounds:
    """Lower and upper operating bounds for each PRV."""

    names: List[str]
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)

        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("PRV bounds must be one-dimensional.")

        if len(self.names) != len(lower) or len(lower) != len(upper):
            raise ValueError(
                "PRV names, lower bounds and upper bounds must have "
                "the same length."
            )

        if len(self.names) == 0:
            raise ValueError("At least one PRV bound is required.")

        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("PRV bounds must contain finite values.")

        if np.any(lower >= upper):
            raise ValueError(
                "Every PRV lower bound must be smaller than its upper bound."
            )

        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def width(self) -> np.ndarray:
        """Operating-range width for every PRV."""

        return self.upper - self.lower

    @property
    def n_prvs(self) -> int:
        return len(self.names)

    def clip(self, values: np.ndarray) -> np.ndarray:
        """Clip candidate settings to valid PRV operating limits."""

        return np.clip(
            np.asarray(values, dtype=float),
            self.lower,
            self.upper,
        )


@dataclass
class ObjectiveBreakdown:
    """Objective components for one selected operating point."""

    total: float
    pressure_violation: float
    pressure_target: float
    stability: float
    reference: float


@dataclass
class PSOPeriodResult:
    """Optimization result for a single demand period."""

    period: int
    demand: float

    prv_names: List[str]
    prv_settings: np.ndarray

    downstream_names: List[str]
    downstream_pressures: np.ndarray

    critical_names: List[str]
    critical_pressures: np.ndarray

    objective: ObjectiveBreakdown
    convergence: List[float] = field(default_factory=list)

    reference_settings: Optional[np.ndarray] = None
    previous_settings: Optional[np.ndarray] = None

    @property
    def all_pressures(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.downstream_pressures, dtype=float),
                np.asarray(self.critical_pressures, dtype=float),
            ]
        )

    @property
    def min_pressure(self) -> float:
        values = self.all_pressures
        return float(np.min(values)) if values.size else float("nan")

    @property
    def mean_pressure(self) -> float:
        values = self.all_pressures
        return float(np.mean(values)) if values.size else float("nan")

    @property
    def max_pressure(self) -> float:
        values = self.all_pressures
        return float(np.max(values)) if values.size else float("nan")


@dataclass
class PSOOptimizationResult:
    """Sequential optimization output."""

    periods: List[PSOPeriodResult]
    bounds: PRVBounds

    def to_frame(self) -> pd.DataFrame:
        """Convert the complete optimization result to a flat DataFrame."""

        rows: List[Dict[str, float]] = []

        for item in self.periods:
            row: Dict[str, float] = {
                "Period": int(item.period),
                "Demand": float(item.demand),
                "Objective": float(item.objective.total),
                "PressureViolationPenalty": float(
                    item.objective.pressure_violation
                ),
                "PressureTargetPenalty": float(
                    item.objective.pressure_target
                ),
                "StabilityPenalty": float(
                    item.objective.stability
                ),
                "ReferencePenalty": float(
                    item.objective.reference
                ),
                "MinPressure": item.min_pressure,
                "MeanPressure": item.mean_pressure,
                "MaxPressure": item.max_pressure,
            }

            for name, value in zip(
                item.prv_names,
                item.prv_settings,
            ):
                row[f"PRV::{name}"] = float(value)

            for name, value in zip(
                item.downstream_names,
                item.downstream_pressures,
            ):
                row[f"Downstream::{name}"] = float(value)

            for name, value in zip(
                item.critical_names,
                item.critical_pressures,
            ):
                row[f"Critical::{name}"] = float(value)

            rows.append(row)

        return pd.DataFrame(rows)


# ======================================================================
# Bounds
# ======================================================================


def derive_prv_bounds(
    df: pd.DataFrame,
    prv_columns: Sequence[str],
    config: Optional[AppConfig] = None,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    margin_fraction: float = 0.10,
) -> PRVBounds:
    """
    Derive PRV operating bounds from historical observations.

    The quantile interval is expanded slightly, then clipped to global
    engineering safety bounds from ``AppConfig``.

    Constant or near-constant PRV histories are widened automatically so
    PSO still has a valid search interval.
    """

    config = config or DEFAULT_CONFIG

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    prv_columns = [str(column) for column in prv_columns]

    if not prv_columns:
        raise ValueError("At least one PRV column is required.")

    missing = [
        column
        for column in prv_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing PRV columns: "
            + ", ".join(missing)
        )

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError(
            "Quantiles must satisfy 0 <= lower < upper <= 1."
        )

    if margin_fraction < 0:
        raise ValueError(
            "margin_fraction cannot be negative."
        )

    lower_values: List[float] = []
    upper_values: List[float] = []

    global_width = float(
        config.max_prv - config.min_prv
    )

    # Minimum useful search width for nearly constant historical data.
    minimum_width = max(
        1.0,
        0.05 * global_width,
    )

    for column in prv_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if series.empty:
            raise ValueError(
                f"PRV column '{column}' contains no numeric observations."
            )

        q_low = float(
            series.quantile(lower_quantile)
        )

        q_high = float(
            series.quantile(upper_quantile)
        )

        observed_width = max(
            q_high - q_low,
            0.0,
        )

        if observed_width < minimum_width:
            center = float(
                series.median()
            )
            half_width = minimum_width / 2.0

            low = center - half_width
            high = center + half_width

        else:
            margin = (
                observed_width
                * margin_fraction
            )

            low = q_low - margin
            high = q_high + margin

        low = max(
            float(config.min_prv),
            float(low),
        )

        high = min(
            float(config.max_prv),
            float(high),
        )

        # If clipping against global limits collapses the interval,
        # reconstruct a small valid interval around the historical median.
        if high <= low:
            center = float(
                np.clip(
                    series.median(),
                    config.min_prv,
                    config.max_prv,
                )
            )

            half_width = minimum_width / 2.0

            low = max(
                float(config.min_prv),
                center - half_width,
            )

            high = min(
                float(config.max_prv),
                center + half_width,
            )

        if high <= low:
            raise ValueError(
                f"Could not construct a valid operating range for '{column}'."
            )

        lower_values.append(
            float(low)
        )

        upper_values.append(
            float(high)
        )

    return PRVBounds(
        names=list(prv_columns),
        lower=np.asarray(
            lower_values,
            dtype=float,
        ),
        upper=np.asarray(
            upper_values,
            dtype=float,
        ),
    )


# ======================================================================
# Optimizer
# ======================================================================


class PSOOptimizer:
    """
    Particle Swarm Optimization over trained pressure surrogate models.

    The optimizer evaluates all particles vectorially through:

        PRV + Demand
            -> downstream surrogate
            -> critical-pressure model
            -> engineering objective
    """

    def __init__(
        self,
        downstream_model: DownstreamPressureModel,
        critical_model: CriticalPressureModel,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.downstream_model = downstream_model
        self.critical_model = critical_model
        self.config = config or DEFAULT_CONFIG

        self._validate_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_period(
        self,
        *,
        demand: float,
        bounds: PRVBounds,
        period: int = 0,
        reference_settings: Optional[Sequence[float]] = None,
        previous_settings: Optional[Sequence[float]] = None,
    ) -> PSOPeriodResult:
        """
        Optimize PRV settings for one operating period.
        """

        self._validate_bounds_against_model(
            bounds
        )

        demand = float(demand)

        if not np.isfinite(demand):
            raise ValueError(
                "Demand must be a finite numeric value."
            )

        reference = self._optional_setting_vector(
            reference_settings,
            bounds,
            label="reference_settings",
        )

        previous = self._optional_setting_vector(
            previous_settings,
            bounds,
            label="previous_settings",
        )

        rng = np.random.default_rng(
            self.config.random_state
            + int(period)
        )

        positions = rng.uniform(
            low=bounds.lower,
            high=bounds.upper,
            size=(
                self.config.pso_particles,
                bounds.n_prvs,
            ),
        )

        # Seed plausible operating points into the population.
        seed_index = 0

        if reference is not None:
            positions[seed_index] = reference
            seed_index += 1

        if (
            previous is not None
            and seed_index < len(positions)
        ):
            positions[seed_index] = previous
            seed_index += 1

        # Historical midpoint gives one deterministic baseline candidate.
        if seed_index < len(positions):
            positions[seed_index] = (
                bounds.lower
                + bounds.upper
            ) / 2.0

        velocity_limit = (
            bounds.width
            * self.config.pso_velocity_fraction
        )

        velocities = rng.uniform(
            low=-velocity_limit,
            high=velocity_limit,
            size=positions.shape,
        )

        scores, _, _ = self._evaluate_population(
            particles=positions,
            demand=demand,
            bounds=bounds,
            reference_settings=reference,
            previous_settings=previous,
        )

        personal_best_positions = (
            positions.copy()
        )

        personal_best_scores = (
            scores.copy()
        )

        global_index = int(
            np.argmin(
                personal_best_scores
            )
        )

        global_best_position = (
            personal_best_positions[
                global_index
            ].copy()
        )

        global_best_score = float(
            personal_best_scores[
                global_index
            ]
        )

        convergence: List[float] = [
            global_best_score
        ]

        iterations = int(
            self.config.pso_iterations
        )

        for iteration in range(
            iterations
        ):
            if iterations <= 1:
                progress = 1.0
            else:
                progress = (
                    iteration
                    / (iterations - 1)
                )

            inertia = (
                self.config.pso_w_max
                - progress
                * (
                    self.config.pso_w_max
                    - self.config.pso_w_min
                )
            )

            r1 = rng.random(
                size=positions.shape
            )

            r2 = rng.random(
                size=positions.shape
            )

            cognitive = (
                self.config.pso_c1
                * r1
                * (
                    personal_best_positions
                    - positions
                )
            )

            social = (
                self.config.pso_c2
                * r2
                * (
                    global_best_position
                    - positions
                )
            )

            velocities = (
                inertia * velocities
                + cognitive
                + social
            )

            velocities = np.clip(
                velocities,
                -velocity_limit,
                velocity_limit,
            )

            positions = (
                positions
                + velocities
            )

            positions = bounds.clip(
                positions
            )

            scores, _, _ = self._evaluate_population(
                particles=positions,
                demand=demand,
                bounds=bounds,
                reference_settings=reference,
                previous_settings=previous,
            )

            improved = (
                scores
                < personal_best_scores
            )

            if np.any(improved):
                personal_best_scores[
                    improved
                ] = scores[
                    improved
                ]

                personal_best_positions[
                    improved
                ] = positions[
                    improved
                ]

            candidate_index = int(
                np.argmin(
                    personal_best_scores
                )
            )

            candidate_score = float(
                personal_best_scores[
                    candidate_index
                ]
            )

            if (
                candidate_score
                < global_best_score
            ):
                global_best_score = (
                    candidate_score
                )

                global_best_position = (
                    personal_best_positions[
                        candidate_index
                    ].copy()
                )

            convergence.append(
                global_best_score
            )

        # Re-evaluate the selected solution to obtain its pressure vectors
        # and objective components.
        final_scores, downstream, critical = self._evaluate_population(
            particles=global_best_position.reshape(
                1,
                -1,
            ),
            demand=demand,
            bounds=bounds,
            reference_settings=reference,
            previous_settings=previous,
        )

        objective = self._objective_breakdown_for_one(
            particle=global_best_position,
            downstream=downstream[0],
            critical=critical[0],
            bounds=bounds,
            reference_settings=reference,
            previous_settings=previous,
        )

        # Numerical sanity: total should equal the vectorized objective.
        objective.total = float(
            final_scores[0]
        )

        downstream_result = (
            self.downstream_model.result
        )

        critical_result = (
            self.critical_model.result
        )

        assert downstream_result is not None
        assert critical_result is not None

        return PSOPeriodResult(
            period=int(period),
            demand=demand,
            prv_names=list(
                bounds.names
            ),
            prv_settings=global_best_position,
            downstream_names=list(
                downstream_result.target_names
            ),
            downstream_pressures=downstream[0],
            critical_names=list(
                critical_result.target_names
            ),
            critical_pressures=critical[0],
            objective=objective,
            convergence=convergence,
            reference_settings=(
                reference.copy()
                if reference is not None
                else None
            ),
            previous_settings=(
                previous.copy()
                if previous is not None
                else None
            ),
        )

    def optimize_sequence(
        self,
        *,
        demands: Sequence[float],
        bounds: PRVBounds,
        reference_settings: Optional[np.ndarray] = None,
        starting_settings: Optional[Sequence[float]] = None,
    ) -> PSOOptimizationResult:
        """
        Optimize consecutive operating periods.

        The optimized solution from period ``t`` becomes the stability
        reference for period ``t+1``.
        """

        demand_values = np.asarray(
            demands,
            dtype=float,
        ).reshape(-1)

        if demand_values.size == 0:
            raise ValueError(
                "At least one demand value is required."
            )

        if not np.all(
            np.isfinite(demand_values)
        ):
            raise ValueError(
                "All demand values must be finite."
            )

        references: Optional[np.ndarray]

        if reference_settings is None:
            references = None

        else:
            references = np.asarray(
                reference_settings,
                dtype=float,
            )

            if references.ndim != 2:
                raise ValueError(
                    "reference_settings must be a 2D array."
                )

            expected_shape = (
                len(demand_values),
                bounds.n_prvs,
            )

            if (
                references.shape
                != expected_shape
            ):
                raise ValueError(
                    "reference_settings shape must be "
                    f"{expected_shape}, received {references.shape}."
                )

            if not np.all(
                np.isfinite(references)
            ):
                raise ValueError(
                    "reference_settings contains non-finite values."
                )

            references = bounds.clip(
                references
            )

        previous = self._optional_setting_vector(
            starting_settings,
            bounds,
            label="starting_settings",
        )

        period_results: List[
            PSOPeriodResult
        ] = []

        for period, demand in enumerate(
            demand_values
        ):
            reference = (
                references[period]
                if references is not None
                else None
            )

            result = self.optimize_period(
                demand=float(demand),
                bounds=bounds,
                period=period,
                reference_settings=reference,
                previous_settings=previous,
            )

            period_results.append(
                result
            )

            previous = (
                result.prv_settings.copy()
            )

        return PSOOptimizationResult(
            periods=period_results,
            bounds=bounds,
        )

    def optimize_dataframe(
        self,
        df: pd.DataFrame,
        *,
        periods: Optional[int] = None,
    ) -> PSOOptimizationResult:
        """
        Optimize directly from a historical dataframe.

        Historical PRV settings are used as per-period reference settings.
        The demand column and PRV order come directly from the trained
        downstream surrogate metadata.
        """

        self._validate_models()

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot optimize an empty dataframe."
            )

        demand_column = (
            self.downstream_model
            .demand_column
        )

        prv_columns = list(
            self.downstream_model
            .prv_columns
        )

        if not demand_column:
            raise RuntimeError(
                "Downstream model has no demand-column metadata."
            )

        required = (
            prv_columns
            + [demand_column]
        )

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Optimization dataframe is missing columns: "
                + ", ".join(missing)
            )

        working = df.loc[
            :,
            required,
        ].copy()

        for column in required:
            working[column] = pd.to_numeric(
                working[column],
                errors="coerce",
            )

        # Demand is essential for every period.
        working = working.dropna(
            subset=[demand_column]
        ).reset_index(drop=True)

        if working.empty:
            raise ValueError(
                "No valid demand observations remain for optimization."
            )

        if periods is None:
            periods = min(
                self.config.optimization_hours,
                len(working),
            )

        periods = int(periods)

        if periods < 1:
            raise ValueError(
                "periods must be at least 1."
            )

        periods = min(
            periods,
            len(working),
        )

        working = working.iloc[
            :periods
        ].copy()

        bounds = derive_prv_bounds(
            df=df,
            prv_columns=prv_columns,
            config=self.config,
        )

        # Historical PRVs can contain missing values. Fill them with robust
        # medians only for the reference penalty; this does NOT train a model.
        references = (
            working.loc[
                :,
                prv_columns,
            ]
            .copy()
        )

        for column in prv_columns:
            median = pd.to_numeric(
                df[column],
                errors="coerce",
            ).median()

            if pd.isna(median):
                raise ValueError(
                    f"Cannot build a historical reference for '{column}'."
                )

            references[column] = (
                references[column]
                .fillna(float(median))
            )

        reference_array = bounds.clip(
            references.to_numpy(
                dtype=float
            )
        )

        demands = working[
            demand_column
        ].to_numpy(
            dtype=float
        )

        return self.optimize_sequence(
            demands=demands,
            bounds=bounds,
            reference_settings=reference_array,
            starting_settings=reference_array[0],
        )

    # ------------------------------------------------------------------
    # Population evaluation
    # ------------------------------------------------------------------

    def _evaluate_population(
        self,
        *,
        particles: np.ndarray,
        demand: float,
        bounds: PRVBounds,
        reference_settings: Optional[np.ndarray],
        previous_settings: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate all candidate PRV configurations in one vectorized call.
        """

        particles = np.asarray(
            particles,
            dtype=float,
        )

        if particles.ndim != 2:
            raise ValueError(
                "particles must be a 2D array."
            )

        if (
            particles.shape[1]
            != bounds.n_prvs
        ):
            raise ValueError(
                "Particle dimension does not match PRV bounds."
            )

        particles = bounds.clip(
            particles
        )

        downstream = (
            self.downstream_model
            .predict_particles(
                particles,
                demand,
            )
        )

        critical = (
            self._predict_critical_population(
                downstream=downstream,
                demand=demand,
            )
        )

        scores = self._objective_vector(
            particles=particles,
            downstream=downstream,
            critical=critical,
            bounds=bounds,
            reference_settings=reference_settings,
            previous_settings=previous_settings,
        )

        # Invalid surrogate outputs must never become attractive particles.
        invalid = (
            ~np.isfinite(downstream).all(axis=1)
            | ~np.isfinite(critical).all(axis=1)
            | ~np.isfinite(scores)
        )

        if np.any(invalid):
            scores = scores.copy()
            scores[invalid] = np.inf

        return (
            scores,
            downstream,
            critical,
        )

    def _predict_critical_population(
        self,
        *,
        downstream: np.ndarray,
        demand: float,
    ) -> np.ndarray:
        """
        Convert downstream predictions into critical-model input features.
        """

        downstream_result = (
            self.downstream_model.result
        )

        critical_result = (
            self.critical_model.result
        )

        if downstream_result is None:
            raise RuntimeError(
                "Downstream surrogate is unavailable."
            )

        if critical_result is None:
            raise RuntimeError(
                "Critical-pressure model is unavailable."
            )

        downstream = np.asarray(
            downstream,
            dtype=float,
        )

        if downstream.ndim != 2:
            raise ValueError(
                "Downstream predictions must be a 2D array."
            )

        frame = pd.DataFrame(
            downstream,
            columns=downstream_result.target_names,
        )

        demand_column = (
            self.downstream_model
            .demand_column
        )

        if not demand_column:
            raise RuntimeError(
                "Downstream model demand metadata is unavailable."
            )

        frame[
            demand_column
        ] = float(demand)

        return (
            self.critical_model
            .predict_frame(frame)
            .to_numpy(dtype=float)
        )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective_vector(
        self,
        *,
        particles: np.ndarray,
        downstream: np.ndarray,
        critical: np.ndarray,
        bounds: PRVBounds,
        reference_settings: Optional[np.ndarray],
        previous_settings: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Calculate the weighted objective for every particle.
        """

        pressures = np.concatenate(
            [
                downstream,
                critical,
            ],
            axis=1,
        )

        below = np.maximum(
            self.config.min_pressure
            - pressures,
            0.0,
        )

        above = np.maximum(
            pressures
            - self.config.max_pressure,
            0.0,
        )

        # Strongly penalize pressure-limit violations.
        violation_component = np.mean(
            below ** 2
            + above ** 2,
            axis=1,
        )

        # Once safe, encourage pressures toward the preferred region.
        target_component = np.mean(
            (
                pressures
                - self.config.target_pressure
            ) ** 2,
            axis=1,
        )

        safe_width = np.maximum(
            bounds.width,
            1e-8,
        )

        stability_component = np.zeros(
            len(particles),
            dtype=float,
        )

        if previous_settings is not None:
            normalized_delta = (
                particles
                - previous_settings
            ) / safe_width

            stability_component = np.mean(
                normalized_delta ** 2,
                axis=1,
            )

        reference_component = np.zeros(
            len(particles),
            dtype=float,
        )

        if reference_settings is not None:
            normalized_reference_delta = (
                particles
                - reference_settings
            ) / safe_width

            reference_component = np.mean(
                normalized_reference_delta ** 2,
                axis=1,
            )

        total = (
            self.config.pressure_violation_weight
            * violation_component

            + self.config.pressure_target_weight
            * target_component

            + self.config.stability_weight
            * stability_component

            + self.config.reference_weight
            * reference_component
        )

        return np.asarray(
            total,
            dtype=float,
        )

    def _objective_breakdown_for_one(
        self,
        *,
        particle: np.ndarray,
        downstream: np.ndarray,
        critical: np.ndarray,
        bounds: PRVBounds,
        reference_settings: Optional[np.ndarray],
        previous_settings: Optional[np.ndarray],
    ) -> ObjectiveBreakdown:
        """
        Return weighted objective components for the final selected particle.
        """

        pressures = np.concatenate(
            [
                np.asarray(
                    downstream,
                    dtype=float,
                ).reshape(-1),
                np.asarray(
                    critical,
                    dtype=float,
                ).reshape(-1),
            ]
        )

        below = np.maximum(
            self.config.min_pressure
            - pressures,
            0.0,
        )

        above = np.maximum(
            pressures
            - self.config.max_pressure,
            0.0,
        )

        raw_violation = float(
            np.mean(
                below ** 2
                + above ** 2
            )
        )

        raw_target = float(
            np.mean(
                (
                    pressures
                    - self.config.target_pressure
                ) ** 2
            )
        )

        raw_stability = 0.0

        if previous_settings is not None:
            raw_stability = float(
                np.mean(
                    (
                        (
                            particle
                            - previous_settings
                        )
                        / bounds.width
                    ) ** 2
                )
            )

        raw_reference = 0.0

        if reference_settings is not None:
            raw_reference = float(
                np.mean(
                    (
                        (
                            particle
                            - reference_settings
                        )
                        / bounds.width
                    ) ** 2
                )
            )

        weighted_violation = (
            self.config.pressure_violation_weight
            * raw_violation
        )

        weighted_target = (
            self.config.pressure_target_weight
            * raw_target
        )

        weighted_stability = (
            self.config.stability_weight
            * raw_stability
        )

        weighted_reference = (
            self.config.reference_weight
            * raw_reference
        )

        total = (
            weighted_violation
            + weighted_target
            + weighted_stability
            + weighted_reference
        )

        return ObjectiveBreakdown(
            total=float(total),
            pressure_violation=float(
                weighted_violation
            ),
            pressure_target=float(
                weighted_target
            ),
            stability=float(
                weighted_stability
            ),
            reference=float(
                weighted_reference
            ),
        )

    # ------------------------------------------------------------------
    # Compatibility validation
    # ------------------------------------------------------------------

    def _validate_models(
        self,
    ) -> None:
        """
        Ensure both trained models form a valid optimization chain.
        """

        downstream_result = (
            self.downstream_model.result
        )

        critical_result = (
            self.critical_model.result
        )

        if downstream_result is None:
            raise RuntimeError(
                "Train or load the downstream-pressure model "
                "before creating PSOOptimizer."
            )

        if critical_result is None:
            raise RuntimeError(
                "Train or load the critical-pressure model "
                "before creating PSOOptimizer."
            )

        if not self.downstream_model.prv_columns:
            raise RuntimeError(
                "Downstream surrogate has no PRV metadata."
            )

        demand_column = (
            self.downstream_model
            .demand_column
        )

        if not demand_column:
            raise RuntimeError(
                "Downstream surrogate has no demand metadata."
            )

        expected_critical_features = (
            list(
                downstream_result.target_names
            )
            + [demand_column]
        )

        if (
            list(
                critical_result.feature_names
            )
            != expected_critical_features
        ):
            raise ValueError(
                "Model-chain mismatch. The critical-pressure model "
                "must be trained using downstream surrogate targets "
                "followed by the same demand column.\n"
                f"Expected critical features: {expected_critical_features}\n"
                f"Actual critical features: {critical_result.feature_names}"
            )

    def _validate_bounds_against_model(
        self,
        bounds: PRVBounds,
    ) -> None:
        """Ensure PSO bounds follow the surrogate's trained PRV order."""

        expected = list(
            self.downstream_model.prv_columns
        )

        if list(bounds.names) != expected:
            raise ValueError(
                "PRV bound order must exactly match the downstream "
                "surrogate's PRV feature order.\n"
                f"Expected: {expected}\n"
                f"Received: {bounds.names}"
            )

    @staticmethod
    def _optional_setting_vector(
        values: Optional[Sequence[float]],
        bounds: PRVBounds,
        *,
        label: str,
    ) -> Optional[np.ndarray]:
        """
        Validate and clip an optional PRV-setting vector.
        """

        if values is None:
            return None

        array = np.asarray(
            values,
            dtype=float,
        ).reshape(-1)

        if (
            len(array)
            != bounds.n_prvs
        ):
            raise ValueError(
                f"{label} must contain exactly "
                f"{bounds.n_prvs} values."
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                f"{label} contains non-finite values."
            )

        return bounds.clip(
            array
        )
