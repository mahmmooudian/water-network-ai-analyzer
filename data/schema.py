"""
Semantic schema detection for water-network datasets.

The detector maps raw CSV columns into engineering groups used by the
application:

- PRV setting columns
- downstream / point-after-valve pressure columns
- critical-point pressure columns
- demand / flow column
- ignored columns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import re

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a dataset does not contain columns required by a workflow."""


@dataclass
class Schema:
    """Detected semantic groups in a water-network dataset."""

    prv_columns: List[str] = field(default_factory=list)

    point_after_valve_columns: List[str] = field(
        default_factory=list
    )

    critical_point_columns: List[str] = field(
        default_factory=list
    )

    demand_column: Optional[str] = None

    ignored_columns: List[str] = field(
        default_factory=list
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly representation."""

        return {
            "prv_columns": list(self.prv_columns),
            "point_after_valve_columns": list(
                self.point_after_valve_columns
            ),
            "critical_point_columns": list(
                self.critical_point_columns
            ),
            "demand_column": self.demand_column,
            "ignored_columns": list(self.ignored_columns),
        }

    @property
    def detected_columns(self) -> List[str]:
        """Return all columns assigned to an engineering role."""

        columns = (
            self.prv_columns
            + self.point_after_valve_columns
            + self.critical_point_columns
        )

        if self.demand_column:
            columns = columns + [self.demand_column]

        return columns

    @property
    def is_complete_for_critical_model(self) -> bool:
        """Check if critical-pressure model inputs are available."""

        return bool(
            self.point_after_valve_columns
            and self.critical_point_columns
            and self.demand_column
        )

    @property
    def is_complete_for_pso(self) -> bool:
        """Check if surrogate PSO requirements are available."""

        return bool(
            self.is_complete_for_critical_model
            and self.prv_columns
        )


@dataclass
class WaterNetworkData:
    """
    Raw dataframe together with its detected semantic schema.
    """

    raw: pd.DataFrame

    schema: Schema

    source_path: Optional[Path] = None

    @property
    def prv_data(self) -> pd.DataFrame:
        """Return PRV setting columns."""

        return self.raw.loc[
            :,
            self.schema.prv_columns
        ].copy()

    @property
    def point_after_valve_data(self) -> pd.DataFrame:
        """Return downstream pressure columns."""

        return self.raw.loc[
            :,
            self.schema.point_after_valve_columns
        ].copy()

    @property
    def critical_point_data(self) -> pd.DataFrame:
        """Return critical-point pressure columns."""

        return self.raw.loc[
            :,
            self.schema.critical_point_columns
        ].copy()

    @property
    def demand_data(self) -> pd.DataFrame:
        """Return demand column as dataframe."""

        if not self.schema.demand_column:
            return pd.DataFrame(
                index=self.raw.index
            )

        return self.raw.loc[
            :,
            [self.schema.demand_column]
        ].copy()


class SchemaDetector:
    """
    Detect common water-network naming conventions.

    Detection is deterministic and designed to avoid ambiguous
    assignments between PRV settings and downstream pressures.
    """

    DEMAND_EXACT_NAMES: Sequence[str] = (
        "p-676",
        "p_676",
        "deby",
        "demand",
        "total_demand",
        "total demand",
        "flow",
        "total_flow",
        "total flow",
    )

    DEMAND_TOKENS: Sequence[str] = (
        "demand",
        "deby",
        "flow",
    )

    PRV_METADATA_TOKENS: Sequence[str] = (
        "status",
        "state",
        "id",
        "name",
        "label",
        "type",
    )

    DOWNSTREAM_TOKENS: Sequence[str] = (
        "downstream",
        "after_valve",
        "after valve",
        "after-prv",
        "after_prv",
        "post_valve",
        "post valve",
    )

    @staticmethod
    def _clean_name(
        name: str
    ) -> str:
        """
        Normalize case and whitespace without changing
        the actual dataframe column name.
        """

        return re.sub(
            r"\s+",
            " ",
            str(name).strip().lower(),
        )

    # ----------------------------------------------------------
    # Downstream detection
    # ----------------------------------------------------------

    @classmethod
    def _is_downstream(
        cls,
        normalized: str
    ) -> bool:
        """
        Detect downstream / point-after-valve columns.
        """

        if any(
            token in normalized
            for token in cls.DOWNSTREAM_TOKENS
        ):
            return True

        # Legacy project convention:
        #
        # PRV-01-B
        # Valve1_B
        #
        return bool(
            re.search(
                r"(?:-|_)b$",
                normalized,
                flags=re.IGNORECASE,
            )
        )

    # ----------------------------------------------------------
    # PRV detection
    # ----------------------------------------------------------

    @classmethod
    def _is_prv_setting(
        cls,
        normalized: str
    ) -> bool:
        """
        Determine whether a column represents a PRV setting.
        """

        if "prv" not in normalized:
            return False

        # Important:
        #
        # PRV-01-B contains the string "PRV" but represents
        # downstream pressure, not the actual valve setting.
        #
        if cls._is_downstream(normalized):
            return False

        if any(
            token in normalized
            for token in cls.PRV_METADATA_TOKENS
        ):
            return False

        return True

    # ----------------------------------------------------------
    # Critical-point detection
    # ----------------------------------------------------------

    @staticmethod
    def _is_critical_point(
        normalized: str
    ) -> bool:
        """
        Detect critical junction / pressure-point columns.
        """

        if normalized.startswith("critical"):
            return True

        if (
            "critical_point" in normalized
            or "critical point" in normalized
        ):
            return True

        # Junction patterns:
        #
        # J-101
        # J_101
        # J101
        #
        return bool(
            re.match(
                r"^j(?:-|_)?\d+",
                normalized,
            )
        )

    # ----------------------------------------------------------
    # Demand detection
    # ----------------------------------------------------------

    @classmethod
    def _detect_demand_column(
        cls,
        columns: Sequence[str],
        normalized: Dict[str, str],
        excluded: set[str],
    ) -> Optional[str]:
        """
        Detect the most appropriate demand / flow column.
        """

        available = [
            column
            for column in columns
            if column not in excluded
        ]

        # Exact names have priority.
        #
        # This keeps legacy P-676 behavior deterministic.
        #
        for expected in cls.DEMAND_EXACT_NAMES:

            expected_normalized = cls._clean_name(
                expected
            )

            for column in available:

                if (
                    normalized[column]
                    == expected_normalized
                ):
                    return column

        # Fallback to semantic tokens.
        for column in available:

            if any(
                token in normalized[column]
                for token in cls.DEMAND_TOKENS
            ):
                return column

        return None

    # ----------------------------------------------------------
    # Main detection
    # ----------------------------------------------------------

    @classmethod
    def detect(
        cls,
        df: pd.DataFrame
    ) -> Schema:
        """
        Detect all engineering column groups.

        Detection order is intentionally:

        1. Downstream
        2. PRV
        3. Demand
        4. Critical Points

        Downstream is evaluated first so a column such as
        ``PRV-01-B`` cannot accidentally become a PRV setting.
        """

        if not isinstance(
            df,
            pd.DataFrame
        ):
            raise TypeError(
                "df must be a pandas DataFrame."
            )

        if (
            df.empty
            and len(df.columns) == 0
        ):
            raise ValueError(
                "Cannot detect schema from a dataframe "
                "with no columns."
            )

        columns = [
            str(column).strip()
            for column in df.columns
        ]

        # ------------------------------------------------------
        # Duplicate validation
        # ------------------------------------------------------

        if len(columns) != len(set(columns)):

            duplicates = sorted(
                {
                    column
                    for column in columns
                    if columns.count(column) > 1
                }
            )

            raise SchemaValidationError(
                "Duplicate column names are not supported: "
                + ", ".join(duplicates)
            )

        normalized = {
            column: cls._clean_name(column)
            for column in columns
        }

        # ------------------------------------------------------
        # Downstream pressures
        # ------------------------------------------------------

        point_after = [
            column
            for column in columns
            if cls._is_downstream(
                normalized[column]
            )
        ]

        point_after_set = set(
            point_after
        )

        # ------------------------------------------------------
        # PRV settings
        # ------------------------------------------------------

        prv = [
            column
            for column in columns
            if (
                column not in point_after_set
                and cls._is_prv_setting(
                    normalized[column]
                )
            )
        ]

        excluded_before_demand = (
            set(prv)
            | point_after_set
        )

        # ------------------------------------------------------
        # Demand
        # ------------------------------------------------------

        demand = cls._detect_demand_column(
            columns,
            normalized,
            excluded_before_demand,
        )

        # ------------------------------------------------------
        # Critical points
        # ------------------------------------------------------

        critical = [
            column
            for column in columns
            if (
                column
                not in excluded_before_demand

                and column != demand

                and cls._is_critical_point(
                    normalized[column]
                )
            )
        ]

        # ------------------------------------------------------
        # Ignored / metadata columns
        # ------------------------------------------------------

        used = (
            set(prv)
            | point_after_set
            | set(critical)
        )

        if demand:
            used.add(demand)

        ignored = [
            column
            for column in columns
            if column not in used
        ]

        return Schema(
            prv_columns=prv,
            point_after_valve_columns=point_after,
            critical_point_columns=critical,
            demand_column=demand,
            ignored_columns=ignored,
        )

    # ----------------------------------------------------------
    # Critical-model validation
    # ----------------------------------------------------------

    @staticmethod
    def validate_for_critical_model(
        schema: Schema
    ) -> None:
        """
        Validate dataset requirements for critical-pressure ML.
        """

        missing: List[str] = []

        if not schema.point_after_valve_columns:

            missing.append(
                "downstream / point-after-valve "
                "pressure columns "
                "(e.g. PRV-01-B or downstream_1)"
            )

        if not schema.critical_point_columns:

            missing.append(
                "critical-point columns "
                "(e.g. J-101 or Critical_Point_1)"
            )

        if not schema.demand_column:

            missing.append(
                "demand/flow column "
                "(e.g. Demand, Total_Demand, or P-676)"
            )

        if missing:

            raise SchemaValidationError(
                "Dataset cannot train the "
                "critical-pressure model. Missing: "
                + "; ".join(missing)
            )

    # ----------------------------------------------------------
    # PSO validation
    # ----------------------------------------------------------

    @classmethod
    def validate_for_pso(
        cls,
        schema: Schema
    ) -> None:
        """
        Validate dataset requirements for PRV optimization.
        """

        cls.validate_for_critical_model(
            schema
        )

        if not schema.prv_columns:

            raise SchemaValidationError(
                "Dataset cannot run PSO because "
                "no PRV setting columns were detected."
            )

    # ----------------------------------------------------------
    # Numeric validation
    # ----------------------------------------------------------

    @staticmethod
    def validate_numeric_columns(
        df: pd.DataFrame,
        columns: Sequence[str],
    ) -> None:
        """
        Ensure model-facing columns contain numeric data.
        """

        invalid: List[str] = []

        for column in columns:

            if column not in df.columns:

                invalid.append(
                    f"{column} (missing)"
                )

                continue

            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            original_non_null = (
                df[column]
                .notna()
                .sum()
            )

            converted_non_null = (
                converted
                .notna()
                .sum()
            )

            # If original values exist but none of them can
            # become numeric, the column is invalid.
            if (
                original_non_null > 0
                and converted_non_null == 0
            ):

                invalid.append(
                    column
                )

        if invalid:

            raise SchemaValidationError(
                "The following model columns "
                "are not numeric: "
                + ", ".join(invalid)
            )
