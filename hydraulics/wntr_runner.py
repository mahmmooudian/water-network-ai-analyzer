WNTR / EPANET hydraulic simulation adapter.

This module provides the physics-based analysis path for
Water Network AI Analyzer. It is intentionally separate from the
machine-learning surrogate and PSO optimizer.

Current responsibilities
------------------------
- Load a real EPANET .inp file
- Inspect basic network metadata
- Run EPANET hydraulic simulation through WNTR
- Return pressure, head, demand and link-flow results as pandas objects
- Filter results to selected nodes
- Export pressure results
- Produce compact pressure summaries

Important
---------
The data-driven PSO optimizer does not call this runner inside its
optimization loop. WNTR/EPANET is currently an independent physics-based
simulation and validation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence
import importlib

import numpy as np
import pandas as pd


class WNTRUnavailableError(ImportError):
    """Raised when WNTR is required but not installed."""


class HydraulicSimulationError(RuntimeError):
    """Raised when an EPANET/WNTR simulation cannot be completed."""


@dataclass(frozen=True)
class HydraulicNetworkInfo:
    """Basic metadata extracted from a WNTR WaterNetworkModel."""

    source_path: Path

    node_names: List[str]
    junction_names: List[str]
    tank_names: List[str]
    reservoir_names: List[str]

    link_names: List[str]
    pipe_names: List[str]
    pump_names: List[str]
    valve_names: List[str]

    duration_seconds: float
    hydraulic_timestep_seconds: float
    report_timestep_seconds: float

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def n_links(self) -> int:
        return len(self.link_names)

    @property
    def n_junctions(self) -> int:
        return len(self.junction_names)

    @property
    def n_valves(self) -> int:
        return len(self.valve_names)

    def as_dict(self) -> Dict[str, Any]:
        """Return a serialization-friendly summary."""

        return {
            "source_path": str(self.source_path),
            "n_nodes": self.n_nodes,
            "n_links": self.n_links,
            "n_junctions": self.n_junctions,
            "n_tanks": len(self.tank_names),
            "n_reservoirs": len(self.reservoir_names),
            "n_pipes": len(self.pipe_names),
            "n_pumps": len(self.pump_names),
            "n_valves": self.n_valves,
            "duration_seconds": self.duration_seconds,
            "hydraulic_timestep_seconds": self.hydraulic_timestep_seconds,
            "report_timestep_seconds": self.report_timestep_seconds,
        }


@dataclass
class HydraulicSimulationResult:
    """Hydraulic results returned by EPANET through WNTR."""

    pressure: pd.DataFrame
    head: pd.DataFrame
    demand: pd.DataFrame
    flowrate: pd.DataFrame

    node_names: List[str]
    link_names: List[str]

    source_path: Path
    epanet_version: float
    demand_model: str

    error_code: Optional[int] = None

    @property
    def timesteps(self) -> pd.Index:
        """Simulation timestamps in seconds."""

        return self.pressure.index

    @property
    def n_timesteps(self) -> int:
        return len(self.pressure.index)

    def pressure_summary(self) -> pd.DataFrame:
        """
        Return min/mean/max/std pressure for every returned node.
        """

        if self.pressure.empty:
            return pd.DataFrame(
                columns=[
                    "Node",
                    "MinPressure",
                    "MeanPressure",
                    "MaxPressure",
                    "StdPressure",
                ]
            )

        summary = pd.DataFrame(
            {
                "Node": self.pressure.columns,
                "MinPressure": self.pressure.min(axis=0).to_numpy(),
                "MeanPressure": self.pressure.mean(axis=0).to_numpy(),
                "MaxPressure": self.pressure.max(axis=0).to_numpy(),
                "StdPressure": self.pressure.std(axis=0).to_numpy(),
            }
        )

        return summary.reset_index(drop=True)

    def pressure_at_time(
        self,
        time_seconds: int | float,
    ) -> pd.Series:
        """
        Return pressure values at an exact reported simulation time.
        """

        if time_seconds not in self.pressure.index:
            available = list(self.pressure.index)

            raise KeyError(
                f"Simulation time {time_seconds} is unavailable. "
                f"Available report times include: {available[:10]}"
            )

        return self.pressure.loc[time_seconds].copy()

    def final_pressure(self) -> pd.Series:
        """Return the final reported node pressures."""

        if self.pressure.empty:
            return pd.Series(dtype=float)

        return self.pressure.iloc[-1].copy()

    def export_pressure_csv(
        self,
        path: str | Path,
    ) -> Path:
        """Export pressure timeseries to CSV."""

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.pressure.to_csv(
            output,
            index=True,
            index_label="TimeSeconds",
        )

        return output

    def export_summary_csv(
        self,
        path: str | Path,
    ) -> Path:
        """Export node pressure summary to CSV."""

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.pressure_summary().to_csv(
            output,
            index=False,
        )

        return output


class WNTRRunner:
    """
    Thin adapter around WNTR's EPANET simulator.

    Parameters
    ----------
    inp_path:
        Optional path to an EPANET INP file.
    epanet_version:
        EPANET toolkit version used by WNTR. Supported values are 2.0 and 2.2.
    convergence_error:
        If True, request an exception when EPANET does not converge.
    """

    SUPPORTED_EPANET_VERSIONS = (
        2.0,
        2.2,
    )

    SUPPORTED_DEMAND_MODELS = {
        "DD",
        "PDD",
    }

    def __init__(
        self,
        inp_path: Optional[str | Path] = None,
        *,
        epanet_version: float = 2.2,
        convergence_error: bool = True,
    ) -> None:
        self.epanet_version = float(
            epanet_version
        )

        if (
            self.epanet_version
            not in self.SUPPORTED_EPANET_VERSIONS
        ):
            raise ValueError(
                "epanet_version must be either 2.0 or 2.2."
            )

        self.convergence_error = bool(
            convergence_error
        )

        self.source_path: Optional[Path] = None
        self.network = None

        if inp_path is not None:
            self.load(
                inp_path
            )

    # ------------------------------------------------------------------
    # Dependency handling
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if WNTR can be imported."""

        try:
            return (
                importlib.util.find_spec(
                    "wntr"
                )
                is not None
            )

        except (
            ImportError,
            AttributeError,
        ):
            return False

    @staticmethod
    def _import_wntr():
        """Import WNTR only when the hydraulic path is actually used."""

        try:
            import wntr  # type: ignore

        except ImportError as exc:
            raise WNTRUnavailableError(
                "WNTR is not installed. Install project dependencies with "
                "'pip install -r requirements.txt' or install WNTR directly "
                "with 'pip install wntr'."
            ) from exc

        return wntr

    # ------------------------------------------------------------------
    # Network loading
    # ------------------------------------------------------------------

    def load(
        self,
        inp_path: str | Path,
    ) -> HydraulicNetworkInfo:
        """
        Load a real EPANET INP network.
        """

        wntr = self._import_wntr()

        path = Path(
            inp_path
        ).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                f"EPANET INP file not found: {path}"
            )

        if path.suffix.lower() != ".inp":
            raise ValueError(
                "Hydraulic network file must use the .inp extension."
            )

        try:
            network = (
                wntr.network
                .WaterNetworkModel(
                    str(path)
                )
            )

        except Exception as exc:
            raise HydraulicSimulationError(
                f"Could not load EPANET network '{path.name}': {exc}"
            ) from exc

        self.network = network
        self.source_path = path.resolve()

        return self.network_info()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def network_info(
        self,
    ) -> HydraulicNetworkInfo:
        """Return metadata for the currently loaded network."""

        network = self._require_network()

        assert self.source_path is not None

        return HydraulicNetworkInfo(
            source_path=self.source_path,

            node_names=list(
                network.node_name_list
            ),
            junction_names=list(
                network.junction_name_list
            ),
            tank_names=list(
                network.tank_name_list
            ),
            reservoir_names=list(
                network.reservoir_name_list
            ),

            link_names=list(
                network.link_name_list
            ),
            pipe_names=list(
                network.pipe_name_list
            ),
            pump_names=list(
                network.pump_name_list
            ),
            valve_names=list(
                network.valve_name_list
            ),

            duration_seconds=float(
                network.options.time.duration
            ),
            hydraulic_timestep_seconds=float(
                network.options.time.hydraulic_timestep
            ),
            report_timestep_seconds=float(
                network.options.time.report_timestep
            ),
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        node_names: Optional[Sequence[str]] = None,
        demand_model: Optional[str] = None,
    ) -> HydraulicSimulationResult:
        """
        Run EPANET hydraulic simulation through WNTR.

        Parameters
        ----------
        node_names:
            Optional node subset. If omitted, all simulated nodes are returned.
        demand_model:
            Optional hydraulic demand model: ``DD`` or ``PDD``.
        """

        wntr = self._import_wntr()
        network = self._require_network()

        selected_nodes = self._validate_node_selection(
            node_names
        )

        original_demand_model = str(
            network.options.hydraulic.demand_model
        )

        resolved_demand_model = (
            original_demand_model
            if demand_model is None
            else self._normalize_demand_model(
                demand_model
            )
        )

        if demand_model is not None:
            network.options.hydraulic.demand_model = (
                resolved_demand_model
            )

        # EPANET 2.0 does not support PDD.
        if (
            self.epanet_version == 2.0
            and resolved_demand_model.upper() == "PDD"
        ):
            network.options.hydraulic.demand_model = (
                original_demand_model
            )

            raise ValueError(
                "EPANET 2.0 does not support pressure-dependent "
                "demand (PDD). Use EPANET 2.2 or demand model DD."
            )

        try:
            simulator = (
                wntr.sim
                .EpanetSimulator(
                    network
                )
            )

            # EpanetSimulator creates binary/report artifacts. Keep them
            # outside the repository and remove them automatically.
            with TemporaryDirectory(
                prefix="water_network_ai_"
            ) as temp_dir:

                file_prefix = str(
                    Path(temp_dir)
                    / "epanet_run"
                )

                raw_results = simulator.run_sim(
                    file_prefix=file_prefix,
                    version=self.epanet_version,
                    convergence_error=self.convergence_error,
                )

        except Exception as exc:
            raise HydraulicSimulationError(
                "EPANET hydraulic simulation failed: "
                f"{exc}"
            ) from exc

        finally:
            # Preserve the loaded model's original option after each run.
            if demand_model is not None:
                network.options.hydraulic.demand_model = (
                    original_demand_model
                )

        try:
            pressure = self._result_frame(
                raw_results.node,
                "pressure",
                label="node pressure",
            )

            head = self._result_frame(
                raw_results.node,
                "head",
                label="node head",
            )

            demand = self._result_frame(
                raw_results.node,
                "demand",
                label="node demand",
            )

            flowrate = self._result_frame(
                raw_results.link,
                "flowrate",
                label="link flowrate",
            )

        except Exception as exc:
            raise HydraulicSimulationError(
                "Simulation completed, but hydraulic results could "
                f"not be parsed: {exc}"
            ) from exc

        if selected_nodes is not None:
            pressure = pressure.loc[
                :,
                selected_nodes,
            ].copy()

            head = head.loc[
                :,
                selected_nodes,
            ].copy()

            demand = demand.loc[
                :,
                selected_nodes,
            ].copy()

        error_code = getattr(
            raw_results,
            "error_code",
            None,
        )

        assert self.source_path is not None

        return HydraulicSimulationResult(
            pressure=pressure,
            head=head,
            demand=demand,
            flowrate=flowrate,

            node_names=list(
                pressure.columns
            ),
            link_names=list(
                flowrate.columns
            ),

            source_path=self.source_path,
            epanet_version=self.epanet_version,
            demand_model=resolved_demand_model.upper(),
            error_code=(
                int(error_code)
                if error_code is not None
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_network(self):
        """Return loaded WaterNetworkModel or raise a clear error."""

        if self.network is None:
            raise RuntimeError(
                "No EPANET network has been loaded. "
                "Call load(path) first."
            )

        return self.network

    def _validate_node_selection(
        self,
        node_names: Optional[
            Sequence[str]
        ],
    ) -> Optional[List[str]]:
        """Validate an optional node subset."""

        if node_names is None:
            return None

        network = self._require_network()

        selected = [
            str(name)
            for name in node_names
        ]

        if not selected:
            raise ValueError(
                "node_names cannot be an empty sequence."
            )

        if (
            len(selected)
            != len(set(selected))
        ):
            raise ValueError(
                "node_names contains duplicates."
            )

        available = set(
            network.node_name_list
        )

        missing = [
            name
            for name in selected
            if name not in available
        ]

        if missing:
            raise ValueError(
                "Unknown network nodes: "
                + ", ".join(missing)
            )

        return selected

    @classmethod
    def _normalize_demand_model(
        cls,
        value: str,
    ) -> str:
        """Normalize accepted demand-model aliases."""

        normalized = str(
            value
        ).strip().upper()

        aliases = {
            "DDA": "DD",
            "DEMAND DRIVEN": "DD",
            "DEMAND-DRIVEN": "DD",

            "PDA": "PDD",
            "PRESSURE DEPENDENT": "PDD",
            "PRESSURE-DEPENDENT": "PDD",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if (
            normalized
            not in cls.SUPPORTED_DEMAND_MODELS
        ):
            raise ValueError(
                "demand_model must be DD or PDD."
            )

        return normalized

    @staticmethod
    def _result_frame(
        result_group,
        key: str,
        *,
        label: str,
    ) -> pd.DataFrame:
        """
        Read a WNTR result DataFrame and validate its numeric contents.
        """

        if key not in result_group:
            raise KeyError(
                f"WNTR result does not contain {label!r}."
            )

        frame = result_group[
            key
        ]

        if not isinstance(
            frame,
            pd.DataFrame,
        ):
            frame = pd.DataFrame(
                frame
            )

        frame = frame.copy()

        # Preserve WNTR's physical units and timestamps; only validate that
        # returned values can be represented numerically.
        for column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        if frame.empty:
            raise HydraulicSimulationError(
                f"WNTR returned an empty {label} result."
            )

        values = frame.to_numpy(
            dtype=float
        )

        if np.isinf(values).any():
            raise HydraulicSimulationError(
                f"WNTR returned infinite values in {label}."
            )

        return frame


def simulate_inp(
    inp_path: str | Path,
    *,
    node_names: Optional[Sequence[str]] = None,
    demand_model: Optional[str] = None,
    epanet_version: float = 2.2,
    convergence_error: bool = True,
) -> HydraulicSimulationResult:
    """
    Convenience function for one-off EPANET simulations.
    """

    runner = WNTRRunner(
        inp_path=inp_path,
        epanet_version=epanet_version,
        convergence_error=convergence_error,
    )

    return runner.run(
        node_names=node_names,
        demand_model=demand_model,
    )
