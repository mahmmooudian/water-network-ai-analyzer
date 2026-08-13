"""
Tkinter desktop interface for Water Network AI Analyzer.

The GUI is intentionally thin. Engineering logic lives in dedicated modules:

- data.schema                  -> semantic dataset detection
- models.critical_pressure     -> critical-point XGBoost model
- models.downstream_pressure   -> downstream-pressure surrogate
- optimization.pso             -> engineering-aware PSO
- hydraulics.wntr_runner       -> WNTR / EPANET simulation
- visualization.plots          -> reusable matplotlib figures

This separation keeps UI code independent from ML and hydraulic logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import threading
import traceback

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from ..config import AppConfig, DEFAULT_CONFIG
from ..data.schema import (
    SchemaDetector,
    SchemaValidationError,
    WaterNetworkData,
)
from ..hydraulics.wntr_runner import (
    HydraulicSimulationResult,
    WNTRRunner,
)
from ..models.critical_pressure import (
    CriticalPressureModel,
    RegressionResult,
)
from ..models.downstream_pressure import (
    DownstreamPressureModel,
)
from ..optimization.pso import (
    PSOOptimizationResult,
    PSOOptimizer,
)
from ..visualization.plots import (
    close_figure,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_hydraulic_pressure,
    plot_hydraulic_pressure_summary,
    plot_objective_components,
    plot_optimized_pressure_summary,
    plot_prediction_series,
    plot_prv_settings,
    plot_pso_convergence,
)


class WaterNetworkAIApp:
    """
    Main desktop application.

    The class can either receive an existing Tk root or create one itself.

    Examples
    --------
    >>> app = WaterNetworkAIApp()
    >>> app.run()
    """

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.root = root or tk.Tk()
        self.config = config or DEFAULT_CONFIG

        # ------------------------------------------------------------------
        # Application state
        # ------------------------------------------------------------------

        self.data: Optional[WaterNetworkData] = None

        self.critical_model = CriticalPressureModel(
            self.config
        )

        self.downstream_model = DownstreamPressureModel(
            self.config
        )

        self.optimization_result: Optional[
            PSOOptimizationResult
        ] = None

        self.wntr_runner: Optional[
            WNTRRunner
        ] = None

        self.hydraulic_result: Optional[
            HydraulicSimulationResult
        ] = None

        self._working = False
        self._table_column_order: List[str] = []

        # ------------------------------------------------------------------
        # UI
        # ------------------------------------------------------------------

        self._configure_root()
        self._configure_style()
        self._build_menu()
        self._build_layout()

        self._set_status(
            "Ready. Load a CSV dataset to begin."
        )

    # ==================================================================
    # Application lifecycle
    # ==================================================================

    def run(self) -> None:
        """Start Tkinter's event loop."""

        self.root.mainloop()

    def _configure_root(self) -> None:
        """Configure the main application window."""

        self.root.title(
            f"{self.config.app_name} - Professional Edition "
            f"v{self.config.version}"
        )

        self.root.geometry(
            "1420x900"
        )

        self.root.minsize(
            1120,
            720,
        )

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self._on_close,
        )

    def _configure_style(self) -> None:
        """Configure conservative native Tk styling."""

        style = ttk.Style(
            self.root
        )

        if (
            "vista"
            in style.theme_names()
        ):
            style.theme_use(
                "vista"
            )

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                17,
                "bold",
            ),
        )

        style.configure(
            "Subtitle.TLabel",
            font=(
                "Segoe UI",
                10,
            ),
        )

        style.configure(
            "Section.TLabel",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        )

        style.configure(
            "Treeview",
            rowheight=25,
        )

    # ==================================================================
    # Menus / layout
    # ==================================================================

    def _build_menu(self) -> None:
        """Create application menus."""

        menubar = tk.Menu(
            self.root
        )

        # ---------------------- File ----------------------

        file_menu = tk.Menu(
            menubar,
            tearoff=False,
        )

        file_menu.add_command(
            label="Load CSV",
            command=self.load_csv,
        )

        file_menu.add_command(
            label="Save Dataset",
            command=self.save_dataset,
        )

        file_menu.add_separator()

        file_menu.add_command(
            label="Save Critical Model",
            command=self.save_critical_model,
        )

        file_menu.add_command(
            label="Load Critical Model",
            command=self.load_critical_model,
        )

        file_menu.add_command(
            label="Save Downstream Model",
            command=self.save_downstream_model,
        )

        file_menu.add_command(
            label="Load Downstream Model",
            command=self.load_downstream_model,
        )

        file_menu.add_separator()

        file_menu.add_command(
            label="Exit",
            command=self._on_close,
        )

        menubar.add_cascade(
            label="File",
            menu=file_menu,
        )

        # ---------------- Machine Learning ----------------

        model_menu = tk.Menu(
            menubar,
            tearoff=False,
        )

        model_menu.add_command(
            label="Train Critical-Point Model",
            command=self.train_critical_model,
        )

        model_menu.add_command(
            label="Train Downstream Surrogate",
            command=self.train_downstream_model,
        )

        model_menu.add_separator()

        model_menu.add_command(
            label="Predict Critical Points",
            command=self.predict_critical_points_dialog,
        )

        model_menu.add_command(
            label="Actual vs Predicted",
            command=self.show_actual_vs_predicted,
        )

        model_menu.add_command(
            label="Prediction Series",
            command=self.show_prediction_series,
        )

        model_menu.add_command(
            label="Feature Importance",
            command=self.show_feature_importance,
        )

        menubar.add_cascade(
            label="Machine Learning",
            menu=model_menu,
        )

        # ---------------- Optimization ----------------

        optimization_menu = tk.Menu(
            menubar,
            tearoff=False,
        )

        optimization_menu.add_command(
            label="Run PRV Optimization",
            command=self.run_pso,
        )

        optimization_menu.add_command(
            label="Export Optimization CSV",
            command=self.export_optimization,
        )

        optimization_menu.add_separator()

        optimization_menu.add_command(
            label="PRV Settings Plot",
            command=self.show_prv_settings_plot,
        )

        optimization_menu.add_command(
            label="Pressure Summary Plot",
            command=self.show_optimized_pressure_plot,
        )

        optimization_menu.add_command(
            label="Objective Components",
            command=self.show_objective_components,
        )

        optimization_menu.add_command(
            label="PSO Convergence",
            command=self.show_convergence_dialog,
        )

        menubar.add_cascade(
            label="Optimization",
            menu=optimization_menu,
        )

        # ---------------- Hydraulics ----------------

        hydraulics_menu = tk.Menu(
            menubar,
            tearoff=False,
        )

        hydraulics_menu.add_command(
            label="Load EPANET INP",
            command=self.load_inp,
        )

        hydraulics_menu.add_command(
            label="Run Demand-Driven Simulation",
            command=lambda: self.run_hydraulic_simulation(
                "DD"
            ),
        )

        hydraulics_menu.add_command(
            label="Run Pressure-Dependent Simulation",
            command=lambda: self.run_hydraulic_simulation(
                "PDD"
            ),
        )

        menubar.add_cascade(
            label="Hydraulics",
            menu=hydraulics_menu,
        )

        # ---------------- Help ----------------

        help_menu = tk.Menu(
            menubar,
            tearoff=False,
        )

        help_menu.add_command(
            label="About",
            command=self.show_about,
        )

        menubar.add_cascade(
            label="Help",
            menu=help_menu,
        )

        self.root.configure(
            menu=menubar
        )

    def _build_layout(self) -> None:
        """Build the main application layout."""

        header = ttk.Frame(
            self.root,
            padding=(
                16,
                12,
            ),
        )

        header.pack(
            fill=tk.X
        )

        ttk.Label(
            header,
            text=self.config.app_name,
            style="Title.TLabel",
        ).pack(
            side=tk.LEFT
        )

        ttk.Label(
            header,
            text=(
                "Leakage-safe ML · Surrogate PSO · "
                "Physics-based WNTR / EPANET"
            ),
            style="Subtitle.TLabel",
        ).pack(
            side=tk.LEFT,
            padx=(
                18,
                0,
            ),
            pady=(
                6,
                0,
            ),
        )

        # ---------------- Toolbar ----------------

        toolbar = ttk.Frame(
            self.root,
            padding=(
                16,
                0,
                16,
                10,
            ),
        )

        toolbar.pack(
            fill=tk.X
        )

        buttons = [
            (
                "Load CSV",
                self.load_csv,
            ),
            (
                "Train Critical",
                self.train_critical_model,
            ),
            (
                "Train Downstream",
                self.train_downstream_model,
            ),
            (
                "Predict",
                self.predict_critical_points_dialog,
            ),
            (
                "Run PSO",
                self.run_pso,
            ),
            (
                "Load INP",
                self.load_inp,
            ),
            (
                "Run EPANET",
                lambda: self.run_hydraulic_simulation(
                    "DD"
                ),
            ),
        ]

        for label, command in buttons:
            ttk.Button(
                toolbar,
                text=label,
                command=command,
            ).pack(
                side=tk.LEFT,
                padx=(
                    0,
                    7,
                ),
            )

        self.progress = ttk.Progressbar(
            toolbar,
            mode="indeterminate",
            length=190,
        )

        self.progress.pack(
            side=tk.RIGHT
        )

        # ---------------- Notebook ----------------

        self.notebook = ttk.Notebook(
            self.root
        )

        self.notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=16,
            pady=(
                0,
                10,
            ),
        )

        self.overview_tab = ttk.Frame(
            self.notebook
        )

        self.data_tab = ttk.Frame(
            self.notebook
        )

        self.model_tab = ttk.Frame(
            self.notebook
        )

        self.optimization_tab = ttk.Frame(
            self.notebook
        )

        self.hydraulics_tab = ttk.Frame(
            self.notebook
        )

        self.log_tab = ttk.Frame(
            self.notebook
        )

        self.notebook.add(
            self.overview_tab,
            text="Overview",
        )

        self.notebook.add(
            self.data_tab,
            text="Data",
        )

        self.notebook.add(
            self.model_tab,
            text="Models",
        )

        self.notebook.add(
            self.optimization_tab,
            text="Optimization",
        )

        self.notebook.add(
            self.hydraulics_tab,
            text="Hydraulics",
        )

        self.notebook.add(
            self.log_tab,
            text="Log",
        )

        self._build_overview_tab()
        self._build_data_tab()
        self._build_model_tab()
        self._build_optimization_tab()
        self._build_hydraulics_tab()
        self._build_log_tab()

        # ---------------- Status bar ----------------

        status_frame = ttk.Frame(
            self.root,
            padding=(
                16,
                0,
                16,
                10,
            ),
        )

        status_frame.pack(
            fill=tk.X
        )

        self.status_var = tk.StringVar()

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
        ).pack(
            side=tk.LEFT
        )

    def _build_overview_tab(self) -> None:
        """Build dataset/schema overview tab."""

        container = ttk.Frame(
            self.overview_tab,
            padding=16,
        )

        container.pack(
            fill=tk.BOTH,
            expand=True,
        )

        ttk.Label(
            container,
            text="Dataset & Semantic Schema",
            style="Section.TLabel",
        ).pack(
            anchor="w"
        )

        self.overview_text = tk.Text(
            container,
            wrap=tk.WORD,
            font=(
                "Consolas",
                10,
            ),
        )

        self.overview_text.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(
                8,
                0,
            ),
        )

        self.overview_text.configure(
            state=tk.DISABLED
        )

    def _build_data_tab(self) -> None:
        """Build editable dataset table."""

        controls = ttk.Frame(
            self.data_tab,
            padding=(
                10,
                10,
                10,
                4,
            ),
        )

        controls.pack(
            fill=tk.X
        )

        ttk.Label(
            controls,
            text="Filter rows:",
        ).pack(
            side=tk.LEFT
        )

        self.filter_var = tk.StringVar()

        entry = ttk.Entry(
            controls,
            textvariable=self.filter_var,
            width=35,
        )

        entry.pack(
            side=tk.LEFT,
            padx=(
                8,
                8,
            ),
        )

        entry.bind(
            "<KeyRelease>",
            lambda _event: self._populate_data_table(),
        )

        ttk.Label(
            controls,
            text="Double-click a cell to edit.",
        ).pack(
            side=tk.LEFT,
            padx=(
                12,
                0,
            ),
        )

        frame = ttk.Frame(
            self.data_tab,
            padding=(
                10,
                4,
                10,
                10,
            ),
        )

        frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.data_tree = ttk.Treeview(
            frame,
            show="headings",
        )

        vsb = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=self.data_tree.yview,
        )

        hsb = ttk.Scrollbar(
            frame,
            orient="horizontal",
            command=self.data_tree.xview,
        )

        self.data_tree.configure(
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )

        self.data_tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vsb.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        hsb.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        frame.rowconfigure(
            0,
            weight=1,
        )

        frame.columnconfigure(
            0,
            weight=1,
        )

        self.data_tree.bind(
            "<Double-1>",
            self._edit_cell,
        )

    def _build_model_tab(self) -> None:
        """Build model-training and evaluation tab."""

        controls = ttk.Frame(
            self.model_tab,
            padding=12,
        )

        controls.pack(
            fill=tk.X
        )

        ttk.Button(
            controls,
            text="Train Critical Model",
            command=self.train_critical_model,
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Train Downstream Surrogate",
            command=self.train_downstream_model,
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Predict",
            command=self.predict_critical_points_dialog,
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Actual vs Predicted",
            command=self.show_actual_vs_predicted,
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Feature Importance",
            command=self.show_feature_importance,
        ).pack(
            side=tk.LEFT
        )

        self.tune_models_var = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            controls,
            text="Hyperparameter tuning",
            variable=self.tune_models_var,
        ).pack(
            side=tk.RIGHT
        )

        body = ttk.Panedwindow(
            self.model_tab,
            orient=tk.HORIZONTAL,
        )

        body.pack(
            fill=tk.BOTH,
            expand=True,
            padx=12,
            pady=(
                0,
                12,
            ),
        )

        left = ttk.Frame(
            body
        )

        right = ttk.Frame(
            body
        )

        body.add(
            left,
            weight=1,
        )

        body.add(
            right,
            weight=2,
        )

        ttk.Label(
            left,
            text="Critical Model Metrics",
            style="Section.TLabel",
        ).pack(
            anchor="w",
            pady=(
                0,
                6,
            ),
        )

        self.metrics_tree = ttk.Treeview(
            left,
            columns=(
                "Metric",
                "Value",
            ),
            show="headings",
            height=12,
        )

        self.metrics_tree.heading(
            "Metric",
            text="Metric",
        )

        self.metrics_tree.heading(
            "Value",
            text="Value",
        )

        self.metrics_tree.column(
            "Metric",
            width=170,
        )

        self.metrics_tree.column(
            "Value",
            width=140,
        )

        self.metrics_tree.pack(
            fill=tk.BOTH,
            expand=True,
        )

        ttk.Label(
            right,
            text="Model Details",
            style="Section.TLabel",
        ).pack(
            anchor="w",
            pady=(
                0,
                6,
            ),
        )

        self.model_text = tk.Text(
            right,
            wrap=tk.NONE,
            font=(
                "Consolas",
                10,
            ),
        )

        self.model_text.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.model_text.configure(
            state=tk.DISABLED
        )

    def _build_optimization_tab(self) -> None:
        """Build optimization result tab."""

        controls = ttk.Frame(
            self.optimization_tab,
            padding=12,
        )

        controls.pack(
            fill=tk.X
        )

        ttk.Label(
            controls,
            text="Periods:",
        ).pack(
            side=tk.LEFT
        )

        self.pso_periods_var = tk.IntVar(
            value=self.config.optimization_hours
        )

        ttk.Spinbox(
            controls,
            from_=1,
            to=500,
            textvariable=self.pso_periods_var,
            width=7,
        ).pack(
            side=tk.LEFT,
            padx=(
                6,
                12,
            ),
        )

        ttk.Button(
            controls,
            text="Run PSO",
            command=self.run_pso,
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="PRV Settings",
            command=self.show_prv_settings_plot,
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Pressure Summary",
            command=self.show_optimized_pressure_plot,
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Convergence",
            command=self.show_convergence_dialog,
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Export CSV",
            command=self.export_optimization,
        ).pack(
            side=tk.RIGHT
        )

        frame = ttk.Frame(
            self.optimization_tab,
            padding=(
                12,
                0,
                12,
                12,
            ),
        )

        frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        columns = (
            "Period",
            "Demand",
            "Objective",
            "MinPressure",
            "MeanPressure",
            "MaxPressure",
        )

        self.optim_tree = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
        )

        for column in columns:
            self.optim_tree.heading(
                column,
                text=column,
            )

            self.optim_tree.column(
                column,
                width=135,
                anchor="center",
            )

        vsb = ttk.Scrollbar(
            frame,
            orient="vertical",
            command=self.optim_tree.yview,
        )

        hsb = ttk.Scrollbar(
            frame,
            orient="horizontal",
            command=self.optim_tree.xview,
        )

        self.optim_tree.configure(
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )

        self.optim_tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        vsb.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        hsb.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        frame.rowconfigure(
            0,
            weight=1,
        )

        frame.columnconfigure(
            0,
            weight=1,
        )

    def _build_hydraulics_tab(self) -> None:
        """Build WNTR / EPANET simulation tab."""

        controls = ttk.Frame(
            self.hydraulics_tab,
            padding=12,
        )

        controls.pack(
            fill=tk.X
        )

        ttk.Button(
            controls,
            text="Load EPANET INP",
            command=self.load_inp,
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Run DD",
            command=lambda: self.run_hydraulic_simulation(
                "DD"
            ),
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Run PDD",
            command=lambda: self.run_hydraulic_simulation(
                "PDD"
            ),
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Pressure Plot",
            command=self.show_hydraulic_pressure_plot,
        ).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Button(
            controls,
            text="Pressure Summary",
            command=self.show_hydraulic_summary_plot,
        ).pack(
            side=tk.LEFT
        )

        ttk.Button(
            controls,
            text="Export Pressure CSV",
            command=self.export_hydraulic_pressure,
        ).pack(
            side=tk.RIGHT
        )

        container = ttk.Panedwindow(
            self.hydraulics_tab,
            orient=tk.HORIZONTAL,
        )

        container.pack(
            fill=tk.BOTH,
            expand=True,
            padx=12,
            pady=(
                0,
                12,
            ),
        )

        left = ttk.Frame(
            container
        )

        right = ttk.Frame(
            container
        )

        container.add(
            left,
            weight=1,
        )

        container.add(
            right,
            weight=2,
        )

        ttk.Label(
            left,
            text="Network Information",
            style="Section.TLabel",
        ).pack(
            anchor="w",
            pady=(
                0,
                6,
            ),
        )

        self.hydraulic_info_text = tk.Text(
            left,
            wrap=tk.WORD,
            font=(
                "Consolas",
                10,
            ),
        )

        self.hydraulic_info_text.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.hydraulic_info_text.configure(
            state=tk.DISABLED
        )

        ttk.Label(
            right,
            text="Pressure Summary",
            style="Section.TLabel",
        ).pack(
            anchor="w",
            pady=(
                0,
                6,
            ),
        )

        self.hydraulic_tree = ttk.Treeview(
            right,
            columns=(
                "Node",
                "Min",
                "Mean",
                "Max",
                "Std",
            ),
            show="headings",
        )

        for column in (
            "Node",
            "Min",
            "Mean",
            "Max",
            "Std",
        ):
            self.hydraulic_tree.heading(
                column,
                text=column,
            )

        self.hydraulic_tree.column(
            "Node",
            width=150,
        )

        for column in (
            "Min",
            "Mean",
            "Max",
            "Std",
        ):
            self.hydraulic_tree.column(
                column,
                width=115,
                anchor="center",
            )

        self.hydraulic_tree.pack(
            fill=tk.BOTH,
            expand=True,
        )

    def _build_log_tab(self) -> None:
        """Build application log tab."""

        container = ttk.Frame(
            self.log_tab,
            padding=12,
        )

        container.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.log_text = tk.Text(
            container,
            wrap=tk.WORD,
            font=(
                "Consolas",
                10,
            ),
        )

        self.log_text.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.log_text.configure(
            state=tk.DISABLED
        )

    # ==================================================================
    # General UI helpers
    # ==================================================================

    def _set_status(
        self,
        text: str,
    ) -> None:
        """Update status bar."""

        self.status_var.set(
            str(text)
        )

    def _append_log(
        self,
        text: str,
    ) -> None:
        """Append a line to the log tab."""

        self.log_text.configure(
            state=tk.NORMAL
        )

        self.log_text.insert(
            tk.END,
            str(text).rstrip()
            + "\n",
        )

        self.log_text.see(
            tk.END
        )

        self.log_text.configure(
            state=tk.DISABLED
        )

    @staticmethod
    def _replace_text(
        widget: tk.Text,
        text: str,
    ) -> None:
        """Replace content of a read-only Text widget."""

        widget.configure(
            state=tk.NORMAL
        )

        widget.delete(
            "1.0",
            tk.END,
        )

        widget.insert(
            tk.END,
            text,
        )

        widget.configure(
            state=tk.DISABLED
        )

    def _set_working(
        self,
        working: bool,
        status: Optional[str] = None,
    ) -> None:
        """Toggle long-running-operation indicator."""

        self._working = bool(
            working
        )

        if working:
            self.progress.start(
                10
            )

        else:
            self.progress.stop()

        if status is not None:
            self._set_status(
                status
            )

    def _run_task(
        self,
        label: str,
        task: Callable[
            [],
            Any,
        ],
        done: Callable[
            [Any],
            None,
        ],
    ) -> None:
        """
        Run CPU/IO work outside Tk's UI thread.
        """

        if self._working:
            messagebox.showinfo(
                "Operation in Progress",
                "Another operation is currently running.",
                parent=self.root,
            )
            return

        self._set_working(
            True,
            label,
        )

        self._append_log(
            label
        )

        def worker() -> None:
            try:
                value = task()

            except Exception as exc:
                trace = traceback.format_exc()

                self.root.after(
                    0,
                    lambda: self._task_error(
                        label,
                        exc,
                        trace,
                    ),
                )

            else:
                self.root.after(
                    0,
                    lambda: self._task_success(
                        done,
                        value,
                    ),
                )

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()

    def _task_error(
        self,
        label: str,
        exc: Exception,
        trace: str,
    ) -> None:
        """Handle background-task error on the UI thread."""

        self._set_working(
            False,
            "Operation failed.",
        )

        self._append_log(
            f"ERROR: {label}\n{trace}"
        )

        messagebox.showerror(
            "Operation Failed",
            str(exc),
            parent=self.root,
        )

    def _task_success(
        self,
        callback: Callable[
            [Any],
            None,
        ],
        value: Any,
    ) -> None:
        """Handle successful background task."""

        self._set_working(
            False
        )

        try:
            callback(
                value
            )

        except Exception as exc:
            self._append_log(
                "ERROR while updating UI:\n"
                + traceback.format_exc()
            )

            messagebox.showerror(
                "UI Error",
                str(exc),
                parent=self.root,
            )

    # ==================================================================
    # Dataset handling
    # ==================================================================

    @staticmethod
    def _read_csv(
        path: str | Path,
    ) -> pd.DataFrame:
        """
        Read CSV using a small encoding fallback chain.
        """

        errors: List[str] = []

        for encoding in (
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin1",
        ):
            try:
                return pd.read_csv(
                    path,
                    encoding=encoding,
                )

            except UnicodeDecodeError as exc:
                errors.append(
                    f"{encoding}: {exc}"
                )

        raise ValueError(
            "Could not decode CSV file with supported encodings.\n"
            + "\n".join(errors)
        )

    def _require_data(
        self,
    ) -> WaterNetworkData:
        """Return loaded dataset or raise."""

        if self.data is None:
            raise RuntimeError(
                "Load a CSV dataset first."
            )

        return self.data

    def load_csv(self) -> None:
        """Load CSV and detect semantic schema."""

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Water Network CSV",
            filetypes=[
                (
                    "CSV files",
                    "*.csv",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:
            frame = self._read_csv(
                path
            )

            # Strip accidental spreadsheet whitespace from headers.
            frame.columns = [
                str(column).strip()
                for column in frame.columns
            ]

            schema = SchemaDetector.detect(
                frame
            )

            self.data = WaterNetworkData(
                raw=frame,
                schema=schema,
                source_path=Path(
                    path
                ),
            )

            # Models trained on another dataset must not silently remain
            # active after a new dataset is loaded.
            self.critical_model = CriticalPressureModel(
                self.config
            )

            self.downstream_model = DownstreamPressureModel(
                self.config
            )

            self.optimization_result = None

            self._refresh_all()

            self.notebook.select(
                self.overview_tab
            )

            self._set_status(
                f"Loaded {Path(path).name}: "
                f"{len(frame):,} rows × {len(frame.columns)} columns"
            )

            self._append_log(
                f"Loaded CSV: {path}"
            )

        except Exception as exc:
            messagebox.showerror(
                "CSV Load Error",
                str(exc),
                parent=self.root,
            )

    def save_dataset(self) -> None:
        """Save the current editable dataframe."""

        try:
            data = self._require_data()

        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                str(exc),
                parent=self.root,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Dataset",
            defaultextension=".csv",
            filetypes=[
                (
                    "CSV files",
                    "*.csv",
                ),
            ],
        )

        if not path:
            return

        try:
            data.raw.to_csv(
                path,
                index=False,
            )

            self._set_status(
                f"Dataset saved: {Path(path).name}"
            )

            self._append_log(
                f"Saved dataset: {path}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                str(exc),
                parent=self.root,
            )

    def _refresh_all(self) -> None:
        """Refresh all data-dependent UI sections."""

        self._refresh_overview()
        self._populate_data_table()
        self._refresh_model_view()
        self._refresh_optimization_view()

    def _refresh_overview(self) -> None:
        """Refresh dataset and schema summary."""

        if self.data is None:
            self._replace_text(
                self.overview_text,
                "No dataset loaded.",
            )
            return

        data = self.data
        schema = data.schema

        missing_counts = (
            data.raw
            .isna()
            .sum()
        )

        total_missing = int(
            missing_counts.sum()
        )

        lines = [
            "DATASET",
            "=======",
            f"Source: {data.source_path or 'In memory'}",
            f"Rows: {len(data.raw):,}",
            f"Columns: {len(data.raw.columns)}",
            f"Missing values: {total_missing:,}",
            "",
            "SEMANTIC SCHEMA",
            "===============",
            f"PRV settings ({len(schema.prv_columns)}):",
            "  "
            + (
                ", ".join(
                    schema.prv_columns
                )
                if schema.prv_columns
                else "None"
            ),
            "",
            (
                "Downstream / point-after-valve "
                f"({len(schema.point_after_valve_columns)}):"
            ),
            "  "
            + (
                ", ".join(
                    schema.point_after_valve_columns
                )
                if schema.point_after_valve_columns
                else "None"
            ),
            "",
            f"Critical points ({len(schema.critical_point_columns)}):",
            "  "
            + (
                ", ".join(
                    schema.critical_point_columns
                )
                if schema.critical_point_columns
                else "None"
            ),
            "",
            f"Demand: {schema.demand_column or 'Not detected'}",
            "",
            f"Ignored / metadata ({len(schema.ignored_columns)}):",
            "  "
            + (
                ", ".join(
                    schema.ignored_columns
                )
                if schema.ignored_columns
                else "None"
            ),
            "",
            "WORKFLOW READINESS",
            "==================",
            (
                "Critical model: "
                + (
                    "READY"
                    if schema.is_complete_for_critical_model
                    else "MISSING REQUIRED COLUMNS"
                )
            ),
            (
                "Surrogate PSO: "
                + (
                    "READY"
                    if schema.is_complete_for_pso
                    else "MISSING REQUIRED COLUMNS"
                )
            ),
        ]

        self._replace_text(
            self.overview_text,
            "\n".join(
                lines
            ),
        )

    def _populate_data_table(self) -> None:
        """Populate editable table with a limited set of matching rows."""

        self.data_tree.delete(
            *self.data_tree.get_children()
        )

        if self.data is None:
            self.data_tree.configure(
                columns=()
            )
            return

        frame = self.data.raw

        columns = list(
            frame.columns
        )

        self._table_column_order = columns

        self.data_tree.configure(
            columns=columns
        )

        for column in columns:
            self.data_tree.heading(
                column,
                text=column,
            )

            self.data_tree.column(
                column,
                width=125,
                minwidth=80,
            )

        filter_text = (
            self.filter_var.get()
            .strip()
            .lower()
        )

        visible_rows = 0

        for index, row in frame.iterrows():
            values = [
                ""
                if pd.isna(
                    row[column]
                )
                else str(
                    row[column]
                )
                for column in columns
            ]

            if filter_text:
                searchable = " ".join(
                    values
                ).lower()

                if (
                    filter_text
                    not in searchable
                ):
                    continue

            self.data_tree.insert(
                "",
                "end",
                iid=str(index),
                values=values,
            )

            visible_rows += 1

            # Keep the desktop UI responsive on very large datasets.
            if visible_rows >= 1500:
                break

    def _edit_cell(
        self,
        event: tk.Event,
    ) -> None:
        """Edit a double-clicked Treeview cell."""

        if self.data is None:
            return

        region = self.data_tree.identify_region(
            event.x,
            event.y,
        )

        if region != "cell":
            return

        row_id = self.data_tree.identify_row(
            event.y
        )

        column_id = self.data_tree.identify_column(
            event.x
        )

        if (
            not row_id
            or not column_id
        ):
            return

        column_index = (
            int(
                column_id.replace(
                    "#",
                    "",
                )
            )
            - 1
        )

        if (
            column_index < 0
            or column_index
            >= len(
                self._table_column_order
            )
        ):
            return

        column = self._table_column_order[
            column_index
        ]

        row_index = int(
            row_id
        )

        current = self.data.raw.at[
            row_index,
            column,
        ]

        new_value = simpledialog.askstring(
            "Edit Cell",
            f"Row {row_index}\nColumn: {column}",
            initialvalue=(
                ""
                if pd.isna(current)
                else str(current)
            ),
            parent=self.root,
        )

        if new_value is None:
            return

        if new_value.strip() == "":
            value: Any = np.nan

        else:
            try:
                value = float(
                    new_value
                )

            except ValueError:
                value = new_value

        self.data.raw.at[
            row_index,
            column,
        ] = value

        # Re-detect schema in case a user edited metadata/header-related data
        # is not necessary because headers did not change.
        self._populate_data_table()

        # Existing models are now potentially stale.
        self.critical_model = CriticalPressureModel(
            self.config
        )

        self.downstream_model = DownstreamPressureModel(
            self.config
        )

        self.optimization_result = None

        self._refresh_model_view()
        self._refresh_optimization_view()

        self._set_status(
            "Dataset edited. Models were reset and should be retrained."
        )

    # ==================================================================
    # Critical-pressure model
    # ==================================================================

    def train_critical_model(self) -> None:
        """Train the critical-point pressure model."""

        try:
            data = self._require_data()

            SchemaDetector.validate_for_critical_model(
                data.schema
            )

            feature_names = (
                list(
                    data.schema.point_after_valve_columns
                )
                + [
                    str(
                        data.schema.demand_column
                    )
                ]
            )

            target_names = list(
                data.schema.critical_point_columns
            )

            SchemaDetector.validate_numeric_columns(
                data.raw,
                feature_names
                + target_names,
            )

        except Exception as exc:
            messagebox.showerror(
                "Training Error",
                str(exc),
                parent=self.root,
            )
            return

        tune = bool(
            self.tune_models_var.get()
        )

        def task() -> RegressionResult:
            return self.critical_model.train(
                data.raw,
                feature_names,
                target_names,
                tune=tune,
            )

        def done(
            result: RegressionResult,
        ) -> None:
            self._refresh_model_view()

            self.notebook.select(
                self.model_tab
            )

            self._set_status(
                "Critical model trained | "
                f"RMSE={result.metrics.get('RMSE', float('nan')):.4f} | "
                f"R2={result.metrics.get('R2', float('nan')):.4f} | "
                f"{result.training_seconds:.2f}s"
            )

            self._append_log(
                "Critical model training completed."
            )

        self._run_task(
            "Training critical-pressure model...",
            task,
            done,
        )

    def _refresh_model_view(self) -> None:
        """Refresh metrics and model metadata."""

        self.metrics_tree.delete(
            *self.metrics_tree.get_children()
        )

        result = self.critical_model.result

        if result is None:
            self._replace_text(
                self.model_text,
                (
                    "Critical model: not trained\n"
                    "Downstream surrogate: "
                    + (
                        "trained"
                        if self.downstream_model.is_ready
                        else "not trained"
                    )
                ),
            )
            return

        for metric, value in result.metrics.items():
            self.metrics_tree.insert(
                "",
                "end",
                values=(
                    metric,
                    f"{value:.6f}",
                ),
            )

        if (
            result.best_cv_score
            is not None
        ):
            self.metrics_tree.insert(
                "",
                "end",
                values=(
                    "Best CV score",
                    f"{result.best_cv_score:.6f}",
                ),
            )

        details = [
            "CRITICAL-PRESSURE MODEL",
            "=======================",
            f"Features ({len(result.feature_names)}):",
            "  "
            + ", ".join(
                result.feature_names
            ),
            "",
            f"Targets ({len(result.target_names)}):",
            "  "
            + ", ".join(
                result.target_names
            ),
            "",
            f"Training time: {result.training_seconds:.3f} s",
            "",
            "Best parameters:",
            (
                pd.Series(
                    result.best_params,
                    dtype=object,
                ).to_string()
                if result.best_params
                else "  Hyperparameter search not used."
            ),
            "",
            "Per-target metrics:",
            (
                result.per_target_metrics
                .to_string(
                    index=False
                )
                if not result.per_target_metrics.empty
                else "  Unavailable"
            ),
            "",
            "DOWNSTREAM SURROGATE",
            "====================",
        ]

        downstream = (
            self.downstream_model.result
        )

        if downstream is None:
            details.append(
                "Not trained."
            )

        else:
            details.extend(
                [
                    f"Features: {', '.join(downstream.feature_names)}",
                    f"Targets: {', '.join(downstream.target_names)}",
                    (
                        "RMSE: "
                        f"{downstream.metrics.get('RMSE', float('nan')):.6f}"
                    ),
                    (
                        "R2: "
                        f"{downstream.metrics.get('R2', float('nan')):.6f}"
                    ),
                ]
            )

        self._replace_text(
            self.model_text,
            "\n".join(
                details
            ),
        )

    def predict_critical_points_dialog(
        self,
    ) -> None:
        """Open manual critical-pressure prediction dialog."""

        result = self.critical_model.result

        if result is None:
            messagebox.showerror(
                "Prediction Error",
                "Train or load the critical-pressure model first.",
                parent=self.root,
            )
            return

        window = tk.Toplevel(
            self.root
        )

        window.title(
            "Critical-Point Prediction"
        )

        window.geometry(
            "580x650"
        )

        window.transient(
            self.root
        )

        container = ttk.Frame(
            window,
            padding=14,
        )

        container.pack(
            fill=tk.BOTH,
            expand=True,
        )

        ttk.Label(
            container,
            text="Enter Model Features",
            style="Section.TLabel",
        ).pack(
            anchor="w",
            pady=(
                0,
                10,
            ),
        )

        entries: Dict[
            str,
            tk.StringVar,
        ] = {}

        for feature in result.feature_names:
            row = ttk.Frame(
                container
            )

            row.pack(
                fill=tk.X,
                pady=4,
            )

            ttk.Label(
                row,
                text=feature,
                width=28,
            ).pack(
                side=tk.LEFT
            )

            variable = tk.StringVar()

            entries[
                feature
            ] = variable

            ttk.Entry(
                row,
                textvariable=variable,
            ).pack(
                side=tk.LEFT,
                fill=tk.X,
                expand=True,
            )

        result_text = tk.Text(
            container,
            height=10,
            font=(
                "Consolas",
                10,
            ),
        )

        result_text.pack(
            fill=tk.BOTH,
            expand=True,
            pady=(
                12,
                8,
            ),
        )

        def do_predict() -> None:
            try:
                values: Dict[
                    str,
                    float,
                ] = {}

                for feature, variable in entries.items():
                    text = variable.get().strip()

                    if not text:
                        values[
                            feature
                        ] = np.nan
                    else:
                        values[
                            feature
                        ] = float(
                            text
                        )

                prediction = (
                    self.critical_model
                    .predict_one(values)
                )

                result_text.delete(
                    "1.0",
                    tk.END,
                )

                result_text.insert(
                    tk.END,
                    "\n".join(
                        f"{name}: {value:.6f}"
                        for name, value
                        in prediction.items()
                    ),
                )

            except Exception as exc:
                messagebox.showerror(
                    "Prediction Error",
                    str(exc),
                    parent=window,
                )

        ttk.Button(
            container,
            text="Predict",
            command=do_predict,
        ).pack()

    # ==================================================================
    # Downstream surrogate
    # ==================================================================

    def train_downstream_model(self) -> None:
        """Train PRV + demand -> downstream pressure surrogate."""

        try:
            data = self._require_data()

            SchemaDetector.validate_for_pso(
                data.schema
            )

            assert (
                data.schema.demand_column
                is not None
            )

            SchemaDetector.validate_numeric_columns(
                data.raw,
                (
                    list(
                        data.schema.prv_columns
                    )
                    + [
                        data.schema.demand_column
                    ]
                    + list(
                        data.schema.point_after_valve_columns
                    )
                ),
            )

        except Exception as exc:
            messagebox.showerror(
                "Training Error",
                str(exc),
                parent=self.root,
            )
            return

        tune = bool(
            self.tune_models_var.get()
        )

        def task():
            return self.downstream_model.train(
                data.raw,
                data.schema.prv_columns,
                str(
                    data.schema.demand_column
                ),
                data.schema.point_after_valve_columns,
                tune=tune,
            )

        def done(
            result,
        ) -> None:
            self._refresh_model_view()

            self.notebook.select(
                self.model_tab
            )

            self._set_status(
                "Downstream surrogate trained | "
                f"RMSE={result.metrics.get('RMSE', float('nan')):.4f} | "
                f"R2={result.metrics.get('R2', float('nan')):.4f}"
            )

            self._append_log(
                "Downstream surrogate training completed."
            )

        self._run_task(
            "Training downstream-pressure surrogate...",
            task,
            done,
        )

    # ==================================================================
    # Model persistence
    # ==================================================================

    def save_critical_model(self) -> None:
        """Save critical model."""

        if (
            self.critical_model.result
            is None
        ):
            messagebox.showerror(
                "Save Error",
                "Train or load the critical model first.",
                parent=self.root,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Critical Model",
            defaultextension=".joblib",
            filetypes=[
                (
                    "Joblib model",
                    "*.joblib",
                ),
            ],
        )

        if not path:
            return

        try:
            self.critical_model.save(
                path
            )

            self._set_status(
                f"Critical model saved: {Path(path).name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                str(exc),
                parent=self.root,
            )

    def load_critical_model(self) -> None:
        """Load critical model."""

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Critical Model",
            filetypes=[
                (
                    "Joblib model",
                    "*.joblib",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:
            self.critical_model.load(
                path
            )

            self._refresh_model_view()

            self._set_status(
                f"Critical model loaded: {Path(path).name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Load Error",
                str(exc),
                parent=self.root,
            )

    def save_downstream_model(self) -> None:
        """Save downstream surrogate."""

        if not self.downstream_model.is_ready:
            messagebox.showerror(
                "Save Error",
                "Train or load the downstream surrogate first.",
                parent=self.root,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Downstream Model",
            defaultextension=".joblib",
            filetypes=[
                (
                    "Joblib model",
                    "*.joblib",
                ),
            ],
        )

        if not path:
            return

        try:
            self.downstream_model.save(
                path
            )

            self._set_status(
                f"Downstream model saved: {Path(path).name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                str(exc),
                parent=self.root,
            )

    def load_downstream_model(self) -> None:
        """Load downstream surrogate."""

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load Downstream Model",
            filetypes=[
                (
                    "Joblib model",
                    "*.joblib",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:
            self.downstream_model.load(
                path
            )

            self._refresh_model_view()

            self._set_status(
                f"Downstream model loaded: {Path(path).name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Load Error",
                str(exc),
                parent=self.root,
            )

    # ==================================================================
    # Model visualization
    # ==================================================================

    def _require_critical_result(
        self,
    ) -> Optional[RegressionResult]:
        """Return critical result or show UI error."""

        result = self.critical_model.result

        if result is None:
            messagebox.showerror(
                "Model Error",
                "Train or load the critical model first.",
                parent=self.root,
            )
            return None

        return result

    def _choose_target(
        self,
        target_names: Sequence[str],
        title: str,
    ) -> Optional[str]:
        """Choose a target when a model has multiple outputs."""

        names = list(
            target_names
        )

        if not names:
            return None

        if len(names) == 1:
            return names[0]

        selected = simpledialog.askstring(
            title,
            "Target name:\n\n"
            + "\n".join(
                names
            ),
            initialvalue=names[0],
            parent=self.root,
        )

        if selected is None:
            return None

        selected = (
            selected.strip()
        )

        if selected not in names:
            messagebox.showerror(
                "Target Error",
                f"Unknown target: {selected}",
                parent=self.root,
            )
            return None

        return selected

    def show_actual_vs_predicted(self) -> None:
        """Show actual-vs-predicted diagnostic."""

        result = self._require_critical_result()

        if result is None:
            return

        target = self._choose_target(
            result.target_names,
            "Actual vs Predicted",
        )

        if target is None:
            return

        try:
            figure = plot_actual_vs_predicted(
                result,
                target=target,
            )

            self._show_figure(
                figure,
                f"Actual vs Predicted - {target}",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def show_prediction_series(self) -> None:
        """Show hold-out prediction series."""

        result = self._require_critical_result()

        if result is None:
            return

        target = self._choose_target(
            result.target_names,
            "Prediction Series",
        )

        if target is None:
            return

        try:
            figure = plot_prediction_series(
                result,
                target=target,
            )

            self._show_figure(
                figure,
                f"Prediction Series - {target}",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def show_feature_importance(self) -> None:
        """Show critical-model feature importance."""

        result = self._require_critical_result()

        if result is None:
            return

        try:
            figure = plot_feature_importance(
                result
            )

            self._show_figure(
                figure,
                "Feature Importance",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    # ==================================================================
    # PSO optimization
    # ==================================================================

    def run_pso(self) -> None:
        """Train any missing compatible models and run surrogate PSO."""

        try:
            data = self._require_data()

            SchemaDetector.validate_for_pso(
                data.schema
            )

            periods = int(
                self.pso_periods_var.get()
            )

            if periods < 1:
                raise ValueError(
                    "Optimization periods must be at least 1."
                )

            assert (
                data.schema.demand_column
                is not None
            )

        except Exception as exc:
            messagebox.showerror(
                "Optimization Error",
                str(exc),
                parent=self.root,
            )
            return

        # Automatic training is deliberately untuned here when missing.
        # The user can train tuned models explicitly from the Models tab.
        def task() -> PSOOptimizationResult:
            schema = data.schema

            critical_features = (
                list(
                    schema.point_after_valve_columns
                )
                + [
                    str(
                        schema.demand_column
                    )
                ]
            )

            if (
                self.critical_model.result
                is None
            ):
                self.critical_model.train(
                    data.raw,
                    critical_features,
                    schema.critical_point_columns,
                    tune=False,
                )

            if not self.downstream_model.is_ready:
                self.downstream_model.train(
                    data.raw,
                    schema.prv_columns,
                    str(
                        schema.demand_column
                    ),
                    schema.point_after_valve_columns,
                    tune=False,
                )

            optimizer = PSOOptimizer(
                self.downstream_model,
                self.critical_model,
                self.config,
            )

            return optimizer.optimize_dataframe(
                data.raw,
                periods=periods,
            )

        def done(
            result: PSOOptimizationResult,
        ) -> None:
            self.optimization_result = (
                result
            )

            self._refresh_model_view()
            self._refresh_optimization_view()

            self.notebook.select(
                self.optimization_tab
            )

            frame = result.to_frame()

            violations = int(
                (
                    (
                        frame[
                            "MinPressure"
                        ]
                        < self.config.min_pressure
                    )
                    | (
                        frame[
                            "MaxPressure"
                        ]
                        > self.config.max_pressure
                    )
                ).sum()
            )

            self._set_status(
                f"PSO complete | {len(result.periods)} periods | "
                f"periods with pressure-limit violation={violations}"
            )

            self._append_log(
                "PSO optimization completed."
            )

            messagebox.showinfo(
                "Optimization Complete",
                (
                    f"Periods optimized: {len(result.periods)}\n"
                    f"PRVs: {result.bounds.n_prvs}\n"
                    f"Periods with pressure-limit violation: {violations}\n\n"
                    "Method: data-driven surrogate PSO.\n"
                    "It is not a hydraulic solver."
                ),
                parent=self.root,
            )

        self._run_task(
            "Training required surrogates and running PSO...",
            task,
            done,
        )

    def _refresh_optimization_view(self) -> None:
        """Refresh optimization table."""

        self.optim_tree.delete(
            *self.optim_tree.get_children()
        )

        result = self.optimization_result

        if result is None:
            return

        for item in result.periods:
            self.optim_tree.insert(
                "",
                "end",
                values=(
                    item.period,
                    f"{item.demand:.4f}",
                    f"{item.objective.total:.6f}",
                    f"{item.min_pressure:.4f}",
                    f"{item.mean_pressure:.4f}",
                    f"{item.max_pressure:.4f}",
                ),
            )

    def export_optimization(self) -> None:
        """Export flattened PSO results."""

        if (
            self.optimization_result
            is None
        ):
            messagebox.showerror(
                "Export Error",
                "Run PSO optimization first.",
                parent=self.root,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Optimization Results",
            defaultextension=".csv",
            filetypes=[
                (
                    "CSV files",
                    "*.csv",
                ),
            ],
        )

        if not path:
            return

        try:
            self.optimization_result.to_frame().to_csv(
                path,
                index=False,
            )

            self._set_status(
                f"Optimization exported: {Path(path).name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Export Error",
                str(exc),
                parent=self.root,
            )

    def _require_optimization(
        self,
    ) -> Optional[
        PSOOptimizationResult
    ]:
        """Return optimization result or show error."""

        if (
            self.optimization_result
            is None
        ):
            messagebox.showerror(
                "Optimization Error",
                "Run PSO optimization first.",
                parent=self.root,
            )
            return None

        return self.optimization_result

    def show_prv_settings_plot(self) -> None:
        """Show optimized PRV trajectories."""

        result = self._require_optimization()

        if result is None:
            return

        try:
            self._show_figure(
                plot_prv_settings(
                    result
                ),
                "Optimized PRV Settings",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def show_optimized_pressure_plot(self) -> None:
        """Show optimized pressure min/mean/max."""

        result = self._require_optimization()

        if result is None:
            return

        try:
            self._show_figure(
                plot_optimized_pressure_summary(
                    result
                ),
                "Optimized Pressure Summary",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def show_objective_components(self) -> None:
        """Show PSO objective components by period."""

        result = self._require_optimization()

        if result is None:
            return

        try:
            self._show_figure(
                plot_objective_components(
                    result
                ),
                "PSO Objective Components",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def show_convergence_dialog(self) -> None:
        """Choose optimization period and show convergence."""

        result = self._require_optimization()

        if result is None:
            return

        selected = simpledialog.askinteger(
            "PSO Convergence",
            (
                "Period to display "
                f"(1-{len(result.periods)}):"
            ),
            minvalue=1,
            maxvalue=len(
                result.periods
            ),
            initialvalue=1,
            parent=self.root,
        )

        if selected is None:
            return

        period_result = result.periods[
            selected - 1
        ]

        try:
            self._show_figure(
                plot_pso_convergence(
                    period_result
                ),
                (
                    "PSO Convergence - Period "
                    f"{period_result.period}"
                ),
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    # ==================================================================
    # WNTR / EPANET
    # ==================================================================

    def load_inp(self) -> None:
        """Load a real EPANET INP file."""

        if not WNTRRunner.is_available():
            messagebox.showinfo(
                "WNTR Not Available",
                (
                    "WNTR is not available in the current Python "
                    "environment.\n\n"
                    "Install dependencies with:\n"
                    "pip install -r requirements.txt"
                ),
                parent=self.root,
            )
            return

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load EPANET INP",
            filetypes=[
                (
                    "EPANET INP",
                    "*.inp",
                ),
                (
                    "All files",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        def task():
            runner = WNTRRunner(
                epanet_version=2.2
            )

            info = runner.load(
                path
            )

            return (
                runner,
                info,
            )

        def done(
            value,
        ) -> None:
            runner, info = value

            self.wntr_runner = runner
            self.hydraulic_result = None

            lines = [
                "EPANET NETWORK",
                "==============",
                f"Source: {info.source_path}",
                f"Nodes: {info.n_nodes}",
                f"Junctions: {info.n_junctions}",
                f"Tanks: {len(info.tank_names)}",
                f"Reservoirs: {len(info.reservoir_names)}",
                f"Links: {info.n_links}",
                f"Pipes: {len(info.pipe_names)}",
                f"Pumps: {len(info.pump_names)}",
                f"Valves: {info.n_valves}",
                "",
                f"Duration: {info.duration_seconds} s",
                (
                    "Hydraulic timestep: "
                    f"{info.hydraulic_timestep_seconds} s"
                ),
                (
                    "Report timestep: "
                    f"{info.report_timestep_seconds} s"
                ),
            ]

            self._replace_text(
                self.hydraulic_info_text,
                "\n".join(
                    lines
                ),
            )

            self.notebook.select(
                self.hydraulics_tab
            )

            self._set_status(
                f"Loaded EPANET network: {Path(path).name}"
            )

            self._append_log(
                f"Loaded EPANET INP: {path}"
            )

        self._run_task(
            "Loading EPANET network...",
            task,
            done,
        )

    def run_hydraulic_simulation(
        self,
        demand_model: str = "DD",
    ) -> None:
        """Run WNTR EpanetSimulator."""

        if self.wntr_runner is None:
            self.load_inp()

            # load_inp is asynchronous; user can click Run after load.
            return

        runner = self.wntr_runner

        # If CSV critical points happen to match actual network node names,
        # restrict returned node results to those points. Otherwise return all.
        node_names: Optional[
            Sequence[str]
        ] = None

        if (
            self.data is not None
            and self.data.schema.critical_point_columns
        ):
            available = set(
                runner.network_info().node_names
            )

            candidates = [
                name
                for name in self.data.schema.critical_point_columns
                if name in available
            ]

            if candidates:
                node_names = candidates

        def task():
            return runner.run(
                node_names=node_names,
                demand_model=demand_model,
            )

        def done(
            result: HydraulicSimulationResult,
        ) -> None:
            self.hydraulic_result = result

            self._refresh_hydraulic_results()

            self.notebook.select(
                self.hydraulics_tab
            )

            self._set_status(
                "Hydraulic simulation complete | "
                f"{result.n_timesteps} report times × "
                f"{len(result.node_names)} nodes | "
                f"Demand model={result.demand_model}"
            )

            self._append_log(
                "WNTR / EPANET simulation completed."
            )

        self._run_task(
            f"Running EPANET simulation ({demand_model})...",
            task,
            done,
        )

    def _refresh_hydraulic_results(self) -> None:
        """Refresh hydraulic summary table."""

        self.hydraulic_tree.delete(
            *self.hydraulic_tree.get_children()
        )

        if (
            self.hydraulic_result
            is None
        ):
            return

        summary = (
            self.hydraulic_result
            .pressure_summary()
        )

        for _, row in summary.iterrows():
            self.hydraulic_tree.insert(
                "",
                "end",
                values=(
                    row[
                        "Node"
                    ],
                    f"{row['MinPressure']:.4f}",
                    f"{row['MeanPressure']:.4f}",
                    f"{row['MaxPressure']:.4f}",
                    f"{row['StdPressure']:.4f}",
                ),
            )

    def _require_hydraulic_result(
        self,
    ) -> Optional[
        HydraulicSimulationResult
    ]:
        """Return hydraulic result or show UI error."""

        if (
            self.hydraulic_result
            is None
        ):
            messagebox.showerror(
                "Hydraulic Error",
                "Run an EPANET simulation first.",
                parent=self.root,
            )
            return None

        return self.hydraulic_result

    def show_hydraulic_pressure_plot(self) -> None:
        """Show WNTR pressure time series."""

        result = (
            self._require_hydraulic_result()
        )

        if result is None:
            return

        try:
            self._show_figure(
                plot_hydraulic_pressure(
                    result,
                    max_nodes=10,
                ),
                "WNTR / EPANET Pressure",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def show_hydraulic_summary_plot(self) -> None:
        """Show mean pressure by node."""

        result = (
            self._require_hydraulic_result()
        )

        if result is None:
            return

        try:
            self._show_figure(
                plot_hydraulic_pressure_summary(
                    result
                ),
                "Hydraulic Pressure Summary",
            )

        except Exception as exc:
            messagebox.showerror(
                "Plot Error",
                str(exc),
                parent=self.root,
            )

    def export_hydraulic_pressure(self) -> None:
        """Export hydraulic pressure timeseries."""

        result = (
            self._require_hydraulic_result()
        )

        if result is None:
            return

        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Hydraulic Pressure",
            defaultextension=".csv",
            filetypes=[
                (
                    "CSV files",
                    "*.csv",
                ),
            ],
        )

        if not path:
            return

        try:
            result.export_pressure_csv(
                path
            )

            self._set_status(
                f"Hydraulic pressure exported: {Path(path).name}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Export Error",
                str(exc),
                parent=self.root,
            )

    # ==================================================================
    # Figure windows
    # ==================================================================

    def _show_figure(
        self,
        figure: Figure,
        title: str,
    ) -> None:
        """Embed a reusable matplotlib Figure in a Tk window."""

        window = tk.Toplevel(
            self.root
        )

        window.title(
            title
        )

        window.geometry(
            "1080x740"
        )

        canvas = FigureCanvasTkAgg(
            figure,
            master=window,
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )

        toolbar = NavigationToolbar2Tk(
            canvas,
            window,
        )

        toolbar.update()

        button_frame = ttk.Frame(
            window,
            padding=6,
        )

        button_frame.pack(
            fill=tk.X
        )

        def save_plot() -> None:
            path = filedialog.asksaveasfilename(
                parent=window,
                title="Save Plot",
                defaultextension=".png",
                filetypes=[
                    (
                        "PNG",
                        "*.png",
                    ),
                    (
                        "PDF",
                        "*.pdf",
                    ),
                    (
                        "SVG",
                        "*.svg",
                    ),
                ],
            )

            if not path:
                return

            try:
                figure.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight",
                )

            except Exception as exc:
                messagebox.showerror(
                    "Save Plot Error",
                    str(exc),
                    parent=window,
                )

        ttk.Button(
            button_frame,
            text="Save Plot",
            command=save_plot,
        ).pack(
            side=tk.RIGHT
        )

        def close() -> None:
            close_figure(
                figure
            )

            window.destroy()

        window.protocol(
            "WM_DELETE_WINDOW",
            close,
        )

    # ==================================================================
    # Miscellaneous
    # ==================================================================

    def show_about(self) -> None:
        """Show application description."""

        messagebox.showinfo(
            "About",
            (
                f"{self.config.app_name}\n"
                f"Version {self.config.version}\n\n"
                "Leakage-safe XGBoost regression\n"
                "Data-driven downstream-pressure surrogate\n"
                "Engineering-aware Particle Swarm Optimization\n"
                "Critical-point pressure prediction\n"
                "Real WNTR / EPANET simulation\n\n"
                "The PSO path is explicitly surrogate-based and "
                "is not presented as a hydraulic solver."
            ),
            parent=self.root,
        )

    def _on_close(self) -> None:
        """Close figures and application safely."""

        if self._working:
            confirmed = messagebox.askyesno(
                "Exit",
                (
                    "An operation is currently running.\n"
                    "Exit anyway?"
                ),
                parent=self.root,
            )

            if not confirmed:
                return

        plt.close(
            "all"
        )

        self.root.destroy()


def launch_app(
    config: Optional[AppConfig] = None,
) -> None:
    """
    Convenience entry point used by the project's root ``main.py``.
    """

    app = WaterNetworkAIApp(
        config=config
    )

    app.run()
