"""Main window: Data / NCA / Compartmental tabs. Wires together the grid,
import wizard, worker thread, and results/plot widgets."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pkpd.core.data_model import Dataset, Route, ValidationError
from pkpd.core.units import CONC_UNITS, DOSE_UNITS, TIME_UNITS, compartmental_units, nca_units
from pkpd.pk import nca
from pkpd.pk.compartmental import fitting, models
from pkpd.ui.grid_widget import DataGrid
from pkpd.ui.help_panel import HelpPanel
from pkpd.ui.import_wizard import ImportWizard
from pkpd.ui.pk_results_view import ResultsTable
from pkpd.ui.plot_widgets import ConcTimePlot
from pkpd.workers.background_task import BackgroundTask

COMPARTMENTAL_CHOICES = {
    "1-compartment: IV bolus (K, V)": ("1c", "iv_bolus"),
    "1-compartment: IV bolus (Cl, V)": ("1c", "iv_bolus_cl"),
    "1-compartment: IV infusion": ("1c", "iv_infusion"),
    "1-compartment: Extravascular": ("1c", "extravascular"),
    "2-compartment: IV bolus": ("2c", "iv_bolus"),
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PKPD Software — Pharmacokinetics")
        self.resize(1400, 750)

        self.dataset: Dataset | None = None
        # Separate task handles per feature — sharing one attribute meant a
        # Compartmental fit started while an NCA run was still finishing (or
        # vice versa) would overwrite the only Python reference to the other's
        # still-running QThread, which PySide6 can hang/crash on when the
        # thread object gets garbage-collected mid-run.
        self._nca_task: BackgroundTask | None = None
        self._comp_task: BackgroundTask | None = None

        # Left nav list + stacked pages — mirrors the source design's
        # sidebar (PROJECT/ANALYSIS section list), not a browser tab strip.
        # QTabWidget's West position only rotates tab text 90°, which isn't
        # what a sidebar nav looks like, hence QListWidget + QStackedWidget.
        nav = QListWidget()
        nav.setObjectName("sidebarNav")
        nav.setFixedWidth(160)
        pages = QStackedWidget()
        nav_tooltips = {
            "Data": "Enter or import concentration-time data.",
            "NCA": "Run non-compartmental analysis on the loaded data.",
            "Compartmental": "Fit a compartmental PK model (1- or 2-compartment) to the loaded data.",
        }
        for label, page in (
            ("Data", self._build_data_tab()),
            ("NCA", self._build_nca_tab()),
            ("Compartmental", self._build_compartmental_tab()),
        ):
            item = QListWidgetItem(label)
            item.setToolTip(nav_tooltips[label])
            nav.addItem(item)
            pages.addWidget(page)
        nav.currentRowChanged.connect(pages.setCurrentIndex)
        nav.setCurrentRow(0)

        nav_col = QVBoxLayout()
        nav_col.setContentsMargins(0, 0, 0, 0)
        nav_col.addWidget(nav)
        nav_widget = QWidget()
        nav_widget.setLayout(nav_col)

        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)
        main_row.addWidget(nav_widget)
        main_row.addWidget(pages, stretch=1)
        main_area = QWidget()
        main_area.setLayout(main_row)

        guide_btn = QPushButton("User Guide")
        guide_btn.setCheckable(True)
        guide_btn.setToolTip("Show/hide the instructions panel on the right.")
        guide_btn.toggled.connect(self._toggle_help_panel)

        self.mode_toggle = QPushButton("Manual Mode")
        self.mode_toggle.setCheckable(True)
        self.mode_toggle.setToolTip(
            "Switch between Automatic (minimal, guided — subject_id/time/concentration/dose/route only)\n"
            "and Manual (full control over AUC method, λz mode, exclusions, steady state)."
        )
        self.mode_toggle.toggled.connect(self._set_manual_mode)

        brand = QLabel("PKPD Studio")
        brand.setStyleSheet(
            'font-family: "Cormorant Garamond", "Constantia", "Georgia", serif;'
            "font-weight: 600; font-size: 18px; color: #20613e;"
        )
        top_bar = QHBoxLayout()
        top_bar.addWidget(brand)
        top_bar.addStretch()
        top_bar.addWidget(self.mode_toggle)
        top_bar.addWidget(guide_btn)

        self.help_panel = HelpPanel()
        self.help_panel.hide()

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(8, 6, 8, 8)
        central_layout.addLayout(top_bar)

        self._splitter = QSplitter()
        self._splitter.addWidget(main_area)
        self._splitter.addWidget(self.help_panel)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        central_layout.addWidget(self._splitter)

        self.setCentralWidget(central)

        # Default to Automatic — the friendlier default for a first-time
        # user; both tab-building methods above already populated the
        # widget references _set_manual_mode needs.
        self._set_manual_mode(False)

    def _toggle_help_panel(self, checked: bool) -> None:
        self.help_panel.setVisible(checked)

    def _set_manual_mode(self, manual: bool) -> None:
        self.mode_toggle.setText("Automatic Mode" if manual else "Manual Mode")
        for w in self._advanced_nca_widgets:
            w.setVisible(manual)
        if manual:
            self.grid.show_column("weight")
        else:
            self.grid.hide_column("weight")
        self._update_infusion_column_visibility()

    def _update_infusion_column_visibility(self) -> None:
        """infusion_duration only matters for iv_infusion — show it in
        Manual mode always, or in Automatic mode only when that route is
        actually selected (bolus/extravascular never need it)."""
        manual = self.mode_toggle.isChecked()
        if manual or self.route_combo.currentText() == "iv_infusion":
            self.grid.show_column("infusion_duration")
        else:
            self.grid.hide_column("infusion_duration")

    # ---------- Data tab ----------
    def _build_data_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        btn_row = QHBoxLayout()
        import_btn = QPushButton("Import File...")
        import_btn.setToolTip("Load a CSV/Excel file and map its columns to the grid.")
        import_btn.clicked.connect(self._import_file)
        add_row_btn = QPushButton("Add Row")
        add_row_btn.setToolTip("Add a blank row to the grid.")
        del_row_btn = QPushButton("Delete Selected Rows")
        del_row_btn.setToolTip("Remove the currently selected row(s) from the grid.")
        btn_row.addWidget(import_btn)
        btn_row.addWidget(add_row_btn)
        btn_row.addWidget(del_row_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Units — Time:"))
        self.time_unit_combo = QComboBox()
        self.time_unit_combo.addItems(TIME_UNITS)
        self.time_unit_combo.setToolTip("Unit the 'time' column is entered in. Labels results — doesn't convert.")
        unit_row.addWidget(self.time_unit_combo)
        unit_row.addWidget(QLabel("Concentration:"))
        self.conc_unit_combo = QComboBox()
        self.conc_unit_combo.addItems(CONC_UNITS)
        self.conc_unit_combo.setToolTip("Unit the 'concentration' column is entered in. Labels results — doesn't convert.")
        unit_row.addWidget(self.conc_unit_combo)
        unit_row.addWidget(QLabel("Dose:"))
        self.dose_unit_combo = QComboBox()
        self.dose_unit_combo.addItems(DOSE_UNITS)
        self.dose_unit_combo.setToolTip("Unit the dose is entered in. Labels results — doesn't convert.")
        unit_row.addWidget(self.dose_unit_combo)
        unit_row.addStretch()
        layout.addLayout(unit_row)

        fill_row = QHBoxLayout()
        fill_row.addWidget(QLabel("Route (applies to whole table):"))
        self.route_combo = QComboBox()
        self.route_combo.addItems([r.value for r in Route])
        self.route_combo.setToolTip("Dosing route for every row — sets the hidden 'route' column.")
        fill_row.addWidget(self.route_combo)

        fill_row.addSpacing(20)
        fill_row.addWidget(QLabel("Dose (applies to whole table):"))
        self.dose_input = QLineEdit()
        self.dose_input.setPlaceholderText("e.g. 500")
        self.dose_input.setMaximumWidth(100)
        self.dose_input.setToolTip("Dose for every row — sets the hidden 'dose' column.")
        fill_row.addWidget(self.dose_input)
        fill_row.addStretch()
        layout.addLayout(fill_row)

        self.grid = DataGrid()
        self.grid.setToolTip("Concentration-time data. Type directly, or paste from Excel with Ctrl+V.")
        add_row_btn.clicked.connect(self.grid.add_row)
        del_row_btn.clicked.connect(self.grid.delete_selected_rows)

        grid_row = QHBoxLayout()
        grid_row.addWidget(self.grid)
        acquire_btn = QPushButton("Acquire Data")
        acquire_btn.setToolTip("Validate the grid and load it as the active dataset for analysis.")
        acquire_btn.clicked.connect(self._use_grid_data)
        grid_row.addWidget(acquire_btn, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addLayout(grid_row)

        # Route/dose are set once above, not per row — hide those grid
        # columns and keep them in sync automatically as the fields change.
        self.grid.hide_column("route")
        self.grid.hide_column("dose")
        self.route_combo.currentTextChanged.connect(
            lambda text: self.grid.fill_column("route", text)
        )
        self.route_combo.currentTextChanged.connect(lambda _: self._update_infusion_column_visibility())
        self.dose_input.textChanged.connect(self._fill_dose_column)
        self.grid.fill_column("route", self.route_combo.currentText())

        self.data_status = QLabel("No dataset loaded.")
        layout.addWidget(self.data_status)
        return widget

    def _fill_dose_column(self, text: str) -> None:
        text = text.strip()
        try:
            float(text)
        except ValueError:
            return  # not a valid number yet (e.g. mid-typing "5.") — wait for more input
        self.grid.fill_column("dose", text)

    def _import_file(self) -> None:
        wizard = ImportWizard(
            self,
            default_route=self.route_combo.currentText(),
            default_dose=self.dose_input.text().strip(),
        )
        if wizard.exec():
            # Imported data may still be missing dose/route/subject_id —
            # load it into the visible grid for the user to fix up, don't
            # force full Dataset validation until they click "Use Grid Data".
            self.grid.load_dataframe(wizard.dataframe)
            self.dataset = None
            self.data_status.setText(
                f"Imported {len(wizard.dataframe)} rows into the grid — review/edit, then click 'Use Grid Data'."
            )

    def _use_grid_data(self) -> None:
        try:
            ds = Dataset(self.grid.to_dataframe())
        except ValidationError as exc:
            QMessageBox.critical(self, "Validation error", "\n".join(exc.problems))
            return
        self._set_dataset(ds)

    def _set_dataset(self, ds: Dataset) -> None:
        self.dataset = ds
        self.grid.load_dataframe(ds.data)
        self.data_status.setText(f"Loaded {len(ds.data)} rows, {len(ds.subject_ids())} subject(s).")
        self._refresh_subject_combo(self.nca_subject_combo)
        self._refresh_subject_combo(self.comp_subject_combo)

    def _refresh_subject_combo(self, combo: QComboBox) -> None:
        combo.clear()
        if self.dataset:
            combo.addItems([str(s) for s in self.dataset.subject_ids()])

    # ---------- NCA tab ----------
    def _build_nca_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        row = QHBoxLayout()
        row.addWidget(QLabel("Subject:"))
        self.nca_subject_combo = QComboBox()
        self.nca_subject_combo.setMaxVisibleItems(15)
        self.nca_subject_combo.setToolTip("Subject to analyze — populated after data is loaded.")
        row.addWidget(self.nca_subject_combo)

        auc_method_label = QLabel("AUC method:")
        row.addWidget(auc_method_label)
        self.nca_auc_method = QComboBox()
        self.nca_auc_method.addItems(["linear", "linear_up_log_down", "linear_log", "linear_interp"])
        self.nca_auc_method.setToolTip(
            "Trapezoidal rule for AUC:\n"
            "linear — linear trapezoid throughout.\n"
            "linear_up_log_down — linear while rising, log while declining (standard default).\n"
            "linear_log — log trapezoid whenever concentration is positive, both directions.\n"
            "linear_interp — linear trapezoid with interpolated partial-area bounds."
        )
        row.addWidget(self.nca_auc_method)

        lz_mode_label = QLabel("λz mode:")
        row.addWidget(lz_mode_label)
        self.nca_lz_mode = QComboBox()
        self.nca_lz_mode.addItems(["Best Fit", "Time Range"])
        self.nca_lz_mode.setToolTip(
            "Best Fit — auto-selects terminal points by best adjusted R².\n"
            "Time Range — use the start,end window you specify instead."
        )
        row.addWidget(self.nca_lz_mode)

        lz_range_label = QLabel("λz start,end:")
        row.addWidget(lz_range_label)
        self.nca_lz_range = QLineEdit()
        self.nca_lz_range.setPlaceholderText("e.g. 4,24")
        self.nca_lz_range.setMaximumWidth(90)
        self.nca_lz_range.setToolTip("Terminal-phase time window for λz, used only when mode is 'Time Range'.")
        row.addWidget(self.nca_lz_range)

        lz_exclude_label = QLabel("Exclude times:")
        row.addWidget(lz_exclude_label)
        self.nca_lz_exclude = QLineEdit()
        self.nca_lz_exclude.setPlaceholderText("e.g. 8,12")
        self.nca_lz_exclude.setMaximumWidth(90)
        self.nca_lz_exclude.setToolTip(
            "Time points to drop from the λz regression only — they still count in AUC."
        )
        row.addWidget(self.nca_lz_exclude)

        self.nca_run_btn = QPushButton("Run NCA")
        self.nca_run_btn.setToolTip("Compute NCA parameters for the selected subject.")
        self.nca_run_btn.clicked.connect(self._run_nca)
        row.addWidget(self.nca_run_btn)
        row.addStretch()
        layout.addLayout(row)

        ss_row = QHBoxLayout()
        self.nca_steady_state = QCheckBox("Steady state (Tau)")
        self.nca_steady_state.setToolTip(
            "Report steady-state parameters (Cmax_ss, Cmin_ss, Ctau, Cavg_ss,\n"
            "%Fluctuation, Accumulation) over one dosing interval instead of single-dose Cmax/Tmax."
        )
        ss_row.addWidget(self.nca_steady_state)
        tau_label = QLabel("Tau:")
        ss_row.addWidget(tau_label)
        self.nca_tau = QLineEdit()
        self.nca_tau.setPlaceholderText("e.g. 12")
        self.nca_tau.setMaximumWidth(90)
        self.nca_tau.setToolTip("Dosing interval (Tau) — the [0, Tau] window steady-state params are computed over.")
        ss_row.addWidget(self.nca_tau)
        ss_row.addStretch()

        # Automatic mode hides every advanced knob — the underlying nca.py
        # functions already default to the same safe behavior these
        # controls would otherwise let a user override.
        self._advanced_nca_widgets = [
            auc_method_label, self.nca_auc_method,
            lz_mode_label, self.nca_lz_mode,
            lz_range_label, self.nca_lz_range,
            lz_exclude_label, self.nca_lz_exclude,
            self.nca_steady_state, tau_label, self.nca_tau,
        ]

        export_btn = QPushButton("Export Core Output...")
        export_btn.setToolTip("Save the last run's full settings, results, and warnings to a text file.")
        export_btn.clicked.connect(self._export_nca_core_output)
        ss_row.addWidget(export_btn)
        layout.addLayout(ss_row)

        splitter = QSplitter()
        self.nca_results = ResultsTable()
        self.nca_plot = ConcTimePlot()
        splitter.addWidget(self.nca_results)
        splitter.addWidget(self.nca_plot)
        layout.addWidget(splitter)
        return widget

    def _run_nca(self) -> None:
        if not self.dataset or not self.nca_subject_combo.currentText():
            QMessageBox.warning(self, "No data", "Load data and pick a subject first.")
            return
        if self._nca_task is not None:
            return  # already running — button is disabled, but guard re-entrancy anyway
        subject = self.nca_subject_combo.currentText()
        df = self.dataset.subject_data(_coerce(subject, self.dataset))
        t = df["time"].to_numpy(dtype=float)
        c = df["concentration"].to_numpy(dtype=float)
        dose = float(df["dose"].iloc[0])
        route = str(df["route"].iloc[0])

        manual = self.mode_toggle.isChecked()

        weight_val = df["weight"].iloc[0] if "weight" in df.columns else None
        weight = float(weight_val) if weight_val is not None and not pd.isna(weight_val) else None

        if not manual:
            # Automatic mode: fixed safe defaults, matching PLANNING.md's
            # standard recommendation — never read the (hidden) advanced
            # widgets, so a stale value the user never touched can't leak in.
            auc_method = "linear_up_log_down"
            lz_t_range = None
            lz_excluded_times = None
            tau = None
        else:
            auc_method = self.nca_auc_method.currentText()

            lz_t_range = None
            if self.nca_lz_mode.currentText() == "Time Range":
                parts = [p.strip() for p in self.nca_lz_range.text().split(",") if p.strip()]
                if len(parts) != 2:
                    QMessageBox.warning(self, "Invalid λz range", "Enter start,end (e.g. 4,24).")
                    return
                try:
                    lz_t_range = (float(parts[0]), float(parts[1]))
                except ValueError:
                    QMessageBox.warning(self, "Invalid λz range", "start,end must be numbers.")
                    return

            lz_excluded_times = None
            if self.nca_lz_exclude.text().strip():
                try:
                    lz_excluded_times = {float(x) for x in self.nca_lz_exclude.text().split(",") if x.strip()}
                except ValueError:
                    QMessageBox.warning(self, "Invalid exclusions", "Excluded times must be comma-separated numbers.")
                    return

            tau = None
            if self.nca_steady_state.isChecked():
                try:
                    tau = float(self.nca_tau.text().strip())
                except ValueError:
                    QMessageBox.warning(self, "Invalid Tau", "Enter a numeric dosing interval (Tau).")
                    return

        route_fn = {
            "iv_bolus": partial(nca.nca_iv_bolus, auc_method=auc_method,
                                 lz_t_range=lz_t_range, lz_excluded_times=lz_excluded_times, tau=tau, weight=weight),
            "iv_infusion": lambda t, c, dose: nca.nca_iv_infusion(
                t, c, dose, float(df["infusion_duration"].iloc[0] or 0.0), auc_method=auc_method,
                lz_t_range=lz_t_range, lz_excluded_times=lz_excluded_times, tau=tau, weight=weight,
            ),
            "extravascular": partial(nca.nca_extravascular, auc_method=auc_method,
                                      lz_t_range=lz_t_range, lz_excluded_times=lz_excluded_times, tau=tau, weight=weight),
        }.get(route)
        if route_fn is None:
            QMessageBox.critical(self, "Unknown route", f"route: {route}")
            return

        settings = {
            "mode": "Manual" if manual else "Automatic",
            "subject": subject,
            "route": route,
            "dose": dose,
            "weight": weight,
            "auc_method": auc_method,
            "lz_mode": self.nca_lz_mode.currentText() if manual else "Best Fit",
            "lz_t_range": lz_t_range,
            "lz_excluded_times": sorted(lz_excluded_times) if lz_excluded_times else None,
            "tau": tau,
            "time_unit": self.time_unit_combo.currentText(),
            "conc_unit": self.conc_unit_combo.currentText(),
            "dose_unit": self.dose_unit_combo.currentText(),
        }

        # Pending context read back by _on_nca_done — NOT passed via a lambda
        # closure. Connecting a cross-thread signal to a lambda deadlocks:
        # a lambda has no QObject thread affinity, so Qt can't build a
        # proper queued-event proxy for it (true even with an explicit
        # QueuedConnection type). Connecting straight to a bound method of
        # `self` (a real QObject) lets Qt detect the GUI-thread affinity and
        # marshal the call correctly.
        self._nca_pending_t = t
        self._nca_pending_c = c
        self._nca_pending_settings = settings

        self.nca_run_btn.setEnabled(False)
        self.nca_run_btn.setText("Running...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._nca_task = BackgroundTask(route_fn, t, c, dose)
        self._nca_task.finished.connect(self._on_nca_done)
        self._nca_task.error.connect(self._on_nca_error)
        self._nca_task.start()

    def _on_nca_error(self, exc: Exception) -> None:
        self._nca_task = None
        self.nca_run_btn.setEnabled(True)
        self.nca_run_btn.setText("Run NCA")
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "NCA error", str(exc))

    def _on_nca_done(self, result: dict) -> None:
        t, c, settings = self._nca_pending_t, self._nca_pending_c, self._nca_pending_settings
        self._nca_task = None
        self.nca_run_btn.setEnabled(True)
        self.nca_run_btn.setText("Run NCA")
        QApplication.restoreOverrideCursor()
        units = nca_units(self.time_unit_combo.currentText(), self.conc_unit_combo.currentText(),
                           self.dose_unit_combo.currentText())
        self.nca_results.set_results(result, units)
        terminal_line = None
        if result.get("slope") is not None:
            terminal_line = {
                "slope": result["slope"],
                "intercept": result["intercept"],
                "t_range": (min(result["terminal_t"]), max(result["terminal_t"])),
                "r_squared": result["r_squared"],
            }
        self.nca_plot.plot_observed(t, c, terminal_line=terminal_line,
                                     time_unit=self.time_unit_combo.currentText(),
                                     conc_unit=self.conc_unit_combo.currentText())

        self._nca_last_result = result
        self._nca_last_settings = settings
        self._nca_last_units = units

    def _export_nca_core_output(self) -> None:
        if not getattr(self, "_nca_last_result", None):
            QMessageBox.warning(self, "No results", "Run NCA first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Core Output", "nca_core_output.txt", "Text files (*.txt)")
        if not path:
            return
        text = nca.format_core_output(self._nca_last_result, self._nca_last_settings, self._nca_last_units)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    # ---------- Compartmental tab ----------
    def _build_compartmental_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        row = QHBoxLayout()
        row.addWidget(QLabel("Subject:"))
        self.comp_subject_combo = QComboBox()
        self.comp_subject_combo.setMaxVisibleItems(15)
        row.addWidget(self.comp_subject_combo)
        row.addWidget(QLabel("Model:"))
        self.comp_model_combo = QComboBox()
        self.comp_model_combo.addItems(list(COMPARTMENTAL_CHOICES.keys()))
        row.addWidget(self.comp_model_combo)
        row.addWidget(QLabel("Weighting:"))
        self.comp_weight_combo = QComboBox()
        self.comp_weight_combo.addItems(["uniform", "inverse_y", "inverse_y2"])
        row.addWidget(self.comp_weight_combo)
        self.comp_run_btn = QPushButton("Fit")
        self.comp_run_btn.clicked.connect(self._run_fit)
        row.addWidget(self.comp_run_btn)
        row.addStretch()
        layout.addLayout(row)

        splitter = QSplitter()
        self.comp_results = ResultsTable()
        self.comp_plot = ConcTimePlot()
        splitter.addWidget(self.comp_results)
        splitter.addWidget(self.comp_plot)
        layout.addWidget(splitter)
        return widget

    def _run_fit(self) -> None:
        if not self.dataset or not self.comp_subject_combo.currentText():
            QMessageBox.warning(self, "No data", "Load data and pick a subject first.")
            return
        if self._comp_task is not None:
            return  # already running — button is disabled, but guard re-entrancy anyway
        subject = self.comp_subject_combo.currentText()
        df = self.dataset.subject_data(_coerce(subject, self.dataset))
        t = df["time"].to_numpy(dtype=float)
        c = df["concentration"].to_numpy(dtype=float)
        dose = float(df["dose"].iloc[0])
        tinf = float(df["infusion_duration"].iloc[0] or 0.0)

        n_comp, route = COMPARTMENTAL_CHOICES[self.comp_model_combo.currentText()]
        weight_scheme = self.comp_weight_combo.currentText()

        try:
            model, p0, bounds, names = _build_model_spec(n_comp, route, t, c, dose, tinf)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Setup error", str(exc))
            return

        # See _run_nca for why context goes on self instead of a lambda
        # closure — a lambda receiver deadlocks a cross-thread signal.
        self._comp_pending_t = t
        self._comp_pending_c = c
        self._comp_pending_model = model

        self.comp_run_btn.setEnabled(False)
        self.comp_run_btn.setText("Fitting...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._comp_task = BackgroundTask(
            fitting.fit_model, model, t, c, p0, bounds, weight_scheme, names
        )
        self._comp_task.finished.connect(self._on_fit_done)
        self._comp_task.error.connect(self._on_fit_error)
        self._comp_task.start()

    def _on_fit_error(self, exc: Exception) -> None:
        self._comp_task = None
        self.comp_run_btn.setEnabled(True)
        self.comp_run_btn.setText("Fit")
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self, "Fit error", str(exc))

    def _on_fit_done(self, result: dict) -> None:
        t, c, model = self._comp_pending_t, self._comp_pending_c, self._comp_pending_model
        self._comp_task = None
        self.comp_run_btn.setEnabled(True)
        self.comp_run_btn.setText("Fit")
        QApplication.restoreOverrideCursor()
        units = compartmental_units(self.time_unit_combo.currentText(), self.conc_unit_combo.currentText(),
                                     self.dose_unit_combo.currentText())
        self.comp_results.set_results(result, units)
        t_fine = np.linspace(t.min(), t.max(), 200)
        params = list(result["params"].values())
        c_fitted = model(t_fine, *params)
        self.comp_plot.plot_observed(t, c, fitted_t=t_fine, fitted_c=c_fitted,
                                      time_unit=self.time_unit_combo.currentText(),
                                      conc_unit=self.conc_unit_combo.currentText())


def _coerce(value: str, dataset: Dataset):
    """Subject IDs may be int or str depending on source data; match the
    combo box's stringified value back to the dataset's actual dtype."""
    for sid in dataset.subject_ids():
        if str(sid) == value:
            return sid
    return value


def _build_model_spec(n_comp: str, route: str, t: np.ndarray, c: np.ndarray, dose: float, tinf: float):
    """Auto initial guesses from the data itself (terminal slope for k,
    dose/Cmax for V) so the user isn't forced to hand-type them."""
    lz = nca.lambda_z(t, c)
    k_guess = lz["lambda_z"] or 0.1
    c0_guess = float(c.max()) if c.max() > 0 else 1.0
    v_guess = dose / c0_guess

    if n_comp == "1c" and route == "iv_bolus":
        return (
            partial(models.conc_1c_iv_bolus, dose=dose),
            [k_guess, v_guess],
            ([1e-8, 1e-8], [10, v_guess * 100]),
            ["k", "V"],
        )
    if n_comp == "1c" and route == "iv_bolus_cl":
        cl_guess = k_guess * v_guess
        return (
            partial(models.conc_1c_iv_bolus_cl, dose=dose),
            [cl_guess, v_guess],
            ([1e-8, 1e-8], [10 * v_guess, v_guess * 100]),
            ["Cl", "V"],
        )
    if n_comp == "1c" and route == "iv_infusion":
        if tinf <= 0:
            raise ValueError("infusion_duration must be set (> 0) for IV infusion route")
        return (
            partial(models.conc_1c_iv_infusion, dose=dose, tinf=tinf),
            [k_guess, v_guess],
            ([1e-8, 1e-8], [10, v_guess * 100]),
            ["k", "V"],
        )
    if n_comp == "1c" and route == "extravascular":
        return (
            partial(models.conc_1c_extravascular, dose=dose),
            [k_guess * 5, k_guess, v_guess],
            ([1e-8, 1e-8, 1e-8], [50, 10, v_guess * 100]),
            ["ka", "k", "V"],
        )
    if n_comp == "2c" and route == "iv_bolus":
        return (
            models.conc_2c_iv_bolus,
            [c0_guess * 0.7, k_guess * 5, c0_guess * 0.3, k_guess],
            ([1e-8, 1e-8, 1e-8, 1e-8], [c0_guess * 10, 50, c0_guess * 10, 10]),
            ["A", "alpha", "B", "beta"],
        )
    raise ValueError(f"unsupported model/route: {n_comp}/{route}")
