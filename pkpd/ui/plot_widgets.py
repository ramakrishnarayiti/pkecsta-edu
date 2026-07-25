"""Matplotlib canvas embedded in Qt. One reusable conc-time plot widget,
used by both the NCA tab and the compartmental-fit tab. Includes a
linear/semi-log toggle button — most PK plots are read in semi-log to see
the terminal slope, but linear is useful for spotting absorption-phase
shape, so both stay one click away instead of picking one and hiding the other."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class ConcTimePlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_scale = True
        self._last_plot: dict | None = None

        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.scale_toggle = QPushButton()
        self.scale_toggle.setCheckable(True)
        self.scale_toggle.setChecked(True)
        self.scale_toggle.toggled.connect(self._on_toggle)
        self._update_toggle_label()

        toggle_row = QHBoxLayout()
        toggle_row.addStretch()
        toggle_row.addWidget(self.scale_toggle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toggle_row)
        layout.addWidget(self.canvas)

    def _update_toggle_label(self) -> None:
        self.scale_toggle.setText("Semi-log (click for Linear)" if self._log_scale
                                   else "Linear (click for Semi-log)")

    def _on_toggle(self, checked: bool) -> None:
        self._log_scale = checked
        self._update_toggle_label()
        if self._last_plot is not None:
            self._draw(**self._last_plot)

    def plot_observed(self, t: np.ndarray, c: np.ndarray, log_scale: bool | None = None,
                       fitted_t: np.ndarray | None = None, fitted_c: np.ndarray | None = None,
                       terminal_line: dict | None = None, time_unit: str = "", conc_unit: str = "") -> None:
        """terminal_line: optional {"slope", "intercept", "t_range": (t0, t1),
        "r_squared"} — draws the lambda_z log-linear regression line used
        for the terminal phase, with R^2 annotated.

        log_scale sets the initial scale for this dataset (defaults to the
        current toggle state if omitted); the on-screen button can still
        switch it afterwards without needing new data.
        """
        if log_scale is not None and log_scale != self._log_scale:
            self._log_scale = log_scale
            self.scale_toggle.blockSignals(True)
            self.scale_toggle.setChecked(log_scale)
            self.scale_toggle.blockSignals(False)
            self._update_toggle_label()

        self._last_plot = dict(t=t, c=c, fitted_t=fitted_t, fitted_c=fitted_c,
                                terminal_line=terminal_line, time_unit=time_unit, conc_unit=conc_unit)
        self._draw(**self._last_plot)

    def _draw(self, t: np.ndarray, c: np.ndarray, fitted_t: np.ndarray | None, fitted_c: np.ndarray | None,
               terminal_line: dict | None, time_unit: str, conc_unit: str) -> None:
        self.ax.clear()
        self.ax.plot(t, c, "o", color="#20613e", label="observed")
        if fitted_t is not None and fitted_c is not None:
            self.ax.plot(fitted_t, fitted_c, "-", color="#2F8F5B", label="fitted")
        if terminal_line is not None:
            t0, t1 = terminal_line["t_range"]
            line_t = np.linspace(t0, t1, 50)
            line_c = np.exp(terminal_line["slope"] * line_t + terminal_line["intercept"])
            self.ax.plot(line_t, line_c, "--", color="#B45309", linewidth=1.5,
                         label=f"terminal fit (R²={terminal_line['r_squared']:.4f})")
        self.ax.set_yscale("log" if self._log_scale else "linear")
        self.ax.set_xlabel(f"Time ({time_unit})" if time_unit else "Time")
        self.ax.set_ylabel(f"Concentration ({conc_unit})" if conc_unit else "Concentration")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_residuals(self, t: np.ndarray, residuals: np.ndarray) -> None:
        self.ax.clear()
        self.ax.axhline(0, color="gray", linewidth=0.8)
        self.ax.plot(t, residuals, "o", color="#20613e")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Residual")
        self.fig.tight_layout()
        self.canvas.draw()
