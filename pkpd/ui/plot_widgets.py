"""Matplotlib canvas embedded in Qt. One reusable conc-time plot widget,
used by both the NCA tab and the compartmental-fit tab. Includes a
linear/semi-log toggle button — most PK plots are read in semi-log to see
the terminal slope, but linear is useful for spotting absorption-phase
shape, so both stay one click away instead of picking one and hiding the other."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Formats offered in the export dropdown -> (file extension, Qt file filter).
# 300 dpi + a white (non-transparent) face is the standard bar for a
# publication-ready raster/vector figure; vector PDF ignores dpi for the
# drawing itself but scipy/matplotlib still honors it for embedded raster
# elements (markers, if ever rasterized), so it's harmless to always pass.
EXPORT_FORMATS = {
    "PNG": ("png", "PNG Image (*.png)"),
    "PDF": ("pdf", "PDF Document (*.pdf)"),
    "JPEG": ("jpg", "JPEG Image (*.jpg *.jpeg)"),
}
EXPORT_DPI = 300


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

        self.export_format = QComboBox()
        self.export_format.addItems(list(EXPORT_FORMATS))
        self.export_format.setToolTip("File format for Export Plot.")
        export_btn = QPushButton("Export Plot...")
        export_btn.setToolTip(
            f"Save this figure at {EXPORT_DPI} dpi — publication quality, "
            "in the format selected to its left."
        )
        export_btn.clicked.connect(self._export)

        self._toggle_row = QHBoxLayout()
        self._toggle_row.addWidget(export_btn)
        self._toggle_row.addWidget(self.export_format)
        self._extra_insert_index = self._toggle_row.count()
        self._toggle_row.addStretch()
        self._toggle_row.addWidget(self.scale_toggle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._toggle_row)
        layout.addWidget(self.canvas)

    def add_toolbar_widget(self, widget) -> None:
        """Insert an extra button into this plot's own toolbar row, next to
        Export Plot — so a caller's related export action (e.g. NCA's Core
        Output text export) lives in the same place instead of a separate
        row elsewhere in the tab. Successive calls append left-to-right."""
        self._toggle_row.insertWidget(self._extra_insert_index, widget)
        self._extra_insert_index += 1

    def _export(self) -> None:
        fmt = self.export_format.currentText()
        ext, file_filter = EXPORT_FORMATS[fmt]
        path, _ = QFileDialog.getSaveFileName(self, "Export Plot", f"plot.{ext}", file_filter)
        if not path:
            return
        # format= is passed explicitly rather than inferred from the path's
        # extension — a user who types a bare filename (or a save dialog
        # that hands back something extension-less) must still get the
        # format they picked in the dropdown, not whatever Matplotlib guesses.
        self.fig.savefig(path, format=ext, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")

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
