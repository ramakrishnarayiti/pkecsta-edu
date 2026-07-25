"""Read-only results table — reused for both NCA and compartmental output.
Takes a flat dict (param name -> value) and renders it as Parameter/Value/Unit."""
from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class ResultsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])

    def set_results(self, results: dict, units: dict[str, str] | None = None) -> None:
        units = units or {}
        self.setRowCount(0)
        for key, value in results.items():
            if key in ("residuals", "predicted", "covariance", "terminal_t", "terminal_conc"):
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self._add_row(f"{key}.{sub_key}", sub_value, units.get(sub_key, ""))
            else:
                self._add_row(key, value, units.get(key, ""))

    def _add_row(self, key: str, value, unit: str) -> None:
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(str(key)))
        display = f"{value:.6g}" if isinstance(value, float) else str(value)
        self.setItem(row, 1, QTableWidgetItem(display))
        self.setItem(row, 2, QTableWidgetItem(unit))
