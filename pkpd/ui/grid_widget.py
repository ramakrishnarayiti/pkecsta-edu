"""Manual data-entry grid: spreadsheet-like table bound to the schema
columns, with add/delete row and paste-from-clipboard (the realistic manual
workflow — researchers paste from Excel, not type cell by cell)."""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget

from pkpd.core.data_model import ALL_COLUMNS


class DataGrid(QTableWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, len(ALL_COLUMNS), parent)
        self.setHorizontalHeaderLabels(ALL_COLUMNS)
        self._defaults: dict[str, str] = {}
        self.add_row()

    def add_row(self) -> None:
        row = self.rowCount()
        self.insertRow(row)
        for col, name in enumerate(ALL_COLUMNS):
            self.setItem(row, col, QTableWidgetItem(self._defaults.get(name, "")))

    def delete_selected_rows(self) -> None:
        rows = sorted({idx.row() for idx in self.selectedIndexes()}, reverse=True)
        for row in rows:
            self.removeRow(row)

    def set_column_default(self, column_name: str, value: str) -> None:
        """Remember this value so future add_row() calls pre-fill the
        column (e.g. route/dose is usually the same for every row)."""
        self._defaults[column_name] = value

    def hide_column(self, column_name: str) -> None:
        """Hide a column that's set globally elsewhere in the UI (route,
        dose) instead of per-row — the cell still holds the value via
        fill_column, it's just not shown to avoid per-row re-entry."""
        if column_name in ALL_COLUMNS:
            self.setColumnHidden(ALL_COLUMNS.index(column_name), True)

    def show_column(self, column_name: str) -> None:
        """Reverse of hide_column — used when switching modes/routes makes
        a previously-hidden column relevant again."""
        if column_name in ALL_COLUMNS:
            self.setColumnHidden(ALL_COLUMNS.index(column_name), False)

    def fill_column(self, column_name: str, value: str) -> None:
        """Set every existing row's cell in this column to `value` — used
        when the user picks route/dose once instead of typing it per row."""
        if column_name not in ALL_COLUMNS:
            return
        col = ALL_COLUMNS.index(column_name)
        for row in range(self.rowCount()):
            self.setItem(row, col, QTableWidgetItem(value))
        self.set_column_default(column_name, value)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_V and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._paste_clipboard()
            return
        super().keyPressEvent(event)

    def _paste_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication

        text = QApplication.clipboard().text()
        if not text:
            return
        # With no cell selected both return -1, and setItem(-1, ...) is a
        # silent no-op in Qt — the paste would vanish with no error. Land at
        # the top-left instead.
        start = max(self.currentRow(), 0), max(self.currentColumn(), 0)
        for r, line in enumerate(text.strip("\n").split("\n")):
            row = start[0] + r
            while row >= self.rowCount():
                self.add_row()
            for c, value in enumerate(line.split("\t")):
                col = start[1] + c
                if col < self.columnCount():
                    self.setItem(row, col, QTableWidgetItem(value.strip()))

    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Replace grid contents with an externally loaded dataset (e.g.
        from file import) so the user sees the actual data on screen
        instead of it only living in memory."""
        self.setRowCount(0)
        for _, record in df.iterrows():
            row = self.rowCount()
            self.insertRow(row)
            for col, name in enumerate(ALL_COLUMNS):
                value = record.get(name, "")
                text = "" if pd.isna(value) else str(value)
                self.setItem(row, col, QTableWidgetItem(text))
        if self.rowCount() == 0:
            self.add_row()

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for row in range(self.rowCount()):
            record = {}
            for col, name in enumerate(ALL_COLUMNS):
                item = self.item(row, col)
                record[name] = item.text().strip() if item else ""
            # a row counts as "used" only if it has actual observation data —
            # route/dose are pre-filled defaults and shouldn't count
            if record["time"] or record["concentration"]:
                records.append(record)
        df = pd.DataFrame.from_records(records, columns=ALL_COLUMNS)
        for col in ("time", "concentration", "dose", "infusion_duration", "weight"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
