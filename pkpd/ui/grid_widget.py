"""Manual data-entry grid: spreadsheet-like table bound to the schema
columns, with add/delete row and paste-from-clipboard (the realistic manual
workflow — researchers paste from Excel, not type cell by cell).

A QTableView over a model rather than a QTableWidget. QTableWidget allocates
a QTableWidgetItem per cell, so loading a 10k-row file built 70,000 widget
objects on the UI thread and froze the window for seconds. The model holds
plain strings and the view only builds delegates for the rows actually on
screen, so load time stops depending on how much data there is.
"""
from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtWidgets import QTableView, QWidget

from pkpd.core.data_model import ALL_COLUMNS


class GridModel(QAbstractTableModel):
    """Rows of plain strings, one per schema column. Editing writes straight
    into the list — no per-cell widgets anywhere."""

    def __init__(self, columns: list[str], parent=None):
        super().__init__(parent)
        self._columns = columns
        self._rows: list[list[str]] = []

    # -- Qt interface ----------------------------------------------------
    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt override
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802 - Qt override
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return self._rows[index.row()][index.column()]
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        self._rows[index.row()][index.column()] = str(value).strip()
        self.dataChanged.emit(index, index)
        return True

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section]
        return str(section + 1)

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable)

    # -- bulk operations -------------------------------------------------
    def append_row(self, values: list[str]) -> None:
        row = len(self._rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append(list(values))
        self.endInsertRows()

    def remove_rows(self, rows: list[int]) -> None:
        for row in sorted(set(rows), reverse=True):
            if 0 <= row < len(self._rows):
                self.beginRemoveRows(QModelIndex(), row, row)
                self._rows.pop(row)
                self.endRemoveRows()

    def replace_all(self, rows: list[list[str]]) -> None:
        """One reset for the whole table — the reason a large file loads
        quickly instead of emitting a signal per cell."""
        self.beginResetModel()
        self._rows = rows
        self.endResetModel()

    def fill_column(self, column: int, value: str) -> None:
        if not self._rows:
            return
        for row in self._rows:
            row[column] = value
        self.dataChanged.emit(self.index(0, column),
                               self.index(len(self._rows) - 1, column))

    def value(self, row: int, column: int) -> str:
        return self._rows[row][column]

    def rows(self) -> list[list[str]]:
        return self._rows


class DataGrid(QTableView):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._model = GridModel(list(ALL_COLUMNS), self)
        self.setModel(self._model)
        self._defaults: dict[str, str] = {}
        self.add_row()
        # Default column width fits neither the uppercase header label nor
        # any data yet, so "CONCENTRATION" etc. clipped until the user
        # dragged every column wider by hand, every time the app opened.
        self.resizeColumnsToContents()

    # -- row/column operations -------------------------------------------
    def rowCount(self) -> int:  # noqa: N802 - mirrors the old QTableWidget API
        return self._model.rowCount()

    def add_row(self) -> None:
        self._model.append_row([self._defaults.get(name, "") for name in ALL_COLUMNS])

    def delete_selected_rows(self) -> None:
        self._model.remove_rows([idx.row() for idx in self.selectedIndexes()])

    def clear_rows(self) -> None:
        self._model.replace_all([])

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
        self._model.fill_column(ALL_COLUMNS.index(column_name), value)
        self.set_column_default(column_name, value)

    def cell(self, row: int, column: int) -> str:
        return self._model.value(row, column)

    def set_current_cell(self, row: int, column: int) -> None:
        self.setCurrentIndex(self._model.index(row, column))

    # -- clipboard -------------------------------------------------------
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
        # With no cell selected currentIndex() is invalid and both coordinates
        # come back as -1, which silently discarded the whole paste. Land at
        # the top-left instead.
        index = self.currentIndex()
        start_row = max(index.row(), 0)
        start_col = max(index.column(), 0)

        for r, line in enumerate(text.strip("\n").split("\n")):
            row = start_row + r
            while row >= self._model.rowCount():
                self.add_row()
            for c, value in enumerate(line.split("\t")):
                col = start_col + c
                if col < self._model.columnCount():
                    self._model.setData(self._model.index(row, col), value.strip())

    # -- dataframe round trip --------------------------------------------
    def load_dataframe(self, df: pd.DataFrame) -> None:
        """Replace grid contents with an externally loaded dataset (e.g.
        from file import) so the user sees the actual data on screen
        instead of it only living in memory."""
        # Reindexed once and read column-wise, so a large file costs one pass
        # rather than a per-cell lookup.
        frame = df.reindex(columns=ALL_COLUMNS)
        columns = [["" if pd.isna(v) else str(v) for v in frame[name]] for name in ALL_COLUMNS]
        rows = [list(values) for values in zip(*columns)] if len(frame) else []
        self._model.replace_all(rows)
        if self._model.rowCount() == 0:
            self.add_row()
        self.resizeColumnsToContents()

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for row in self._model.rows():
            record = dict(zip(ALL_COLUMNS, row))
            # a row counts as "used" only if it has actual observation data —
            # route/dose are pre-filled defaults and shouldn't count
            if record["time"] or record["concentration"]:
                records.append(record)
        df = pd.DataFrame.from_records(records, columns=ALL_COLUMNS)
        for col in ("time", "concentration", "dose", "infusion_duration", "weight"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
