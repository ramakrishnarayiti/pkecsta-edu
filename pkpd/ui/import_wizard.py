"""File import dialog: pick a CSV/Excel file, map its columns to the
internal schema, hand back a raw DataFrame for the grid to show. Only
time/concentration are truly required from the file — subject_id/route/dose
get a default and stay editable in the grid rather than blocking import."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from pkpd.core.data_model import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, ALL_COLUMNS
from pkpd.core.io_import import fill_missing_columns, read_raw

#  Only these truly can't be defaulted — everything else falls back to a
#  Data-tab value or a placeholder and stays editable in the grid.
FILE_REQUIRED_COLUMNS = ("time", "concentration")
#  route/dose/subject_id can be set once in the Data tab (or left blank and
#  fixed up in the grid) instead of living in the file.
FALLBACK_COLUMNS = ("route", "dose", "subject_id")


class ImportWizard(QDialog):
    def __init__(self, parent: QWidget | None = None, default_route: str = "", default_dose: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import data file")
        self.dataframe: pd.DataFrame | None = None
        self._raw = None
        self._default_route = default_route
        self._default_dose = default_dose

        layout = QVBoxLayout(self)
        self._file_label = QLabel("No file selected")
        layout.addWidget(self._file_label)

        pick_btn = QDialogButtonBox()
        pick_btn.addButton("Choose File...", QDialogButtonBox.ButtonRole.ActionRole)
        pick_btn.clicked.connect(self._choose_file)
        layout.addWidget(pick_btn)

        note = QLabel(
            f"Only time/concentration need mapping. subject_id/route/dose not "
            f"in your file? Leave unmapped — Data tab values "
            f"(route={default_route or '?'}, dose={default_dose or '?'}) fill every "
            f"row, or fix them up in the grid after import."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._mapping_form = QFormLayout()
        layout.addLayout(self._mapping_form)
        self._combos: dict[str, QComboBox] = {}
        for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
            combo = QComboBox()
            combo.setEnabled(False)
            self._combos[col] = combo
            if col in FALLBACK_COLUMNS:
                label = col + " (optional — falls back to Data tab value)"
            else:
                label = col + (" *" if col in REQUIRED_COLUMNS else " (optional)")
            self._mapping_form.addRow(label, combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select data file", "", "Data files (*.csv *.xlsx *.xls)")
        if not path:
            return
        try:
            self._raw = read_raw(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import error", str(exc))
            return
        self._file_label.setText(path)
        for col, combo in self._combos.items():
            combo.clear()
            combo.addItem("")
            combo.addItems(list(self._raw.columns))
            combo.setEnabled(True)
            guess = next((c for c in self._raw.columns if c.strip().lower() == col.lower()), None)
            if guess:
                combo.setCurrentText(guess)

    def _accept(self) -> None:
        if self._raw is None:
            QMessageBox.warning(self, "No file", "Choose a file first.")
            return
        mapping = {col: combo.currentText() for col, combo in self._combos.items() if combo.currentText()}

        missing = [c for c in FILE_REQUIRED_COLUMNS if c not in mapping]
        if missing:
            QMessageBox.warning(self, "Incomplete mapping", f"Map required columns: {missing}")
            return

        mapped = self._raw.rename(columns={v: k for k, v in mapping.items()})[list(mapping.keys())]

        # route/dose/subject_id: use Data-tab default if set, otherwise
        # leave blank — the grid stays editable, import never blocks on them.
        defaults = {}
        if "route" not in mapping and self._default_route:
            defaults["route"] = self._default_route
        if "dose" not in mapping and self._default_dose:
            defaults["dose"] = self._default_dose
        if "subject_id" not in mapping:
            defaults["subject_id"] = "1"
        mapped = fill_missing_columns(mapped, defaults)
        for col in ALL_COLUMNS:
            if col not in mapped.columns:
                mapped[col] = ""
        # numeric columns can arrive as strings either from the file itself
        # or from a Data-tab default (e.g. dose="500") — coerce explicitly,
        # same as the manual grid does.
        for col in ("time", "concentration", "dose", "infusion_duration", "weight"):
            mapped[col] = pd.to_numeric(mapped[col], errors="coerce")

        self.dataframe = mapped[ALL_COLUMNS]
        self.accept()
