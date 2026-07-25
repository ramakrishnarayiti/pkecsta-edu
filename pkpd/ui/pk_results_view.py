"""Read-only results table — reused for both NCA and compartmental output.
Takes a flat dict (param name -> value) and renders it as
Parameter/Full Name/Value/Unit."""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

# The app's one accent green (pkpd/ui/theme.py) — every tint below is mixed
# from this single color toward white, never a separately chosen hex, so the
# whole app stays one green rather than a handful of similar-but-different
# ones.
_ACCENT = QColor("#2F8F5B")
_WHITE = QColor("#ffffff")


def _lerp_color(top: QColor, bottom: QColor, t: float) -> QColor:
    return QColor(
        round(top.red() + (bottom.red() - top.red()) * t),
        round(top.green() + (bottom.green() - top.green()) * t),
        round(top.blue() + (bottom.blue() - top.blue()) * t),
    )


def _tint(fraction: float) -> QColor:
    """Accent mixed toward white — fraction 0 is white, 1 is full accent."""
    return _lerp_color(_WHITE, _ACCENT, fraction)


# Tint gradient endpoints for the Parameter/Value columns, top row to bottom
# row — a flat accent would read as a random highlight; the gradient reads
# as deliberate column shading instead.
_PARAM_TINT_TOP = _tint(0.12)
_PARAM_TINT_BOTTOM = _tint(0.30)
_VALUE_TINT_TOP = _tint(0.06)
_VALUE_TINT_BOTTOM = _tint(0.20)

# key -> spelled-out parameter name, for both NCA and compartmental result
# dicts. Dict-valued results (params/param_se/param_cv_pct) are looked up by
# their sub_key (e.g. "k", "V"), not the "params.k" composite.
FULL_NAMES = {
    # NCA
    "cmax": "Maximum observed concentration",
    "tmax": "Time of maximum observed concentration",
    "clast": "Last observed concentration",
    "tlast": "Time of last observed concentration",
    "lambda_z": "Terminal elimination rate constant",
    "half_life": "Terminal half-life",
    "n_points": "Terminal points used in λz regression",
    "r_squared": "R-squared",
    "adj_r_squared": "Adjusted R-squared",
    "slope": "λz regression slope",
    "intercept": "λz regression intercept",
    "auc_t": "AUC from 0 to last observed time",
    "auc_inf": "AUC extrapolated to infinity (observed Clast)",
    "auc_inf_pred": "AUC extrapolated to infinity (predicted Clast)",
    "pct_extrap": "Percent of AUCinf that is extrapolated",
    "pct_back_ext": "Percent of AUCinf from back-extrapolation",
    "aumc_t": "Area under the first-moment curve to last observed time",
    "aumc_inf": "Area under the first-moment curve extrapolated to infinity",
    "cl": "Total clearance",
    "vz": "Volume of distribution (terminal phase)",
    "vss": "Volume of distribution at steady state",
    "mrt": "Mean residence time",
    "dose": "Dose administered",
    "dose_per_kg": "Dose normalized to body weight",
    "cl_per_kg": "Clearance normalized to body weight",
    "vz_per_kg": "Terminal volume of distribution normalized to body weight",
    "vss_per_kg": "Volume of distribution at steady state normalized to body weight",
    "span": "λz window span, in terminal half-lives",
    "route": "Dosing route",
    "n_excluded": "Samples excluded as missing/BQL",
    "note": "Note",
    "flag_n_samples": "Sufficient-samples flag",
    "flag_low_rsq": "Low terminal-phase R-squared flag",
    "flag_high_extrap": "High AUC-extrapolation flag",
    "flag_low_span": "Low λz-span flag",
    "flag_tau_beyond_tlast": "Tau-beyond-last-sample flag",
    "tau": "Dosing interval (Tau)",
    "cmax_ss": "Maximum concentration at steady state",
    "tmax_ss": "Time of maximum concentration at steady state",
    "cmin_ss": "Minimum (trough) concentration at steady state",
    "ctau": "Concentration at end of dosing interval",
    "auc_tau": "AUC over one dosing interval",
    "cavg_ss": "Average concentration at steady state",
    "pct_fluctuation": "Percent fluctuation over the dosing interval",
    "accumulation_index": "Accumulation index",
    # Compartmental
    "k": "Elimination rate constant",
    "ka": "Absorption rate constant",
    "tlag": "Absorption lag time",
    "alpha": "Distribution-phase hybrid rate constant",
    "beta": "Elimination-phase hybrid rate constant",
    "V": "Volume of distribution",
    "V1": "Central compartment volume",
    "Cl": "Total clearance",
    "A": "Distribution-phase hybrid coefficient",
    "B": "Elimination-phase hybrid coefficient",
    "aic": "Akaike information criterion",
    "aicc": "Corrected Akaike information criterion (small-sample)",
    "fit_scale": "Scale the fit was performed on",
    "params_at_bounds": "Parameters pinned at a fit bound",
}


class ResultsTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["Parameter", "Value", "Unit", "Full Name"])
        # Sane starting widths for the fixed-content columns, but every
        # column stays user-draggable (Interactive, the default) — Stretch
        # filled the window but made columns fixed, so a user could no
        # longer pull one wider. Full Name is last, so it auto-stretches to
        # absorb whatever space is left over instead of a dead gray gap.
        header = self.horizontalHeader()
        self.setColumnWidth(0, 110)
        self.setColumnWidth(1, 110)
        self.setColumnWidth(2, 90)
        header.setStretchLastSection(True)

    def set_results(self, results: dict, units: dict[str, str] | None = None) -> None:
        units = units or {}
        self.setRowCount(0)
        rows = []
        for key, value in results.items():
            if key in ("residuals", "predicted", "covariance", "terminal_t", "terminal_conc"):
                continue
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    rows.append((f"{key}.{sub_key}", sub_key, sub_value, units.get(sub_key, "")))
            else:
                rows.append((key, key, value, units.get(key, "")))

        n = len(rows)
        for i, (key, name_key, value, unit) in enumerate(rows):
            t = i / (n - 1) if n > 1 else 0.0
            self._add_row(key, name_key, value, unit, t)

    def _add_row(self, key: str, name_key: str, value, unit: str, t: float) -> None:
        row = self.rowCount()
        self.insertRow(row)
        param_item = QTableWidgetItem(str(key))
        param_item.setBackground(_lerp_color(_PARAM_TINT_TOP, _PARAM_TINT_BOTTOM, t))
        self.setItem(row, 0, param_item)
        display = f"{value:.6g}" if isinstance(value, float) else str(value)
        value_item = QTableWidgetItem(display)
        value_item.setBackground(_lerp_color(_VALUE_TINT_TOP, _VALUE_TINT_BOTTOM, t))
        self.setItem(row, 1, value_item)
        self.setItem(row, 2, QTableWidgetItem(unit))
        self.setItem(row, 3, QTableWidgetItem(FULL_NAMES.get(name_key, "")))
