"""Unit conversion + labeling for NCA/compartmental output. Real conversion
factors (not just labels) so results are meaningful regardless of which
unit the user picked — a dose entered in mcg and a result reported in mg
must actually convert, not just carry a label.

Weight is assumed kg (standard PK convention) — no weight-unit selector in
v1; add one if a user actually needs lb input."""
from __future__ import annotations

TIME_UNITS = ["hr", "mins", "day"]
CONC_UNITS = ["ng/mL", "mcg/mL", "mg/L", "mg/mL"]
DOSE_UNITS = ["mg", "mcg", "g"]

# Each table maps unit -> factor to convert 1 unit into the table's base
# unit (hr / ng/mL / mg respectively).
_TIME_TO_HR = {"hr": 1.0, "mins": 1.0 / 60.0, "day": 24.0}
_CONC_TO_NG_ML = {"ng/mL": 1.0, "mcg/mL": 1000.0, "mg/L": 1000.0, "mg/mL": 1_000_000.0}
_DOSE_TO_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}


def convert(value: float, from_unit: str, to_unit: str, table: dict[str, float]) -> float:
    """Convert value from from_unit to to_unit using a unit->base-factor
    table (one of the module-level _*_TO_* dicts)."""
    if from_unit == to_unit:
        return value
    return value * table[from_unit] / table[to_unit]


def convert_time(value: float, from_unit: str, to_unit: str) -> float:
    return convert(value, from_unit, to_unit, _TIME_TO_HR)


def convert_conc(value: float, from_unit: str, to_unit: str) -> float:
    return convert(value, from_unit, to_unit, _CONC_TO_NG_ML)


def convert_dose(value: float, from_unit: str, to_unit: str) -> float:
    return convert(value, from_unit, to_unit, _DOSE_TO_MG)


def nca_units(time_unit: str, conc_unit: str, dose_unit: str) -> dict[str, str]:
    ct = f"{conc_unit}*{time_unit}"
    return {
        "cmax": conc_unit,
        "tmax": time_unit,
        "lambda_z": f"1/{time_unit}",
        "half_life": time_unit,
        "clast": conc_unit,
        "tlast": time_unit,
        "auc_t": ct,
        "auc_inf": ct,
        "auc_inf_pred": ct,
        "pct_extrap": "%",
        "pct_back_ext": "%",
        "tau": time_unit,
        "cmax_ss": conc_unit,
        "tmax_ss": time_unit,
        "cmin_ss": conc_unit,
        "ctau": conc_unit,
        "auc_tau": ct,
        "cavg_ss": conc_unit,
        "pct_fluctuation": "%",
        "accumulation_index": "",
        "aumc_t": f"{conc_unit}*{time_unit}^2",
        "aumc_inf": f"{conc_unit}*{time_unit}^2",
        "cl": f"{dose_unit}/({ct})",
        "vz": f"{dose_unit}/{conc_unit}",
        "vss": f"{dose_unit}/{conc_unit}",
        "mrt": time_unit,
        "dose": dose_unit,
        "slope": f"1/{time_unit}",
        "dose_per_kg": f"{dose_unit}/kg",
        "cl_per_kg": f"{dose_unit}/({ct})/kg",
        "vz_per_kg": f"{dose_unit}/{conc_unit}/kg",
        "vss_per_kg": f"{dose_unit}/{conc_unit}/kg",
    }


def compartmental_units(time_unit: str, conc_unit: str, dose_unit: str) -> dict[str, str]:
    return {
        "k": f"1/{time_unit}",
        "ka": f"1/{time_unit}",
        "alpha": f"1/{time_unit}",
        "beta": f"1/{time_unit}",
        "V": f"{dose_unit}/{conc_unit}",
        "Cl": f"{dose_unit}/{conc_unit}/{time_unit}",
        "A": conc_unit,
        "B": conc_unit,
    }
