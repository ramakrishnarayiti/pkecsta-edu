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
# Doses are masses, so one mass table serves both dose conversion and the
# dose->concentration-mass reconciliation below.
_MASS_TO_MG = {"ng": 1e-6, "mcg": 1e-3, "mg": 1.0, "g": 1000.0}

# A concentration unit is a mass over a volume; splitting it is what lets
# CL and V come out in real volume units instead of a mixed-mass composite.
_CONC_PARTS = {
    "ng/mL": ("ng", "mL"),
    "mcg/mL": ("mcg", "mL"),
    "mg/L": ("mg", "L"),
    "mg/mL": ("mg", "mL"),
}

# Parameters carrying a volume dimension (CL is volume/time). These are the
# only NCA outputs whose value depends on dose and concentration agreeing on
# a mass unit — everything else is already self-consistent.
_VOLUME_PARAMS = ("cl", "vz", "vss", "cl_per_kg", "vz_per_kg", "vss_per_kg")


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
    return convert(value, from_unit, to_unit, _MASS_TO_MG)


def volume_unit(conc_unit: str) -> str:
    """Volume half of a concentration unit — the unit CL and V come out in."""
    return _CONC_PARTS[conc_unit][1]


def dose_mass_factor(dose_unit: str, conc_unit: str) -> float:
    """Factor putting a dose expressed in `dose_unit` onto the mass unit the
    concentration uses. CL = Dose/AUC is only a volume/time once those two
    masses agree: 500 mg against ng/mL gives CL in mg/(ng/mL*hr), which is a
    real number in a unit nobody can use. Multiplying by this factor turns
    it into mL/hr."""
    mass, _volume = _CONC_PARTS[conc_unit]
    return convert(1.0, dose_unit, mass, _MASS_TO_MG)


def apply_nca_units(result: dict, conc_unit: str, dose_unit: str) -> dict:
    """Rescale the volume-dimensioned NCA outputs so they read in the
    concentration's volume unit. Returns a new dict — the raw analysis
    result stays untouched, keeping pk/nca.py unit-agnostic."""
    factor = dose_mass_factor(dose_unit, conc_unit)
    out = dict(result)
    for key in _VOLUME_PARAMS:
        if out.get(key) is not None:
            out[key] = out[key] * factor
    return out


def apply_compartmental_units(result: dict, conc_unit: str, dose_unit: str) -> dict:
    """Same rescaling for a compartmental fit, where V and Cl live inside
    the nested params/param_se dicts. param_cv_pct is a ratio of the two, so
    it is scale-invariant and deliberately left alone."""
    factor = dose_mass_factor(dose_unit, conc_unit)
    out = dict(result)
    for block in ("params", "param_se"):
        if block in out:
            out[block] = {
                name: (value * factor if name in ("V", "V1", "Cl") else value)
                for name, value in out[block].items()
            }
    return out


def nca_units(time_unit: str, conc_unit: str, dose_unit: str) -> dict[str, str]:
    """Labels for the values `apply_nca_units` produces — the two must be
    changed together or the table lies about what it is showing."""
    ct = f"{conc_unit}*{time_unit}"
    vol = volume_unit(conc_unit)
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
        "cl": f"{vol}/{time_unit}",
        "vz": vol,
        "vss": vol,
        "mrt": time_unit,
        "dose": dose_unit,
        "slope": f"1/{time_unit}",
        "dose_per_kg": f"{dose_unit}/kg",
        "cl_per_kg": f"{vol}/{time_unit}/kg",
        "vz_per_kg": f"{vol}/kg",
        "vss_per_kg": f"{vol}/kg",
    }


def compartmental_units(time_unit: str, conc_unit: str, dose_unit: str) -> dict[str, str]:
    """Labels for the values `apply_compartmental_units` produces."""
    vol = volume_unit(conc_unit)
    return {
        "k": f"1/{time_unit}",
        "ka": f"1/{time_unit}",
        "alpha": f"1/{time_unit}",
        "beta": f"1/{time_unit}",
        "V": vol,
        "V1": vol,
        "Cl": f"{vol}/{time_unit}",
        "A": conc_unit,
        "B": conc_unit,
        "tlag": time_unit,
    }
