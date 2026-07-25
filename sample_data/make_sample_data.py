"""Regenerate every sample dataset.

    python sample_data/make_sample_data.py

Concentrations come from the closed-form models in pkpd/pk/compartmental/
with known true parameters, so each dataset has a right answer to check
against rather than being an arbitrary pile of numbers. Seeds are fixed and
values are rounded, so regenerating produces a byte-identical file and any
change shows up as a reviewable diff.

Every file uses the app's own schema column names except 08_messy_headers,
which deliberately does not — that one exists to exercise the import
wizard's column mapping.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pkpd.core.data_model import ALL_COLUMNS  # noqa: E402
from pkpd.pk.compartmental.models import (  # noqa: E402
    conc_1c_extravascular,
    conc_1c_iv_bolus,
    conc_1c_iv_infusion,
)

OUT = Path(__file__).parent
DOSE = 500.0

# Sampling schedules used across the datasets. Literal, not linspace — a
# schedule that shifts when a library changes defeats the purpose.
RICH = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
ORAL = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0]
TAU_SCHEDULE = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Every dataset carries the full schema so the app never has to guess."""
    df = pd.DataFrame(rows)
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[ALL_COLUMNS]


def _rows(subject, times, concs, route, dose=DOSE, infusion_duration="", weight=""):
    return [
        {
            "subject_id": subject,
            "time": t,
            "concentration": round(float(c), 4),
            "dose": dose,
            "route": route,
            "infusion_duration": infusion_duration,
            "weight": weight,
        }
        for t, c in zip(times, concs)
    ]


def _noise(n: int, seed: int, pct: float = 0.07) -> np.ndarray:
    """Fixed-seed proportional noise. Realistic assay scatter, so the
    lambda_z fit and the curve fits have something to actually do."""
    return 1.0 + pct * np.random.default_rng(seed).standard_normal(n)


# --- 01 / 02: IV bolus, with and without a t=0 sample -----------------------
# True: k = 0.15 /hr, V = 10 L, Dose = 500 mg -> CL = 1.5 L/hr, C0 = 50 mg/L

def bolus_postdose() -> pd.DataFrame:
    t = np.array(RICH)
    c = conc_1c_iv_bolus(t, 0.15, 10.0, DOSE) * _noise(len(t), 1)
    return _frame(_rows(1, t, c, "iv_bolus"))


def bolus_with_t0() -> pd.DataFrame:
    t = np.array([0.0] + RICH)
    c = conc_1c_iv_bolus(t, 0.15, 10.0, DOSE) * _noise(len(t), 2)
    return _frame(_rows(1, t, c, "iv_bolus"))


# --- 03: extravascular ------------------------------------------------------
# True: ka = 1.2 /hr, k = 0.15 /hr, V/F = 30 L. Tmax ~ 1.98 hr.

def extravascular() -> pd.DataFrame:
    t = np.array(ORAL)
    c = conc_1c_extravascular(t, 1.2, 0.15, 30.0, DOSE) * _noise(len(t), 3)
    return _frame(_rows(1, t, c, "extravascular"))


# --- 04: IV infusion --------------------------------------------------------
# True: k = 0.2 /hr, V = 15 L, Tinf = 2 hr. MRT must lose Tinf/2 = 1 hr.

def infusion() -> pd.DataFrame:
    t = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0])
    c = conc_1c_iv_infusion(t, 0.2, 15.0, DOSE, 2.0) * _noise(len(t), 4)
    return _frame(_rows(1, t, c, "iv_infusion", infusion_duration=2.0))


# --- 05: six subjects -------------------------------------------------------

def multi_subject() -> pd.DataFrame:
    rows = []
    for i, (k, v) in enumerate(
        [(0.12, 9.0), (0.15, 10.0), (0.18, 11.5), (0.10, 8.5), (0.22, 12.0), (0.14, 10.5)],
        start=1,
    ):
        t = np.array(RICH)
        c = conc_1c_iv_bolus(t, k, v, DOSE) * _noise(len(t), 100 + i)
        rows += _rows(i, t, c, "iv_bolus")
    return _frame(rows)


# --- 06: steady state, Tau = 12 --------------------------------------------
# Repeated 1-compartment bolus at steady state:
#   C(t) = (Dose/V) * exp(-k*t) / (1 - exp(-k*Tau))
# so t=0 is the trough and the profile is trough-to-trough.

def steady_state() -> pd.DataFrame:
    t = np.array(TAU_SCHEDULE)
    k, v, tau = 0.15, 10.0, 12.0
    c = (DOSE / v) * np.exp(-k * t) / (1.0 - np.exp(-k * tau))
    return _frame(_rows(1, t, c * _noise(len(t), 6), "iv_bolus"))


# --- 07: body weights, for the per-kg parameters ---------------------------

def with_weight() -> pd.DataFrame:
    rows = []
    for i, (k, v, wt) in enumerate([(0.15, 10.0, 70.0), (0.13, 8.0, 58.0), (0.17, 12.5, 91.0)], start=1):
        t = np.array(RICH)
        c = conc_1c_iv_bolus(t, k, v, DOSE) * _noise(len(t), 200 + i)
        rows += _rows(i, t, c, "iv_bolus", weight=wt)
    return _frame(rows)


# --- 08: non-schema headers, for the import wizard -------------------------

def messy_headers() -> pd.DataFrame:
    t = np.array(RICH)
    c = conc_1c_iv_bolus(t, 0.15, 10.0, DOSE) * _noise(len(t), 8)
    return pd.DataFrame({
        "ID": 1,
        "Time": t,
        "Conc": np.round(c, 4),
        "dose": DOSE,
        "route": "iv_bolus",
    })


# --- 09: BQL and missing ---------------------------------------------------

def bql_missing() -> pd.DataFrame:
    t = np.array(RICH)
    c = conc_1c_iv_bolus(t, 0.15, 10.0, DOSE) * _noise(len(t), 9)
    df = _frame(_rows(1, t, c, "iv_bolus"))
    # A blank cell and a literal BQL marker are how this actually arrives in
    # real files. All three must coerce to NaN and be excluded, not crash.
    # The column has to hold strings, so it goes to object first.
    df["concentration"] = df["concentration"].astype(object)
    df.loc[2, "concentration"] = ""
    df.loc[7, "concentration"] = "BQL"
    df.loc[10, "concentration"] = "<LLOQ"
    return df


# --- 10: degenerate profiles -----------------------------------------------

def edge_cases() -> pd.DataFrame:
    rows = []
    rows += _rows("single_point", [2.0], [42.0], "iv_bolus")
    rows += _rows("all_missing", [1.0, 2.0, 4.0], [np.nan] * 3, "iv_bolus")
    rows += _rows("flat", [1.0, 2.0, 4.0, 8.0, 12.0], [10.0] * 5, "iv_bolus")
    rows += _rows("two_points", [1.0, 4.0], [30.0, 12.0], "iv_bolus")
    rows += _rows("no_decline", [1.0, 2.0, 4.0, 8.0], [5.0, 9.0, 14.0, 22.0], "extravascular")
    df = _frame(rows)
    df["concentration"] = df["concentration"].astype(object).where(
        df["concentration"].notna(), "")
    return df


# --- 11: rows the validator must reject ------------------------------------

def invalid() -> pd.DataFrame:
    rows = [
        {"subject_id": 1, "time": 1.0, "concentration": 40.0, "dose": DOSE, "route": "iv_bolus"},
        {"subject_id": 1, "time": 1.0, "concentration": 38.0, "dose": DOSE, "route": "iv_bolus"},  # duplicate time
        {"subject_id": 1, "time": 2.0, "concentration": -5.0, "dose": DOSE, "route": "iv_bolus"},  # negative
        {"subject_id": 1, "time": -1.0, "concentration": 30.0, "dose": DOSE, "route": "iv_bolus"},  # negative time
        {"subject_id": 1, "time": 4.0, "concentration": "abc", "dose": DOSE, "route": "iv_bolus"},  # text
        {"subject_id": 1, "time": 8.0, "concentration": 12.0, "dose": DOSE, "route": "teleport"},  # bad route
    ]
    return _frame(rows)


# --- 12: large, for stall measurement --------------------------------------

def large() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    t = np.round(np.linspace(0.25, 48.0, 200), 4)
    rows = []
    for subject in range(1, 51):
        k = 0.10 + 0.002 * subject
        v = 8.0 + 0.1 * subject
        c = conc_1c_iv_bolus(t, k, v, DOSE) * (1 + 0.07 * rng.standard_normal(len(t)))
        rows += _rows(subject, t, c, "iv_bolus")
    return _frame(rows)


DATASETS = {
    "01_iv_bolus_postdose.csv": bolus_postdose,
    "02_iv_bolus_t0.csv": bolus_with_t0,
    "03_extravascular.csv": extravascular,
    "04_iv_infusion.csv": infusion,
    "05_multi_subject.csv": multi_subject,
    "06_steady_state.csv": steady_state,
    "07_with_weight.csv": with_weight,
    "08_messy_headers.xlsx": messy_headers,
    "09_bql_missing.csv": bql_missing,
    "10_edge_cases.csv": edge_cases,
    "11_invalid.csv": invalid,
    "12_large.csv": large,
}


def main() -> None:
    for name, build in DATASETS.items():
        df = build()
        path = OUT / name
        if name.endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            # newline="" keeps line endings identical across platforms, so a
            # regenerated file diffs cleanly instead of showing every row.
            with open(path, "w", newline="", encoding="utf-8") as handle:
                df.to_csv(handle, index=False)
        print(f"{name:32s} {len(df):6d} rows")


if __name__ == "__main__":
    main()
