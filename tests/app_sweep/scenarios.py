"""The A-Z catalog of things the sweep does to the app.

Steps are plain data, not code, so the catalog can be counted, filtered and
resumed by index after a crash. The driver knows how to perform each `action`;
this file only says what to perform and what should happen.

`expect_dialog` is the important field. A step that pops a QMessageBox when it
shouldn't is a bug, and so is a validation step that pops nothing — silent
acceptance of bad input is how wrong numbers reach a report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"

TIME_UNITS = ["hr", "mins", "day"]
CONC_UNITS = ["ng/mL", "mcg/mL", "mg/L", "mg/mL"]
DOSE_UNITS = ["mg", "mcg", "g"]
AUC_METHODS = ["linear", "linear_up_log_down", "linear_log", "linear_interp"]
LZ_MODES = ["Best Fit", "Time Range"]
COMPARTMENTAL_MODELS = [
    "1-compartment: IV bolus (K, V)",
    "1-compartment: IV bolus (Cl, V)",
    "1-compartment: IV infusion",
    "1-compartment: Extravascular",
    "2-compartment: IV bolus",
]
WEIGHT_SCHEMES = ["uniform", "inverse_y", "inverse_y2"]

# Datasets the app should accept without complaint.
CLEAN = [
    "01_iv_bolus_postdose.csv",
    "02_iv_bolus_t0.csv",
    "03_extravascular.csv",
    "04_iv_infusion.csv",
    "05_multi_subject.csv",
    "06_steady_state.csv",
    "07_with_weight.csv",
    "09_bql_missing.csv",
]


@dataclass
class Step:
    group: str
    action: str
    name: str
    params: dict = field(default_factory=dict)
    expect_dialog: bool = False
    # Steps known to be slow on purpose (the 10k-row file) get more headroom
    # before the parent calls it a hang.
    timeout_s: float = 30.0

    def as_dict(self) -> dict:
        return {"group": self.group, "action": self.action, "name": self.name,
                "params": self.params, "expect_dialog": self.expect_dialog,
                "timeout_s": self.timeout_s}


def _data_tab_steps() -> list[Step]:
    steps: list[Step] = [
        Step("data", "reset", "fresh window"),
        Step("data", "grid_add_row", "add row"),
        Step("data", "grid_delete_row", "delete selected row"),
        Step("data", "grid_paste", "paste TSV from clipboard"),
        Step("data", "grid_paste_no_selection", "paste with nothing selected",
             params={"clear_selection": True}),
        Step("data", "toggle_mode", "switch to Manual mode", params={"manual": True}),
        Step("data", "toggle_mode", "switch to Automatic mode", params={"manual": False}),
        Step("data", "toggle_help", "open user guide panel", params={"visible": True}),
        Step("data", "toggle_help", "close user guide panel", params={"visible": False}),
        Step("data", "set_dose_text", "dose mid-typing, not yet a number", params={"text": "5."}),
        Step("data", "set_dose_text", "dose not a number at all", params={"text": "abc"}),
        Step("data", "set_dose_text", "valid dose", params={"text": "500"}),
        # Acquiring an empty grid must complain, not crash.
        Step("data", "acquire_empty", "acquire with an empty grid", expect_dialog=True),
    ]

    for route in ("iv_bolus", "iv_infusion", "extravascular"):
        steps.append(Step("data", "set_route", f"route {route} (infusion column visibility)",
                          params={"route": route}))

    # Load every dataset through the import wizard, including the file whose
    # columns do not match the schema.
    for name in CLEAN + ["08_messy_headers.xlsx", "10_edge_cases.csv"]:
        steps.append(Step("data", "import_file", f"import {name}", params={"file": name}))
    # The invalid file must be rejected with a dialog listing the problems.
    steps.append(Step("data", "import_file", "import 11_invalid.csv (must be rejected)",
                      params={"file": "11_invalid.csv"}, expect_dialog=True))
    steps.append(Step("data", "import_file", "import 12_large.csv (10k rows)",
                      params={"file": "12_large.csv"}, timeout_s=120.0))

    # Units are display-only work, so sweep the whole 3x4x3 grid cheaply.
    for time_unit in TIME_UNITS:
        for conc_unit in CONC_UNITS:
            for dose_unit in DOSE_UNITS:
                steps.append(Step(
                    "units", "set_units", f"units {time_unit}/{conc_unit}/{dose_unit}",
                    params={"time": time_unit, "conc": conc_unit, "dose": dose_unit},
                ))
    return steps


def _nca_steps() -> list[Step]:
    steps: list[Step] = []

    # Full cross-product on the two datasets that exercise the most rules.
    for name in ("01_iv_bolus_postdose.csv", "03_extravascular.csv"):
        for auc in AUC_METHODS:
            for lz_mode in LZ_MODES:
                for steady in (False, True):
                    steps.append(Step(
                        "nca", "run_nca",
                        f"{name} {auc} / {lz_mode}{' / steady state' if steady else ''}",
                        params={
                            "file": name, "auc_method": auc, "lz_mode": lz_mode,
                            "lz_range": "4,24" if lz_mode == "Time Range" else "",
                            "steady_state": steady, "tau": "12" if steady else "",
                        },
                    ))

    # One pass per remaining dataset, so every route and shape is covered
    # without multiplying the whole matrix by twelve.
    for name in CLEAN:
        steps.append(Step("nca", "run_nca", f"{name} default settings",
                          params={"file": name, "auc_method": "linear_up_log_down",
                                   "lz_mode": "Best Fit"}))

    steps += [
        Step("nca", "run_nca", "manual point exclusion",
             params={"file": "01_iv_bolus_postdose.csv", "auc_method": "linear_up_log_down",
                      "lz_mode": "Best Fit", "exclude": "8,12"}),
        Step("nca", "run_nca", "every subject of the 6-subject file",
             params={"file": "05_multi_subject.csv", "all_subjects": True}),
        Step("nca", "run_nca", "degenerate profiles must still report",
             params={"file": "10_edge_cases.csv", "all_subjects": True}),
        Step("nca", "run_nca", "10k rows — UI stall measurement",
             params={"file": "12_large.csv", "all_subjects": False}, timeout_s=120.0),
        Step("nca", "toggle_plot_scale", "NCA plot semi-log/linear toggle"),
        Step("nca", "export_core_output", "export Core Output to a file"),

        # Input validation: each of these must raise exactly one dialog.
        Step("nca", "run_nca", "λz range that is not two numbers",
             params={"file": "01_iv_bolus_postdose.csv", "lz_mode": "Time Range",
                      "lz_range": "4"}, expect_dialog=True),
        Step("nca", "run_nca", "λz range that is not numeric",
             params={"file": "01_iv_bolus_postdose.csv", "lz_mode": "Time Range",
                      "lz_range": "a,b"}, expect_dialog=True),
        Step("nca", "run_nca", "exclusion list that is not numeric",
             params={"file": "01_iv_bolus_postdose.csv", "exclude": "eight"},
             expect_dialog=True),
        Step("nca", "run_nca", "steady state with a non-numeric Tau",
             params={"file": "01_iv_bolus_postdose.csv", "steady_state": True,
                      "tau": "soon"}, expect_dialog=True),
        Step("nca", "run_nca_no_data", "run NCA before loading anything",
             expect_dialog=True),
        Step("nca", "export_no_results", "export before running", expect_dialog=True),
    ]
    return steps


def _compartmental_steps() -> list[Step]:
    steps: list[Step] = []
    fittable = {
        "1-compartment: IV bolus (K, V)": "01_iv_bolus_postdose.csv",
        "1-compartment: IV bolus (Cl, V)": "01_iv_bolus_postdose.csv",
        "1-compartment: IV infusion": "04_iv_infusion.csv",
        "1-compartment: Extravascular": "03_extravascular.csv",
        "2-compartment: IV bolus": "01_iv_bolus_postdose.csv",
    }
    for model, name in fittable.items():
        for scheme in WEIGHT_SCHEMES:
            steps.append(Step("compartmental", "run_fit", f"{model} / {scheme}",
                              params={"file": name, "model": model, "weight_scheme": scheme}))
        steps.append(Step("compartmental", "run_fit", f"{model} / log residuals",
                          params={"file": name, "model": model, "log_residuals": True}))

    steps += [
        # Infusion model against data with no infusion_duration: must be a
        # setup dialog, not a crash and not a silent NaN fit.
        Step("compartmental", "run_fit", "infusion model without a duration",
             params={"file": "01_iv_bolus_postdose.csv",
                      "model": "1-compartment: IV infusion"}, expect_dialog=True),
        # A flat profile does converge — onto k at its lower bound. That is
        # not an error, so no dialog; it must be *flagged* instead, or a
        # meaningless fit reads as an ordinary one.
        Step("compartmental", "run_fit", "flat profile pins a parameter at its bound",
             params={"file": "10_edge_cases.csv", "subject": "flat",
                      "model": "1-compartment: IV bolus (K, V)",
                      "expect_params_at_bounds": True}),
        Step("compartmental", "run_fit", "single-point profile",
             params={"file": "10_edge_cases.csv", "subject": "single_point",
                      "model": "1-compartment: IV bolus (K, V)"}, expect_dialog=True),
        Step("compartmental", "run_fit_no_data", "fit before loading anything",
             expect_dialog=True),
        Step("compartmental", "toggle_plot_scale", "fit plot semi-log/linear toggle"),
    ]
    return steps


def _concurrency_steps() -> list[Step]:
    """Where the reported freezes and sudden stops most likely live: two
    QThreads in flight, or a second run started before the first finishes."""
    return [
        Step("concurrency", "double_run_nca", "click Run NCA twice in a row",
             params={"file": "01_iv_bolus_postdose.csv"}),
        Step("concurrency", "nca_and_fit_together", "NCA and a fit running at once",
             params={"file": "01_iv_bolus_postdose.csv"}),
        Step("concurrency", "rapid_subject_switch", "switch subject mid-run",
             params={"file": "05_multi_subject.csv"}),
        Step("concurrency", "repeated_runs", "20 back-to-back NCA runs",
             params={"file": "01_iv_bolus_postdose.csv", "count": 20}, timeout_s=120.0),
        Step("concurrency", "repeated_fits", "10 back-to-back fits",
             params={"file": "01_iv_bolus_postdose.csv", "count": 10}, timeout_s=120.0),
        Step("concurrency", "tab_switch_during_run", "navigate tabs while a run is in flight",
             params={"file": "12_large.csv"}, timeout_s=120.0),
    ]


def build_steps() -> list[Step]:
    return (_data_tab_steps() + _nca_steps() + _compartmental_steps() + _concurrency_steps())


STEPS = build_steps()
