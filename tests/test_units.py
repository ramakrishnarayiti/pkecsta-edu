import numpy as np
import pytest

from pkpd.core.units import (
    apply_compartmental_units,
    apply_nca_units,
    compartmental_units,
    convert_conc,
    convert_dose,
    convert_time,
    dose_mass_factor,
    nca_units,
    volume_unit,
)
from pkpd.pk.nca import nca_iv_bolus


def test_convert_time_hr_to_mins():
    assert convert_time(1.0, "hr", "mins") == pytest.approx(60.0)


def test_convert_time_day_to_hr():
    assert convert_time(1.0, "day", "hr") == pytest.approx(24.0)


def test_convert_time_same_unit_is_identity():
    assert convert_time(5.0, "hr", "hr") == pytest.approx(5.0)


def test_convert_conc_mg_l_equals_mcg_ml():
    # standard PK equivalence: 1 mg/L == 1 mcg/mL
    assert convert_conc(1.0, "mg/L", "mcg/mL") == pytest.approx(1.0)


def test_convert_conc_mcg_ml_to_ng_ml():
    assert convert_conc(1.0, "mcg/mL", "ng/mL") == pytest.approx(1000.0)


def test_convert_dose_g_to_mg():
    assert convert_dose(1.0, "g", "mg") == pytest.approx(1000.0)


def test_convert_dose_mcg_to_mg():
    assert convert_dose(1000.0, "mcg", "mg") == pytest.approx(1.0)


def test_convert_round_trip():
    original = 42.0
    converted = convert_conc(original, "ng/mL", "mg/mL")
    back = convert_conc(converted, "mg/mL", "ng/mL")
    assert back == pytest.approx(original)


# ---- dose/concentration mass reconciliation ----

def test_volume_unit_split_from_concentration_unit():
    assert volume_unit("ng/mL") == "mL"
    assert volume_unit("mg/L") == "L"


def test_dose_mass_factor_mg_dose_against_ng_per_ml():
    # 1 mg is 1e6 ng, so CL in mg/(ng/mL*hr) scales by 1e6 to reach mL/hr
    assert dose_mass_factor("mg", "ng/mL") == pytest.approx(1e6)


def test_dose_mass_factor_is_identity_when_masses_already_agree():
    assert dose_mass_factor("mg", "mg/L") == pytest.approx(1.0)


def test_clearance_lands_in_a_real_volume_unit():
    # 1-compartment bolus: C(t) = (Dose/V) * exp(-k*t), with V = 10 L and
    # k = 0.15/hr for a 500 mg dose. Concentrations recorded in ng/mL.
    dose_mg, v_litres, k = 500.0, 10.0, 0.15
    t = np.linspace(0, 48, 25)
    c_mg_per_l = (dose_mg / v_litres) * np.exp(-k * t)
    c_ng_per_ml = c_mg_per_l * 1000.0  # 1 mg/L == 1000 ng/mL

    raw = nca_iv_bolus(t, c_ng_per_ml, dose=dose_mg, auc_method="linear_up_log_down")
    scaled = apply_nca_units(raw, conc_unit="ng/mL", dose_unit="mg")

    # true CL = k*V = 1.5 L/hr = 1500 mL/hr, true Vz = 10 L = 10000 mL
    assert scaled["cl"] == pytest.approx(1500.0, rel=1e-6)
    assert scaled["vz"] == pytest.approx(10_000.0, rel=1e-6)
    assert nca_units("hr", "ng/mL", "mg")["cl"] == "mL/hr"
    assert nca_units("hr", "ng/mL", "mg")["vz"] == "mL"


def test_same_data_in_different_units_gives_the_same_physical_clearance():
    dose_mg, v_litres, k = 500.0, 10.0, 0.15
    t = np.linspace(0, 48, 25)
    c_mg_per_l = (dose_mg / v_litres) * np.exp(-k * t)

    in_mg_per_l = apply_nca_units(
        nca_iv_bolus(t, c_mg_per_l, dose=dose_mg, auc_method="linear_up_log_down"),
        conc_unit="mg/L", dose_unit="mg")
    in_ng_per_ml = apply_nca_units(
        nca_iv_bolus(t, c_mg_per_l * 1000.0, dose=dose_mg * 1000.0,
                      auc_method="linear_up_log_down"),
        conc_unit="ng/mL", dose_unit="mcg")

    # 1.5 L/hr and 1500 mL/hr are the same clearance in the two volume units
    assert in_mg_per_l["cl"] == pytest.approx(1.5, rel=1e-6)
    assert in_ng_per_ml["cl"] == pytest.approx(in_mg_per_l["cl"] * 1000.0, rel=1e-6)


def test_apply_nca_units_leaves_non_volume_parameters_alone():
    t = np.linspace(0, 24, 9)
    c = 100.0 * np.exp(-0.15 * t)
    raw = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down", weight=70.0)
    scaled = apply_nca_units(raw, conc_unit="ng/mL", dose_unit="mg")

    for key in ("cmax", "tmax", "auc_t", "auc_inf", "half_life", "mrt", "dose", "dose_per_kg"):
        assert scaled[key] == pytest.approx(raw[key])
    assert scaled["cl_per_kg"] == pytest.approx(raw["cl_per_kg"] * 1e6)


def test_apply_nca_units_does_not_mutate_the_raw_result():
    t = np.linspace(0, 24, 9)
    c = 100.0 * np.exp(-0.15 * t)
    raw = nca_iv_bolus(t, c, dose=500.0, auc_method="linear_up_log_down")
    before = raw["cl"]
    apply_nca_units(raw, conc_unit="ng/mL", dose_unit="mg")
    assert raw["cl"] == pytest.approx(before)


def test_apply_nca_units_tolerates_missing_lambda_z_parameters():
    raw = nca_iv_bolus(np.array([0.0, 1.0]), np.array([10.0, 5.0]), dose=500.0)
    scaled = apply_nca_units(raw, conc_unit="ng/mL", dose_unit="mg")
    assert scaled["cl"] is None


def test_apply_compartmental_units_scales_v_and_cl_only():
    result = {
        "params": {"Cl": 1.5, "V": 10.0, "k": 0.15},
        "param_se": {"Cl": 0.1, "V": 0.5, "k": 0.01},
        "param_cv_pct": {"Cl": 6.7, "V": 5.0, "k": 6.7},
    }
    scaled = apply_compartmental_units(result, conc_unit="ng/mL", dose_unit="mg")
    assert scaled["params"]["V"] == pytest.approx(10.0 * 1e6)
    assert scaled["params"]["Cl"] == pytest.approx(1.5 * 1e6)
    assert scaled["params"]["k"] == pytest.approx(0.15)  # 1/time, unaffected
    assert scaled["param_se"]["V"] == pytest.approx(0.5 * 1e6)
    # CV% is se/value, so scaling both leaves it unchanged
    assert scaled["param_cv_pct"] == result["param_cv_pct"]
    assert compartmental_units("hr", "ng/mL", "mg")["Cl"] == "mL/hr"
