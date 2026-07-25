import pytest

from pkpd.core.units import convert_conc, convert_dose, convert_time


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
