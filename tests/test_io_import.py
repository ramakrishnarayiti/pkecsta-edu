import pandas as pd

from pkpd.core.data_model import Dataset
from pkpd.core.io_import import fill_missing_columns


def test_fill_missing_columns_adds_absent_columns_only():
    df = pd.DataFrame({"subject_id": [1, 1], "time": [0, 1], "concentration": [100, 80]})
    filled = fill_missing_columns(df, {"route": "iv_bolus", "dose": "500"})

    assert (filled["route"] == "iv_bolus").all()
    assert (filled["dose"] == "500").all()
    assert list(filled["time"]) == [0, 1]


def test_fill_missing_columns_does_not_overwrite_present_column():
    df = pd.DataFrame({"subject_id": [1], "route": ["extravascular"]})
    filled = fill_missing_columns(df, {"route": "iv_bolus"})
    assert filled["route"].iloc[0] == "extravascular"


def test_dose_default_string_must_be_numeric_before_dataset_validation():
    """Regression: fill_missing_columns fills dose as a string (e.g. "500"
    from a UI text field); Dataset validation requires numeric dtype, so
    the caller (ImportWizard) must coerce with pd.to_numeric before
    constructing Dataset, or validation incorrectly rejects it."""
    df = pd.DataFrame({
        "subject_id": [1, 1], "time": [0.0, 1.0], "concentration": [100.0, 80.0],
    })
    filled = fill_missing_columns(df, {"route": "iv_bolus", "dose": "500"})
    filled["dose"] = pd.to_numeric(filled["dose"], errors="coerce")

    ds = Dataset(filled)  # must not raise ValidationError
    assert (ds.data["dose"] == 500).all()


# ---- column-name guessing ----

def test_guess_column_matches_exact_name_first():
    from pkpd.ui.import_wizard import guess_column
    assert guess_column("concentration", ["conc", "concentration"]) == "concentration"


def test_guess_column_handles_the_common_real_world_headers():
    # The shipped Sample Data.xlsx uses exactly these, and mapped nothing
    # but time before aliases existed.
    from pkpd.ui.import_wizard import guess_column
    columns = ["ID", "Time", "Conc", "dose", "route"]
    assert guess_column("subject_id", columns) == "ID"
    assert guess_column("time", columns) == "Time"
    assert guess_column("concentration", columns) == "Conc"
    assert guess_column("dose", columns) == "dose"


def test_guess_column_returns_none_when_nothing_matches():
    from pkpd.ui.import_wizard import guess_column
    assert guess_column("concentration", ["alpha", "beta"]) is None
