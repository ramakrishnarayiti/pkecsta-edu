import pandas as pd
import pytest

from pkpd.core.data_model import Dataset, ValidationError


def make_records():
    return [
        {"subject_id": 1, "time": 0.0, "concentration": 0.0, "dose": 100.0, "route": "iv_bolus"},
        {"subject_id": 1, "time": 1.0, "concentration": 50.0, "dose": 100.0, "route": "iv_bolus"},
        {"subject_id": 2, "time": 0.0, "concentration": 0.0, "dose": 100.0, "route": "iv_bolus"},
    ]


def test_valid_dataset_builds_and_fills_optional_column():
    ds = Dataset.from_records(make_records())
    assert "infusion_duration" in ds.data.columns
    assert ds.subject_ids() == [1, 2]


def test_subject_data_sorted_by_time():
    records = make_records()
    records[1]["time"] = -1  # will trigger validation error separately; use valid instead
    records[1]["time"] = 5.0
    ds = Dataset.from_records(records)
    sub1 = ds.subject_data(1)
    assert list(sub1["time"]) == sorted(sub1["time"])


def test_missing_required_column_raises():
    df = pd.DataFrame([{"subject_id": 1, "time": 0.0, "concentration": 0.0}])
    with pytest.raises(ValidationError):
        Dataset(df)


def test_duplicate_time_per_subject_raises():
    records = [
        {"subject_id": 1, "time": 0.0, "concentration": 0.0, "dose": 100.0, "route": "iv_bolus"},
        {"subject_id": 1, "time": 0.0, "concentration": 5.0, "dose": 100.0, "route": "iv_bolus"},
    ]
    with pytest.raises(ValidationError):
        Dataset.from_records(records)


def test_invalid_route_raises():
    records = [{"subject_id": 1, "time": 0.0, "concentration": 0.0, "dose": 100.0, "route": "bogus"}]
    with pytest.raises(ValidationError):
        Dataset.from_records(records)
