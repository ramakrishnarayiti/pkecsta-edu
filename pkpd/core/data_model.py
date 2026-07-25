"""Shared PK/PD dataset model. Stays neutral to PK vs PD — a concentration-time
row is just a row; pk/ and (later) pd/ each interpret it their own way."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["subject_id", "time", "concentration", "dose", "route"]
OPTIONAL_COLUMNS = ["infusion_duration", "weight"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


class Route(str, Enum):
    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"
    EXTRAVASCULAR = "extravascular"


class ValidationError(Exception):
    """Raised with all row-level problems found, not just the first."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def validate(df: pd.DataFrame) -> None:
    """Validate a dataset against the shared schema. Raises ValidationError
    with every problem found (not just the first) so the UI can show a table."""
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError([f"missing required column(s): {missing}"])

    for col in ("time", "concentration", "dose"):
        non_numeric = df[~df[col].apply(lambda v: isinstance(v, (int, float, np.integer, np.floating)))]
        if len(non_numeric):
            problems.append(f"column '{col}' has {len(non_numeric)} non-numeric value(s), rows: {non_numeric.index.tolist()}")

    if pd.api.types.is_numeric_dtype(df["time"]):
        negative_time = df[df["time"] < 0]
        if len(negative_time):
            problems.append(f"negative time values at rows: {negative_time.index.tolist()}")

    if pd.api.types.is_numeric_dtype(df["concentration"]):
        negative_conc = df[df["concentration"] < 0]
        if len(negative_conc):
            problems.append(f"negative concentration values at rows: {negative_conc.index.tolist()}")

    invalid_route = df[~df["route"].isin([r.value for r in Route])]
    if len(invalid_route):
        problems.append(f"invalid route value(s) at rows: {invalid_route.index.tolist()} (must be one of {[r.value for r in Route]})")

    dupes = df.duplicated(subset=["subject_id", "time"], keep=False)
    if dupes.any():
        problems.append(f"duplicate (subject_id, time) pairs at rows: {df[dupes].index.tolist()}")

    if problems:
        raise ValidationError(problems)


@dataclass
class Dataset:
    """Validated concentration-time dataset. Immutable after construction —
    build a new Dataset rather than mutating .data in place."""

    data: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.data.columns]
        if missing:
            raise ValidationError([f"missing required column(s): {missing}"])
        for col in OPTIONAL_COLUMNS:
            if col not in self.data.columns:
                self.data[col] = np.nan
        self.data = self.data[ALL_COLUMNS].reset_index(drop=True)
        validate(self.data)

    def subject_ids(self) -> list:
        return sorted(self.data["subject_id"].unique().tolist())

    def subject_data(self, subject_id) -> pd.DataFrame:
        """Rows for one subject, sorted by time."""
        return (
            self.data[self.data["subject_id"] == subject_id]
            .sort_values("time")
            .reset_index(drop=True)
        )

    @classmethod
    def from_records(cls, records: list[dict]) -> "Dataset":
        return cls(pd.DataFrame.from_records(records))
