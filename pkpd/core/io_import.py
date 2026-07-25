"""CSV/Excel import with column mapping. Real research files rarely already
use the internal schema's column names, so callers must supply a mapping."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_model import ALL_COLUMNS, Dataset


def read_raw(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file as-is (no schema applied yet), for the UI to
    show column names and let the user map them."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def apply_column_mapping(raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """mapping: internal column name -> source column name in `raw`.
    Only required/optional schema columns present in `mapping` are kept."""
    out = pd.DataFrame()
    for internal_col, source_col in mapping.items():
        if internal_col not in ALL_COLUMNS:
            continue
        out[internal_col] = raw[source_col]
    return out


def fill_missing_columns(df: pd.DataFrame, defaults: dict[str, str]) -> pd.DataFrame:
    """Fill columns absent from `df` with a constant value (e.g. route/dose
    set once in the UI instead of present in the imported file). Columns
    already present are left untouched."""
    df = df.copy()
    for col, value in defaults.items():
        if col not in df.columns:
            df[col] = value
    return df


def load_dataset(path: str | Path, mapping: dict[str, str]) -> Dataset:
    raw = read_raw(path)
    mapped = apply_column_mapping(raw, mapping)
    return Dataset(mapped)
