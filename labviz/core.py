"""Data loading, validation, cleaning, and profiling primitives."""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".json", ".xlsx")
MissingStrategy = Literal["keep", "drop", "ffill", "bfill", "mean", "median"]


@dataclass(frozen=True)
class DataSummary:
    """Compact, serializable quality summary for one table."""

    rows: int
    columns: int
    numeric_columns: int
    categorical_columns: int
    missing_cells: int
    duplicate_rows: int
    memory_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _validate_payload(data: bytes, filename: str) -> str:
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"The file exceeds the {limit_mb} MB upload limit.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Use: {supported}.")
    return suffix


def load_data_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Load a supported tabular file from bytes without writing it to disk."""

    suffix = _validate_payload(data, filename)
    source = io.BytesIO(data)

    try:
        if suffix == ".csv":
            frame = pd.read_csv(source)
        elif suffix == ".tsv":
            frame = pd.read_csv(source, sep="\t")
        elif suffix == ".txt":
            frame = pd.read_csv(source, sep=None, engine="python")
        elif suffix == ".json":
            frame = pd.read_json(source)
        else:
            frame = pd.read_excel(source, engine="openpyxl")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValueError("Excel support requires the openpyxl dependency.") from exc
    except Exception as exc:
        raise ValueError(f"Could not read '{filename}' as tabular data.") from exc

    if frame.columns.empty:
        raise ValueError("The file does not contain any columns.")
    column_names = [str(column).strip() for column in frame.columns]
    duplicates = sorted(name for name, count in Counter(column_names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Column names must be unique after trimming: {duplicates}")
    frame.columns = column_names
    return frame


def load_data(path: Path) -> pd.DataFrame:
    """Load a supported local file using the same validation as the web app."""

    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return load_data_bytes(path.read_bytes(), path.name)


def validate_columns_exist(frame: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        available = list(map(str, frame.columns))
        raise ValueError(f"Missing columns: {missing}; available: {available}")


def summarize_dataframe(frame: pd.DataFrame) -> DataSummary:
    numeric_count = len(frame.select_dtypes(include="number").columns)
    return DataSummary(
        rows=len(frame),
        columns=len(frame.columns),
        numeric_columns=numeric_count,
        categorical_columns=len(frame.columns) - numeric_count,
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
    )


def clean_dataframe(
    frame: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    missing: MissingStrategy = "drop",
) -> pd.DataFrame:
    """Return a cleaned copy; the input DataFrame is never mutated."""

    if missing not in {"keep", "drop", "ffill", "bfill", "mean", "median"}:
        raise ValueError("Unknown missing-value strategy.")

    cleaned = frame.copy()
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    if missing == "drop":
        cleaned = cleaned.dropna()
    elif missing == "ffill":
        cleaned = cleaned.ffill()
    elif missing == "bfill":
        cleaned = cleaned.bfill()
    elif missing in {"mean", "median"}:
        numeric = cleaned.select_dtypes(include="number").columns
        if len(numeric):
            fill_values = getattr(cleaned[numeric], missing)()
            cleaned.loc[:, numeric] = cleaned[numeric].fillna(fill_values)

    return cleaned.reset_index(drop=True)


def column_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the compact column-quality table shown by the front end."""

    return pd.DataFrame(
        {
            "column": frame.columns,
            "type": [str(dtype) for dtype in frame.dtypes],
            "missing": frame.isna().sum().to_numpy(),
            "unique": frame.nunique(dropna=True).to_numpy(),
        }
    )
