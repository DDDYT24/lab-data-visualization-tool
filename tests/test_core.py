from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from labviz.core import (
    clean_dataframe,
    column_profile,
    load_data,
    load_data_bytes,
    summarize_dataframe,
    validate_columns_exist,
)


@pytest.fixture
def experiment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0, 1, 1, 2, 3],
            "temperature": [20.0, 21.0, 21.0, None, 22.0],
            "sample": ["a", "b", "b", None, "c"],
        }
    )


@pytest.mark.parametrize(
    ("filename", "payload"),
    [
        ("data.csv", b"time,value\n0,1\n1,2\n"),
        ("data.tsv", b"time\tvalue\n0\t1\n1\t2\n"),
        ("data.txt", b"time;value\n0;1\n1;2\n"),
        ("data.json", b'[{"time":0,"value":1},{"time":1,"value":2}]'),
    ],
)
def test_load_data_bytes_supported_formats(filename: str, payload: bytes) -> None:
    frame = load_data_bytes(payload, filename)
    assert list(frame.columns) == ["time", "value"]
    assert len(frame) == 2


def test_load_excel_bytes() -> None:
    source = pd.DataFrame({"time": [0, 1], "value": [2.0, 3.0]})
    buffer = __import__("io").BytesIO()
    source.to_excel(buffer, index=False, engine="openpyxl")
    loaded = load_data_bytes(buffer.getvalue(), "experiment.xlsx")
    pd.testing.assert_frame_equal(loaded, source, check_dtype=False)


def test_load_data_path_and_missing_path(tmp_path: Path) -> None:
    path = tmp_path / "experiment.csv"
    path.write_text("x,y\n1,2\n", encoding="utf-8")
    assert load_data(path).iloc[0].to_dict() == {"x": 1, "y": 2}
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


@pytest.mark.parametrize(
    ("payload", "filename"),
    [(b"", "data.csv"), (b"x\n1\n", "data.parquet")],
)
def test_load_rejects_invalid_input(payload: bytes, filename: str) -> None:
    with pytest.raises(ValueError):
        load_data_bytes(payload, filename)


def test_load_rejects_duplicate_trimmed_column_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        load_data_bytes(b"value,value \n1,2\n", "data.csv")


def test_summary_and_profile(experiment_frame: pd.DataFrame) -> None:
    summary = summarize_dataframe(experiment_frame)
    assert summary.rows == 5
    assert summary.columns == 3
    assert summary.numeric_columns == 2
    assert summary.missing_cells == 2
    assert summary.duplicate_rows == 1
    assert summary.memory_bytes > 0

    profile = column_profile(experiment_frame)
    assert list(profile.columns) == ["column", "type", "missing", "unique"]
    assert profile.loc[profile["column"] == "temperature", "missing"].item() == 1


def test_validate_columns(experiment_frame: pd.DataFrame) -> None:
    validate_columns_exist(experiment_frame, ["time", "temperature"])
    with pytest.raises(ValueError, match="pressure"):
        validate_columns_exist(experiment_frame, ["pressure"])


@pytest.mark.parametrize(
    ("strategy", "expected_missing"),
    [
        ("keep", 2),
        ("drop", 0),
        ("ffill", 0),
        ("bfill", 0),
        ("mean", 1),
        ("median", 1),
    ],
)
def test_cleaning_strategies(
    experiment_frame: pd.DataFrame, strategy: str, expected_missing: int
) -> None:
    original = experiment_frame.copy(deep=True)
    cleaned = clean_dataframe(
        experiment_frame,
        drop_duplicates=False,
        missing=strategy,  # type: ignore[arg-type]
    )
    assert int(cleaned.isna().sum().sum()) == expected_missing
    pd.testing.assert_frame_equal(experiment_frame, original)


def test_cleaning_removes_exact_duplicates() -> None:
    frame = pd.DataFrame({"x": [1, 1], "y": [2, 2]})
    assert len(clean_dataframe(frame, missing="keep")) == 1
    assert len(clean_dataframe(frame, drop_duplicates=False, missing="keep")) == 2


def test_cleaning_rejects_unknown_strategy(experiment_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        clean_dataframe(experiment_frame, missing="magic")  # type: ignore[arg-type]
