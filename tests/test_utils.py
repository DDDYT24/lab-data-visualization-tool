import pandas as pd
import pytest
import app


@pytest.fixture
def df_basic() -> pd.DataFrame:
    return pd.DataFrame({"time": [0, 1, 2, 3, 4], "temperature": [20, 21, 19, 22, 20]})


@pytest.fixture
def df_with_nan() -> pd.DataFrame:
    return pd.DataFrame({"time": [0, 1, 2, 3, 4], "temperature": [20.0, 21.0, None, 22.0, 20.0]})


def test_validate_columns_exist_ok(df_basic: pd.DataFrame) -> None:
    app.validate_columns_exist(df_basic, ["time", "temperature"])


def test_validate_columns_exist_missing(df_basic: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        app.validate_columns_exist(df_basic, ["time", "temperature", "pressure"])


def test_clean_dataframe_dropna(df_with_nan: pd.DataFrame) -> None:
    cleaned = app.clean_dataframe(df_with_nan, dropna=True)
    assert cleaned["temperature"].isna().sum() == 0


def test_clean_dataframe_keepna(df_with_nan: pd.DataFrame) -> None:
    kept = app.clean_dataframe(df_with_nan, dropna=False)
    assert kept["temperature"].isna().any()
    assert len(kept) <= len(df_with_nan)


def test_load_csv_bytes_ok() -> None:
    csv = b"time,temperature\n0,20\n1,21\n"
    df = app.load_csv_bytes(csv)
    assert list(df.columns) == ["time", "temperature"]
    assert len(df) == 2


def test_load_csv_bytes_empty() -> None:
    with pytest.raises(ValueError):
        app.load_csv_bytes(b"")
