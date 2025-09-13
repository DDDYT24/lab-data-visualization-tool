import pytest
import pandas as pd

import app


@pytest.fixture
def df_basic():
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "temperature": [20, 21, 19, 22, 20],
        }
    )


@pytest.fixture
def df_with_nan():
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "temperature": [20, 21, None, 22, 20],
        }
    )


def test_validate_columns_exist_ok(df_basic):
    app.validate_columns_exist(df_basic, ["time", "temperature"])


def test_validate_columns_exist_missing(df_basic):
    with pytest.raises(ValueError):
        app.validate_columns_exist(df_basic, ["time", "temp_missing"])


def test_clean_dataframe_dropna(df_with_nan):
    cleaned = app.clean_dataframe(df_with_nan, dropna=True)
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == len(df_with_nan) - 1


def test_clean_dataframe_keepna(df_with_nan):
    kept = app.clean_dataframe(df_with_nan, dropna=False)
    assert len(kept) == len(df_with_nan)
    assert kept.isna().sum().sum() >= 1
