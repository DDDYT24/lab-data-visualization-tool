import numpy as np
import pandas as pd
import pytest

import app


# ---------- 辅助 fixtures ----------


@pytest.fixture()
def df_basic() -> pd.DataFrame:
    """一个最小可用的 DataFrame."""
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "temperature": [20, 21, 19, 22, 20],
        }
    )


@pytest.fixture()
def df_with_nan() -> pd.DataFrame:
    """包含 NaN 的 DataFrame，用于清洗函数测试。"""
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "temperature": [20.0, 21.0, np.nan, 22.0, 20.0],
        }
    )


# ---------- load_csv_bytes ----------


def test_load_csv_bytes_ok(df_basic: pd.DataFrame) -> None:
    """bytes -> DataFrame 正常读入。"""
    csv_bytes = df_basic.to_csv(index=False).encode("utf-8")
    df = app.load_csv_bytes(csv_bytes)
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), df_basic.reset_index(drop=True)
    )


def test_load_csv_bytes_empty_raise() -> None:
    """空 bytes 触发 ValueError。"""
    with pytest.raises(ValueError):
        app.load_csv_bytes(b"")


# ---------- validate_columns_exist ----------


def test_validate_columns_exist_ok(df_basic: pd.DataFrame) -> None:
    """所需列都存在则不报错。"""
    app.validate_columns_exist(df_basic, ["time", "temperature"])


def test_validate_columns_exist_missing(df_basic: pd.DataFrame) -> None:
    """缺列时应抛 ValueError。"""
    with pytest.raises(ValueError):
        app.validate_columns_exist(df_basic, ["time", "temperature", "pressure"])


# ---------- clean_dataframe ----------


def test_clean_dataframe_dropna(df_with_nan: pd.DataFrame) -> None:
    """dropna=True 时，结果中应无 NaN。"""
    cleaned = app.clean_dataframe(df_with_nan, dropna=True)
    assert cleaned.isna().sum().sum() == 0
    # 同时保留原有列
    app.validate_columns_exist(cleaned, ["time", "temperature"])


def test_clean_dataframe_keepna(df_with_nan: pd.DataFrame) -> None:
    """dropna=False 时，保留 NaN，但仍会去重。"""
    kept = app.clean_dataframe(df_with_nan, dropna=False)
    # 仍包含 NaN
    assert kept["temperature"].isna().any()
    # 列仍然存在
    app.validate_columns_exist(kept, ["time", "temperature"])
