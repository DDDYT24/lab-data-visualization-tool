from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from labviz.cli import clean_dataframe, load_csv, plot_df, validate_columns_exist


def _make_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "time": [0, 1, 1, 2, 3],
            "temperature": [20.0, 21.0, 21.0, None, 22.0],
            "humidity": [40, 42, 42, 43, 44],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


def test_load_and_validate(tmp_path: Path) -> None:
    p = _make_csv(tmp_path)
    df = load_csv(p)
    assert list(df.columns) == ["time", "temperature", "humidity"]
    validate_columns_exist(df, ["time", "temperature"])  # no raise


def test_validate_missing(tmp_path: Path) -> None:
    p = _make_csv(tmp_path)
    df = load_csv(p)
    with pytest.raises(ValueError):
        validate_columns_exist(df, ["time", "pressure"])  # pressure not exist


def test_clean_dataframe_dropna_and_method(tmp_path: Path) -> None:
    p = _make_csv(tmp_path)
    df = load_csv(p)
    # 去重 + 前向填充 + 丢弃 NaN
    cleaned = clean_dataframe(df, dropna=True, method="ffill")
    assert cleaned.isna().sum().sum() == 0
    # 原本有重复 time=1 行，应至少去掉一行
    assert len(cleaned) <= len(df) - 1


def test_plot_df(tmp_path: Path) -> None:
    p = _make_csv(tmp_path)
    df = load_csv(p)
    out = tmp_path / "plot"
    out_img = plot_df(df, x="time", ys=["temperature"], kind="line", outpath=out, show=False)
    assert out_img.exists()
    assert out_img.suffix == ".png"
