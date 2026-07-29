from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import app
from labviz.core import DataSummary
from labviz.database import HistoryStore
from labviz.plotting import create_figure, figure_to_bytes, sample_dataframe, save_figure


@pytest.fixture
def plot_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "temperature": [20.0, 21.0, 19.0, 22.0, 20.0],
            "pressure": [1.0, 1.1, 1.2, 1.0, 0.9],
            "sample": ["a", "a", "b", "b", "c"],
        }
    )


@pytest.mark.parametrize(
    ("kind", "x", "ys"),
    [
        ("line", "time", ["temperature"]),
        ("scatter", "time", ["temperature", "pressure"]),
        ("bar", "sample", ["temperature"]),
        ("histogram", None, ["temperature"]),
        ("box", None, ["temperature", "pressure"]),
        ("heatmap", None, ["time", "temperature", "pressure"]),
    ],
)
def test_create_standard_figures(
    plot_frame: pd.DataFrame, kind: str, x: str | None, ys: list[str]
) -> None:
    figure = create_figure(plot_frame, kind=kind, x=x, ys=ys, title="Experiment")
    assert figure.axes
    assert figure_to_bytes(figure).startswith(b"\x89PNG")


def test_create_surface_and_save(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "x": [0, 0, 1, 1, 0.5],
            "y": [0, 1, 0, 1, 0.5],
            "z": [0, 1, 1, 2, 1.2],
        }
    )
    figure = create_figure(frame, kind="surface3d", x="x", ys=["y", "z"])
    target = save_figure(figure, tmp_path / "surface")
    assert target.suffix == ".png"
    assert target.stat().st_size > 1_000


def test_plot_validation(plot_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Numeric"):
        create_figure(plot_frame, kind="line", x="time", ys=["sample"])
    with pytest.raises(ValueError, match="at least two"):
        create_figure(plot_frame, kind="heatmap", x=None, ys=["temperature"])
    with pytest.raises(ValueError, match="one of"):
        create_figure(plot_frame, kind="pie", x=None, ys=["temperature"])
    with pytest.raises(ValueError, match="no complete"):
        empty = pd.DataFrame({"x": [1.0], "y": pd.Series([float("nan")], dtype=float)})
        create_figure(empty, kind="line", x="x", ys=["y"])


def test_sampling_is_bounded_and_deterministic() -> None:
    frame = pd.DataFrame({"x": range(10_000)})
    sampled = sample_dataframe(frame, 100)
    assert len(sampled) == 100
    assert sampled.iloc[0, 0] == 0
    assert sampled.iloc[-1, 0] == 9_999
    pd.testing.assert_frame_equal(sampled, sample_dataframe(frame, 100))


def test_history_store_round_trip_and_clear(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    summary = DataSummary(5, 3, 3, 0, 1, 0, 500)
    row_id = store.record(
        filename="private-experiment.csv",
        file_hash="abc123",
        file_size=100,
        summary=summary,
        chart_type="line",
        x_column="time",
        y_columns=["temperature"],
    )
    records = store.recent()
    assert records[0].id == row_id
    assert records[0].y_columns == ("temperature",)
    assert store.clear() == 1
    assert store.recent() == []


def test_history_store_validates_limit(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "history.db")
    with pytest.raises(ValueError):
        store.recent(0)


def test_app_compatibility_helpers() -> None:
    frame = app.load_csv_bytes(b"time,value\n0,1\n1,\n")
    app.validate_columns_exist(frame, ["time", "value"])
    assert len(app.clean_dataframe(frame, dropna=True)) == 1
