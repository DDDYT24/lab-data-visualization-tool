"""Streamlit front end for Lab Data Visualization Tool."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from labviz.core import (
    MAX_UPLOAD_BYTES,
    DataSummary,
    MissingStrategy,
    column_profile,
    load_data_bytes,
    summarize_dataframe,
)
from labviz.core import (
    clean_dataframe as clean_table,
)
from labviz.core import (
    validate_columns_exist as validate_required_columns,
)
from labviz.database import HistoryRecord, HistoryStore
from labviz.plotting import MAX_PLOT_POINTS, create_figure, figure_to_bytes

FILE_TYPES = ["csv", "tsv", "txt", "json", "xlsx"]
CHART_LABELS = {
    "Line": "line",
    "Scatter": "scatter",
    "Bar": "bar",
    "Histogram": "histogram",
    "Box": "box",
    "Correlation heatmap": "heatmap",
    "3D surface": "surface3d",
}
MISSING_LABELS: dict[str, MissingStrategy] = {
    "Drop incomplete rows": "drop",
    "Keep missing values": "keep",
    "Forward fill": "ffill",
    "Backward fill": "bfill",
    "Fill numeric values with mean": "mean",
    "Fill numeric values with median": "median",
}


# Compatibility helpers kept for existing imports.
def load_csv_bytes(data: bytes) -> pd.DataFrame:
    return load_data_bytes(data, "upload.csv")


def validate_columns_exist(frame: pd.DataFrame, required: Iterable[str]) -> None:
    validate_required_columns(frame, list(required))


def clean_dataframe(frame: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    return clean_table(frame, missing="drop" if dropna else "keep")


@st.cache_data(show_spinner=False, max_entries=4)
def _load_cached(data: bytes, filename: str) -> pd.DataFrame:
    return load_data_bytes(data, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def _clean_cached(
    frame: pd.DataFrame, drop_duplicates: bool, missing_strategy: MissingStrategy
) -> pd.DataFrame:
    return clean_table(
        frame,
        drop_duplicates=drop_duplicates,
        missing=missing_strategy,
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _summary_cached(frame: pd.DataFrame) -> DataSummary:
    return summarize_dataframe(frame)


@st.cache_data(show_spinner=False, max_entries=8)
def _profile_cached(frame: pd.DataFrame) -> pd.DataFrame:
    return column_profile(frame)


@st.cache_data(show_spinner=False, max_entries=8)
def _csv_cached(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


@st.cache_resource(show_spinner=False)
def _history_store(path: str) -> HistoryStore:
    return HistoryStore(Path(path))


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _metric_row(frame: pd.DataFrame) -> None:
    summary = _summary_cached(frame)
    columns = st.columns(5)
    columns[0].metric("Rows", f"{summary.rows:,}")
    columns[1].metric("Columns", summary.columns)
    columns[2].metric("Missing", f"{summary.missing_cells:,}")
    columns[3].metric("Duplicates", f"{summary.duplicate_rows:,}")
    columns[4].metric("Memory", _format_bytes(summary.memory_bytes))


def _history_frame(records: Iterable[HistoryRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Time (UTC)": record.created_at,
                "File": record.filename,
                "Rows": record.row_count,
                "Columns": record.column_count,
                "Chart": record.chart_type,
                "X": record.x_column or "—",
                "Values": ", ".join(record.y_columns),
            }
            for record in records
        ]
    )


def _render_history(store: HistoryStore) -> None:
    records = store.recent()
    if not records:
        st.info("No plot history yet. Raw experiment data is never stored in the database.")
        return
    st.dataframe(_history_frame(records), hide_index=True, width="stretch")
    if st.button("Clear history", type="secondary"):
        removed = store.clear()
        st.success(f"Removed {removed} history record(s).")
        st.rerun()


def _chart_columns(frame: pd.DataFrame, kind: str) -> tuple[str | None, list[str]] | None:
    numeric = list(frame.select_dtypes(include="number").columns)
    if not numeric:
        st.warning("This dataset has no numeric columns to visualize.")
        return None

    if kind in {"line", "scatter", "bar"}:
        x_column = st.selectbox("X axis", list(frame.columns))
        candidates = [column for column in numeric if column != x_column]
        defaults = candidates[: min(2, len(candidates))]
        values = st.multiselect("Value column(s)", candidates, default=defaults)
        return str(x_column), values

    if kind in {"histogram", "box"}:
        values = st.multiselect("Numeric column(s)", numeric, default=numeric[:1])
        return None, values

    if kind == "heatmap":
        values = st.multiselect(
            "Columns for correlation", numeric, default=numeric[: min(6, len(numeric))]
        )
        return None, values

    if len(numeric) < 3:
        st.warning("A 3D surface needs at least three numeric columns.")
        return None
    x_column = st.selectbox("X coordinate", numeric, index=0)
    y_column = st.selectbox("Y coordinate", numeric, index=1)
    z_column = st.selectbox("Z value", numeric, index=2)
    return x_column, [y_column, z_column]


def _render_visualization(
    frame: pd.DataFrame,
    *,
    filename: str,
    file_hash: str,
    file_size: int,
    store: HistoryStore,
) -> None:
    label = st.selectbox("Chart type", list(CHART_LABELS))
    kind = CHART_LABELS[label]
    selection = _chart_columns(frame, kind)
    title = st.text_input("Title", placeholder="Optional figure title")
    max_points = st.slider(
        "Maximum rendered points",
        min_value=1_000,
        max_value=50_000,
        value=MAX_PLOT_POINTS,
        step=1_000,
        help=(
            "Large tables are sampled evenly for faster rendering; "
            "exported data remains complete."
        ),
    )

    if st.button("Generate visualization", type="primary", disabled=selection is None):
        assert selection is not None
        x_column, y_columns = selection
        try:
            figure = create_figure(
                frame,
                kind=kind,
                x=x_column,
                ys=y_columns,
                title=title.strip(),
                max_points=max_points,
            )
            image = figure_to_bytes(figure)
            figure.clear()
            st.session_state["plot_image"] = image
            st.session_state["plot_filename"] = f"{Path(filename).stem}_{kind}.png"
            st.session_state["plot_hash"] = file_hash
            store.record(
                filename=filename,
                file_hash=file_hash,
                file_size=file_size,
                summary=_summary_cached(frame),
                chart_type=kind,
                x_column=x_column,
                y_columns=y_columns,
            )
        except ValueError as exc:
            st.error(str(exc))

    if st.session_state.get("plot_hash") == file_hash and "plot_image" in st.session_state:
        st.image(st.session_state["plot_image"], width="stretch")
        st.download_button(
            "Download PNG",
            data=st.session_state["plot_image"],
            file_name=st.session_state["plot_filename"],
            mime="image/png",
        )
        render_limit = min(max_points, 500) if kind == "bar" else max_points
        if len(frame) > render_limit:
            st.caption(
                f"Rendered an evenly spaced sample of up to {render_limit:,} rows for performance."
            )


def main() -> None:
    st.set_page_config(page_title="Lab Data Visualization", page_icon="🔬", layout="wide")
    st.title("Lab Data Visualization")
    st.caption("Load, inspect, clean, visualize, and export experimental tables locally.")

    database_path = os.environ.get("LABVIZ_DB_PATH", ".labviz/history.db")
    store = _history_store(database_path)
    uploaded = st.sidebar.file_uploader(
        "Experimental data",
        type=FILE_TYPES,
        help=f"CSV, TSV, TXT, JSON, or XLSX up to {MAX_UPLOAD_BYTES // 1_048_576} MB.",
    )

    if uploaded is None:
        st.info("Upload a data file from the sidebar to begin.")
        with st.expander("Recent plot history", expanded=True):
            _render_history(store)
        return

    payload = uploaded.getvalue()
    file_hash = hashlib.sha256(payload).hexdigest()
    try:
        raw = _load_cached(payload, uploaded.name)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.sidebar.subheader("Cleaning")
    drop_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=True)
    missing_label = st.sidebar.selectbox("Missing values", list(MISSING_LABELS))
    cleaned = _clean_cached(raw, drop_duplicates, MISSING_LABELS[missing_label])

    st.subheader(uploaded.name)
    _metric_row(cleaned)
    if cleaned.empty:
        st.warning("Cleaning removed every row. Change the cleaning options in the sidebar.")

    data_tab, chart_tab, history_tab = st.tabs(["Data quality", "Visualization", "History"])
    with data_tab:
        before, after = _summary_cached(raw), _summary_cached(cleaned)
        st.caption(
            f"Cleaning result: {before.rows:,} → {after.rows:,} rows; "
            f"{before.missing_cells:,} → {after.missing_cells:,} missing cells."
        )
        st.dataframe(cleaned.head(200), hide_index=True, width="stretch")
        st.caption("Preview is limited to 200 rows; downloads contain the complete cleaned table.")
        st.download_button(
            "Download cleaned CSV",
            data=_csv_cached(cleaned),
            file_name=f"{Path(uploaded.name).stem}_cleaned.csv",
            mime="text/csv",
        )
        with st.expander("Column profile"):
            st.dataframe(_profile_cached(cleaned), hide_index=True, width="stretch")
        with st.expander("Machine-readable summary"):
            st.json(asdict(after))

    with chart_tab:
        if cleaned.empty:
            st.info("No cleaned rows are available to plot.")
        else:
            _render_visualization(
                cleaned,
                filename=uploaded.name,
                file_hash=file_hash,
                file_size=len(payload),
                store=store,
            )

    with history_tab:
        _render_history(store)


if __name__ == "__main__":
    main()
