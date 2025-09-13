import io
import pandas as pd

def load_csv_bytes(data: bytes) -> pd.DataFrame:
    """
    Read CSV from raw bytes into a DataFrame.

    Raises:
        ValueError: if data is None or empty.
        pandas.errors.ParserError: if the CSV cannot be parsed.
    """
    if data is None or len(data) == 0:
        raise ValueError("No CSV bytes provided")
    return pd.read_csv(io.BytesIO(data))

import os
from typing import Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def validate_columns_exist(df: pd.DataFrame, required_cols: Iterable[str]) -> bool:
    required = set(required_cols)
    present = set(df.columns)
    missing = required - present
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return True


def clean_dataframe(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    out = df.dropna(axis=1, how="all")
    if dropna:
        out = out.dropna().reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def _read_csv_flexible(
    source: Union[str, os.PathLike, bytes, io.BytesIO, io.StringIO, "st.runtime.uploaded_file_manager.UploadedFile"]
) -> pd.DataFrame:
    if source is None:
        raise ValueError("No CSV source provided")
    if hasattr(source, "read"):
        return pd.read_csv(source)
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source))
    return pd.read_csv(source)


def _make_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    kind: str = "line",
    title: Optional[str] = None,
):
    fig, ax = plt.subplots()
    if kind == "line":
        for y in y_cols:
            ax.plot(df[x_col], df[y], label=y)
    elif kind == "scatter":
        for y in y_cols:
            ax.scatter(df[x_col], df[y], label=y)
    elif kind == "bar":
        if len(y_cols) == 1:
            ax.bar(df[x_col], df[y_cols[0]])
        else:
            width = 0.8 / len(y_cols)
            x_vals = pd.Series(range(len(df[x_col])))
            for i, y in enumerate(y_cols):
                ax.bar(x_vals + i * width, df[y], width=width, label=y)
            ax.set_xticks(x_vals + (len(y_cols) - 1) * width / 2)
            ax.set_xticklabels(df[x_col])
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="Lab Data Visualization Tool", layout="wide")
    st.title("Lab Data Visualization Tool")
    st.caption("Upload → Clean → Visualize → (optional) Predict")

    with st.sidebar:
        st.header("Options")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV file to get started.")
        return

    try:
        df_raw = _read_csv_flexible(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.subheader("Preview")
    st.dataframe(df_raw.head())

    st.subheader("Cleaning")
    dropna = st.checkbox("Drop rows with missing values", value=True)
    try:
        df_clean = clean_dataframe(df_raw, dropna=dropna)
    except Exception as e:
        st.error(f"Cleaning failed: {e}")
        return

    st.write(f"Rows: {len(df_clean)}, Columns: {list(df_clean.columns)}")

    buf = io.StringIO()
    df_clean.to_csv(buf, index=False)
    st.download_button(
        "Download cleaned CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="cleaned.csv",
        mime="text/csv",
    )

    if len(df_clean.columns) >= 2:
        st.subheader("Visualization")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", df_clean.columns, index=0)
        with col2:
            y_cols = st.multiselect("Y-axis (one or more)", df_clean.columns.difference([x_col]).tolist())
        kind = st.radio("Plot type", options=["line", "scatter", "bar"], horizontal=True)
        title = st.text_input("Plot title", value="")
        if st.button("Generate Plot") and y_cols:
            fig = _make_plot(df_clean, x_col, y_cols, kind=kind, title=title or None)
            st.pyplot(fig)
    else:
        st.info("Need at least two columns to plot.")


if __name__ == "__main__":
    main()
