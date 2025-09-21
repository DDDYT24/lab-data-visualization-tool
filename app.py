from __future__ import annotations

import io
from typing import Iterable

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    if data is None or len(data) == 0:
        raise ValueError("No CSV bytes provided")
    return pd.read_csv(io.BytesIO(data))


def validate_columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataframe(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates()
    if dropna:
        out = out.dropna()
    return out


def _plot(df: pd.DataFrame, x: str, ys: list[str], kind: str) -> None:
    fig, ax = plt.subplots()
    if kind == "line":
        for y in ys:
            ax.plot(df[x], df[y], label=y)
    elif kind == "scatter":
        for y in ys:
            ax.scatter(df[x], df[y], label=y)
    elif kind == "bar":
        for y in ys:
            ax.bar(df[x], df[y], label=y)
    ax.set_xlabel(x)
    ax.set_ylabel(", ".join(ys))
    ax.legend()
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Lab Data Visualization Tool", layout="wide")
    st.title("Lab Data Visualization Tool")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV file to get started.")
        return

    df_raw = load_csv_bytes(uploaded.read())
    st.success(f"Loaded shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    with st.expander("Preview raw"):
        st.dataframe(df_raw.head(20))

    # Cleaning
    dropna = st.checkbox("Drop rows containing NaN", value=True)
    df_clean = clean_dataframe(df_raw, dropna=dropna)

    with st.expander("Preview cleaned"):
        st.dataframe(df_clean.head(20))
        st.download_button(
            "Download cleaned CSV",
            df_clean.to_csv(index=False).encode("utf-8"),
            file_name="cleaned.csv",
            mime="text/csv",
        )

    # Visualization
    st.subheader("Visualization")
    if df_clean.empty:
        st.warning("No data to visualize after cleaning.")
        return

    cols = list(df_clean.columns)
    x = st.selectbox("X-axis", options=cols, index=0)
    y_multi = st.multiselect("Y-axis (one or more)", options=[c for c in cols if c != x])
    kind = st.radio("Plot type", options=["line", "scatter", "bar"], index=0)
    title = st.text_input("Plot title", value="")

    if st.button("Generate Plot") and y_multi:
        st.write(f"Selected: x='{x}', y={y_multi}, type='{kind}'")
        _plot(df_clean, x, list(y_multi), kind)
        if title:
            st.markdown(f"**{title}**")


if __name__ == "__main__":
    main()
