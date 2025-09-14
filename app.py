# app.py
from typing import Iterable
import io
import pandas as pd
import streamlit as st


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    if data is None or len(data) == 0:
        raise ValueError("No CSV bytes provided")
    return pd.read_csv(io.BytesIO(data))


def validate_columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataframe(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    out = df.drop_duplicates()
    if dropna:
        out = out.dropna()
    return out


def main() -> None:
    st.set_page_config(page_title="Lab Data Visualization Tool", layout="wide")
    st.title("Lab Data Visualization Tool")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to get started.")
        return

    df_raw = pd.read_csv(uploaded)
    st.subheader("Raw preview")
    st.dataframe(df_raw.head())

    required = st.text_input("Required columns (comma-separated)", "").strip()
    if required:
        cols = [c.strip() for c in required.split(",") if c.strip()]
        try:
            validate_columns_exist(df_raw, cols)
            st.success("All required columns are present.")
        except ValueError as e:
            st.error(str(e))

    drop_na = st.checkbox("Drop rows with missing values", value=True)
    df_clean = clean_dataframe(df_raw, dropna=drop_na)

    st.subheader("Cleaned preview")
    st.dataframe(df_clean.head())


if __name__ == "__main__":
    main()
