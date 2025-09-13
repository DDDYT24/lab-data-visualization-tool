from __future__ import annotations

import io
from typing import Iterable, List, Optional

import pandas as pd


def load_csv_bytes(data: bytes) -> pd.DataFrame:
    """
    将原始 CSV 字节读取为 DataFrame。
    若 data 为空或 None，抛 ValueError。
    """
    if data is None or len(data) == 0:
        raise ValueError("No CSV bytes provided")
    return pd.read_csv(io.BytesIO(data))


def validate_columns_exist(df: pd.DataFrame, required: Iterable[str]) -> None:
    """
    校验 df 是否包含 required 中的所有列。
    若缺少则抛 ValueError，并给出缺失列表。
    """
    required_list = list(required)
    missing = [c for c in required_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_dataframe(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    基础清洗：去重 + 可选丢弃 NA。
    始终返回一个新 DataFrame，不在原地修改。
    """
    out = df.copy()
    out = out.drop_duplicates()
    if dropna:
        out = out.dropna()
    return out

def _try_plot(
    df: pd.DataFrame,
    kind: str,
    x: Optional[str],
    y_cols: List[str],
    title: str = "",
) -> None:
    """在 Streamlit 环境下绘图（延迟导入以避免导入时的 UI 执行）"""
    import streamlit as st
    import matplotlib.pyplot as plt

    if df.empty or not y_cols:
        st.info("请选择要绘制的列。")
        return

    # 处理 X 轴
    x_axis = None
    if x and x in df.columns:
        x_axis = df[x]
    else:
        x_axis = df.index

    fig, ax = plt.subplots()
    for col in y_cols:
        if col not in df.columns:
            continue
        if kind == "line":
            ax.plot(x_axis, df[col], label=col)
        elif kind == "scatter":
            ax.scatter(x_axis, df[col], label=col)
        elif kind == "bar":
            ax.bar(x_axis, df[col], label=col)

    ax.set_title(title or "Plot")
    ax.legend(loc="best")
    st.pyplot(fig)



def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Lab Data Visualization Tool", layout="wide")
    st.title("Lab Data Visualization Tool")
    st.caption("Upload → Validate → Clean → Visualize / Download")

    st.subheader("1) Upload CSV")
    up = st.file_uploader("Drag & drop or Browse… (CSV)", type=["csv"])
    raw_df: Optional[pd.DataFrame] = None

    if up is not None:
        try:
            raw_bytes = up.read()
            raw_df = load_csv_bytes(raw_bytes)
            st.success(f"Loaded CSV with shape: {raw_df.shape}")
            with st.expander("Preview (head)"):
                st.dataframe(raw_df.head(20), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to read CSV: {e}")

    st.divider()
    st.subheader("2) Validate required columns (optional)")
    required_cols = st.text_input("Required columns (comma-separated)", value="time,temperature")
    required_list = [c.strip() for c in required_cols.split(",") if c.strip()]

    if st.button("Validate") and raw_df is not None:
        try:
            validate_columns_exist(raw_df, required_list)
            st.success("All required columns exist ")
        except Exception as e:  # noqa: BLE001
            st.error(f"Validation failed: {e}")

    st.divider()
    st.subheader("3) Clean data")
    dropna = st.checkbox("Drop NA rows", value=True)
    cleaned_df: Optional[pd.DataFrame] = None
    if raw_df is not None:
        try:
            cleaned_df = clean_dataframe(raw_df, dropna=dropna)
            st.success(f"Cleaned shape: {cleaned_df.shape}")
            with st.expander("Cleaned Preview (head)"):
                st.dataframe(cleaned_df.head(20), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Clean failed: {e}")

    st.divider()
    st.subheader("4) Visualize")
    if cleaned_df is not None and not cleaned_df.empty:
        cols = list(cleaned_df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("X axis", ["<index>"] + cols, index=0)
            x_sel = None if x_col == "<index>" else x_col
        with c2:
            y_cols = st.multiselect("Y axis (one or more)", cols, default=[cols[1]] if len(cols) > 1 else cols[:1])
        with c3:
            kind = st.radio("Plot type", options=["line", "scatter", "bar"], horizontal=True, index=0)

        title = st.text_input("Plot title", value="")
        if st.button("Generate Plot"):
            _try_plot(cleaned_df, kind, x_sel, y_cols, title)

    st.divider()
    st.subheader("5) Download cleaned CSV")
    if cleaned_df is not None and not cleaned_df.empty:
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download cleaned.csv",
            data=csv_bytes,
            file_name="cleaned.csv",
            mime="text/csv",
        )


# 仅在直接运行 `streamlit run app.py` 或 `python app.py` 时执行 UI
if __name__ == "__main__":
    main()
