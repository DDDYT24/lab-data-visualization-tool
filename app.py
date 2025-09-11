import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

st.set_page_config(page_title="Lab Data Visualization Tool", layout="wide")
st.title("🔬 Lab Data Visualization Tool")
st.write("Day 6–7: Interactive Web App (Clean • Visualize • Predict)")

# ---------- helpers ----------
def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def coerce_datetime_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """将可解析的日期列转为整数时间戳（纳秒），便于建模。"""
    df = df.copy()
    for c in cols:
        try:
            dt = pd.to_datetime(df[c], errors="raise")
            df[c] = dt.view("int64")  # ns since epoch
        except Exception:
            pass
    return df

# ---------- file upload ----------
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

df_raw = pd.read_csv(uploaded)
st.subheader(" Preview")
st.dataframe(df_raw.head(), use_container_width=True)

tabs = st.tabs([" Clean & Visualize", " Predict (Linear Regression)"])

# =================== Tab 1: Clean & Visualize ===================
with tabs[0]:
    st.markdown("### Column selection & cleaning")

    keep_cols = st.multiselect("Columns to keep", df_raw.columns.tolist(), default=df_raw.columns.tolist())
    df = df_raw[keep_cols].copy()

    dropna = st.checkbox("Drop rows with missing values")
    if dropna:
        before = len(df)
        df = df.dropna()
        st.info(f"Dropped {before - len(df)} rows. Remaining: {len(df)}")

    st.markdown("#### Describe")
    with st.expander("Show descriptive statistics"):
        st.write(df.describe(include="all"))

    st.markdown("### Visualization")
    if len(df.columns) >= 2:
        left, right = st.columns([1,1])
        with left:
            x_col = st.selectbox("X-axis", df.columns)
        with right:
            y_cols = st.multiselect("Y-axis (one or more)", [c for c in df.columns if c != x_col])

        kind = st.radio("Plot type", ["line", "scatter", "bar"], horizontal=True)
        title = st.text_input("Plot title", "")

        if st.button("Generate Plot", type="primary"):
            fig, ax = plt.subplots(figsize=(7, 4))
            for y in y_cols:
                if kind == "line":
                    ax.plot(df[x_col], df[y], label=y)
                elif kind == "scatter":
                    ax.scatter(df[x_col], df[y], label=y)
                else:
                    ax.bar(df[x_col], df[y], label=y)
            ax.set_xlabel(x_col)
            ax.set_ylabel(", ".join(y_cols) if y_cols else "")
            if title:
                ax.set_title(title)
            ax.legend()
            st.pyplot(fig)
            st.download_button("Download Plot (PNG)", data=fig_to_png(fig), file_name="plot.png", mime="image/png")

    # 导出清洗后的 CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Cleaned CSV", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")

# =================== Tab 2: Predict ===================
with tabs[1]:
    st.markdown("### Choose target and features")
    all_cols = df_raw.columns.tolist()
    target = st.selectbox("Target (y)", all_cols, index=len(all_cols)-1 if all_cols else 0)
    features = st.multiselect("Features (X)", [c for c in all_cols if c != target])

    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random seed", value=42, step=1)

    if st.button("Train model", type="primary", use_container_width=False, help="Fit a linear regression model"):
        if not features:
            st.error("Please select at least one feature column.")
            st.stop()

        # 准备数据：选择列、丢 NA、处理时间与非数值列
        df_model = df_raw[[target] + features].copy()
        df_model = coerce_datetime_to_numeric(df_model, cols=features)  # 将可解析时间列转换为数值
        # 将非数值特征做 one-hot
        X = pd.get_dummies(df_model[features], drop_first=True)
        y = df_model[target]

        # 丢弃 y 或 X 中的缺失
        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data[target]

        if len(X) < 5:
            st.error("Not enough rows after cleaning for training (need ≥ 5).")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=int(random_state))
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = metrics.mean_absolute_error(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        r2   = metrics.r2_score(y_test, y_pred)


        st.markdown("### Metrics")
        st.write(f"- **MAE**: {mae:.4f}")
        st.write(f"- **RMSE**: {rmse:.4f}")
        st.write(f"- **R²**: {r2:.4f}")

        st.markdown("### Pred vs. Actual")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.scatter(y_test, y_pred)
        # 45° reference line
        min_v = float(min(y_test.min(), y_pred.min()))
        max_v = float(max(y_test.max(), y_pred.max()))
        ax2.plot([min_v, max_v], [min_v, max_v])
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_title("Prediction vs. Actual")
        st.pyplot(fig2)
        st.download_button("Download Pred-Actual Plot (PNG)",
                           data=fig_to_png(fig2), file_name="pred_vs_actual.png", mime="image/png")

        # 预测结果导出
        out = pd.DataFrame({"y_actual": y_test.to_numpy().ravel(),
                            "y_pred":   y_pred.ravel()})
        st.download_button("Download Predictions (CSV)",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv",
                           mime="text/csv")

        st.success("Model trained successfully!")
