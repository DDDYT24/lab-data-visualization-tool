import io
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def validate_columns_exist(df: pd.DataFrame, cols: List[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return False
    return True


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> List[str]:
    non_numeric = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() >= 0.8:
                df[c] = converted
            else:
                non_numeric.append(c)
    return non_numeric


def coerce_datetime_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        try:
            dt = pd.to_datetime(df[c], errors="raise")
            df[c] = dt.view("int64")
        except Exception:
            pass
    return df


@st.cache_resource(show_spinner=False)
def get_linreg_resources():
    from sklearn import metrics
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    return train_test_split, LinearRegression, metrics


def main():
    st.set_page_config(page_title="Lab Data Visualization Tool", layout="wide")
    st.title("🔬 Lab Data Visualization Tool")
    with st.sidebar:
        st.header("About")
        st.write("Clean • Visualize • Predict experimental datasets.")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV file to get started.")
        st.stop()

    try:
        df_raw = load_csv_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df_raw.head(), use_container_width=True)

    tab_clean, tab_predict = st.tabs(["🧹 Clean & Visualize", "🤖 Predict"])

    with tab_clean:
        keep_cols = st.multiselect("Columns to keep", df_raw.columns.tolist(), default=df_raw.columns.tolist())
        if not keep_cols:
            st.warning("Select at least one column.")
            st.stop()
        df = df_raw[keep_cols].copy()
        if st.checkbox("Drop rows with missing values"):
            before = len(df)
            df = df.dropna()
            st.info(f"Dropped {before - len(df)} rows. Remaining: {len(df)}")
        with st.expander("Describe (after cleaning)"):
            st.write(df.describe(include="all"))

        st.markdown("### Visualization")
        if len(df.columns) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_col = st.selectbox("X-axis", df.columns, key="viz_x")
            with c2:
                y_cols = st.multiselect("Y-axis (one or more)", [c for c in df.columns if c != x_col], key="viz_y")
            kind = st.radio("Plot type", ["line", "scatter", "bar"], horizontal=True, index=0)
            title = st.text_input("Plot title", "")
            if st.button("Generate Plot", type="primary"):
                if not validate_columns_exist(df, [x_col] + y_cols):
                    st.stop()
                non_num = ensure_numeric(df, y_cols)
                if non_num:
                    st.warning(f"Skipped non-numeric Y columns: {non_num}")
                    y_cols = [c for c in y_cols if c not in non_num]
                if not y_cols:
                    st.error("No valid numeric Y columns to plot.")
                    st.stop()
                fig, ax = plt.subplots(figsize=(7, 4))
                for y in y_cols:
                    if kind == "line":
                        ax.plot(df[x_col], df[y], label=y)
                    elif kind == "scatter":
                        ax.scatter(df[x_col], df[y], label=y)
                    else:
                        ax.bar(df[x_col], df[y], label=y)
                ax.set_xlabel(x_col)
                ax.set_ylabel(", ".join(y_cols))
                if title:
                    ax.set_title(title)
                ax.legend()
                st.pyplot(fig)
                st.download_button("Download Plot (PNG)", data=fig_to_png(fig), file_name="plot.png", mime="image/png")

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Cleaned CSV", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")

    with tab_predict:
        all_cols = df_raw.columns.tolist()
        if not all_cols:
            st.error("No columns found.")
            st.stop()
        target = st.selectbox("Target (y)", all_cols, index=len(all_cols) - 1)
        features = st.multiselect("Features (X)", [c for c in all_cols if c != target])
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42, step=1)
        if st.button("Train model", type="primary"):
            train_test_split, LinearRegression, metrics = get_linreg_resources()
            if not features:
                st.error("Please select at least one feature.")
                st.stop()
            need_cols = [target] + features
            if not validate_columns_exist(df_raw, need_cols):
                st.stop()
            df_model = df_raw[need_cols].copy()
            df_model = coerce_datetime_to_numeric(df_model, features)
            X = pd.get_dummies(df_model[features], drop_first=True)
            y = df_model[target]
            data = pd.concat([X, y], axis=1).dropna()
            X = data[X.columns]
            y = data[target]
            if len(X) < 5:
                st.error("Not enough rows after cleaning for training (need ≥ 5).")
                st.stop()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=float(test_size), random_state=int(random_state)
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = metrics.r2_score(y_test, y_pred)
            st.markdown("### Metrics")
            st.write(f"- MAE: {mae:.4f}")
            st.write(f"- RMSE: {rmse:.4f}")
            st.write(f"- R²: {r2:.4f}")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(y_test, y_pred)
            vmin = float(min(np.min(y_test), np.min(y_pred)))
            vmax = float(max(np.max(y_test), np.max(y_pred)))
            ax2.plot([vmin, vmax], [vmin, vmax])
            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")
            ax2.set_title("Prediction vs Actual")
            st.pyplot(fig2)
            st.download_button(
                "Download Pred-Actual Plot (PNG)", data=fig_to_png(fig2), file_name="pred_vs_actual.png", mime="image/png"
            )
            out = pd.DataFrame({"y_actual": y_test.to_numpy().ravel(), "y_pred": y_pred.ravel()})
            st.download_button(
                "Download Predictions (CSV)",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )


__all__ = [
    "validate_columns_exist",
    "ensure_numeric",
    "coerce_datetime_to_numeric",
    "load_csv_bytes",
    "fig_to_png",
    "get_linreg_resources",
]

if __name__ == "__main__":
    main()
