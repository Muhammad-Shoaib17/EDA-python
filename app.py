import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="📊 Simple EDA Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Simple EDA Dashboard")
st.caption("Upload a CSV or Excel file to instantly explore your data — Power BI style layout.")

# -------------------------------
# Helper
# -------------------------------
@st.cache_data(show_spinner=False)
def load_file(file):
    name = getattr(file, "name", "")
    suffix = os.path.splitext(name)[1].lower()
    if suffix in [".csv", ".txt"]:
        try:
            return pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1")
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file, engine="openpyxl")
    else:
        raise RuntimeError("Unsupported file type. Please upload CSV or Excel.")

# -------------------------------
# File Upload
# -------------------------------
Ufile = st.file_uploader("📂 Upload your dataset", type=["csv", "txt", "xlsx", "xls"])

# -------------------------------
# Main EDA
# -------------------------------
if Ufile:
    try:
        df = load_file(Ufile)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("📋 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Total Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("🧮 Summary Statistics")
    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    if not num_df.empty:
        st.markdown("**Numeric Columns**")
        st.dataframe(num_df.describe().T, use_container_width=True)
    if not cat_df.empty:
        st.markdown("**Categorical Columns (Unique Counts)**")
        st.dataframe(cat_df.nunique().sort_values(ascending=False).to_frame("Unique Values"))

    st.subheader("❌ Missing Values")
    na_table = df.isna().sum().to_frame("Missing Count")
    na_table["Missing %"] = (na_table["Missing Count"] / len(df) * 100).round(2)
    st.dataframe(na_table, use_container_width=True)

    # -------------------------------
    # Numeric Histograms (2 per row)
    # -------------------------------
    if not num_df.empty:
        st.subheader("📊 Numeric Distributions")
        num_cols = num_df.columns.tolist()
        for i in range(0, len(num_cols), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(num_cols):
                    colname = num_cols[i + j]
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4,3))
                        num_df[colname].dropna().hist(ax=ax, bins=25)
                        ax.set_title(colname)
                        st.pyplot(fig, clear_figure=True)

    # -------------------------------
    # Categorical Bar Charts (3 per row)
    # -------------------------------
    if not cat_df.empty:
        st.subheader("📈 Top Categories")
        cat_cols = cat_df.columns.tolist()
        for i in range(0, len(cat_cols), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(cat_cols):
                    colname = cat_cols[i + j]
                    with cols[j]:
                        vc = cat_df[colname].astype(str).value_counts().head(8)
                        fig, ax = plt.subplots(figsize=(4,3))
                        vc.plot(kind="bar", ax=ax)
                        ax.set_title(colname)
                        ax.tick_params(axis='x', rotation=45)
                        st.pyplot(fig, clear_figure=True)

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    if len(num_df.columns) >= 2:
        st.subheader("🔗 Correlation Heatmap")
        corr = num_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(max(5, len(corr.columns)*0.6), max(5, len(corr.columns)*0.6)))
        cax = ax.imshow(corr.values, cmap="coolwarm", interpolation="nearest")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(cax)
        st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV or Excel file to start.")
