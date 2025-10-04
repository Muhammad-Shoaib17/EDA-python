import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Page Setup
# ---------------------------------
st.set_page_config(
    page_title="📊 Simple EDA Tool",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Simple EDA Tool")
st.caption("Upload a CSV or Excel file to instantly explore your data.")

# ---------------------------------
# Helper
# ---------------------------------
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

# ---------------------------------
# File Upload
# ---------------------------------
Ufile = st.file_uploader(
    "Upload your dataset",
    type=["csv", "txt", "xlsx", "xls"]
)

# ---------------------------------
# Main EDA
# ---------------------------------
if Ufile:
    try:
        df = load_file(Ufile)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("📋 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Rows", f"{len(df):,}")
    with c2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    st.subheader("🧮 Summary Statistics")
    if df.select_dtypes(include=np.number).shape[1] > 0:
        st.markdown("**Numeric Columns**")
        st.dataframe(df.describe().T)

    if df.select_dtypes(exclude=np.number).shape[1] > 0:
        st.markdown("**Categorical Columns (Unique Counts)**")
        cat_counts = df.select_dtypes(exclude=np.number).nunique().sort_values(ascending=False)
        st.dataframe(cat_counts.to_frame("Unique Values"))

    st.subheader("❌ Missing Values")
    na_table = df.isna().sum().to_frame("Missing Count")
    na_table["Missing %"] = (na_table["Missing Count"] / len(df) * 100).round(2)
    st.dataframe(na_table)

    st.subheader("📊 Basic Graphs")
    # Histograms for numeric columns
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        st.markdown(f"**Histogram — {col}**")
        fig, ax = plt.subplots()
        df[col].dropna().hist(ax=ax, bins=30)
        ax.set_title(col)
        st.pyplot(fig)

    # Bar charts for categorical columns (top 10 categories)
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        st.markdown(f"**Top Categories — {col}**")
        vc = df[col].astype(str).value_counts().head(10)
        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

    # Correlation heatmap (numeric only)
    if len(num_cols) >= 2:
        st.subheader("🔗 Correlation Heatmap")
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(12, 1 + 0.7 * corr.shape[1]), 8))
        cax = ax.imshow(corr.values, cmap="coolwarm", interpolation="nearest")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(cax)
        st.pyplot(fig)

else:
    st.info("📂 Please upload a CSV or Excel file to start.")
