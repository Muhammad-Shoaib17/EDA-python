import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Page Setup
# ---------------------------------
st.set_page_config(
    page_title="Economic Data Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Economic Data Explorer")
st.caption("Upload your dataset to explore its structure, quality, and insights interactively.")

# ---------------------------------
# Helpers
# ---------------------------------
@st.cache_data(show_spinner=False)
def load_file(file, sheet_name=None, encoding="utf-8", sep=","):
    name = getattr(file, "name", "")
    suffix = os.path.splitext(name)[1].lower()

    if suffix in [".csv", ".txt"]:
        try:
            return pd.read_csv(file, encoding=encoding, sep=sep)
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, encoding="latin-1", sep=sep)
        except Exception as e:
            raise RuntimeError(f"CSV read error: {e}")

    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"Excel read error: {e}")

    raise RuntimeError("Unsupported file type. Please upload CSV/XLSX/XLS.")

def is_datetime_series(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    try:
        pd.to_datetime(s.dropna().head(10), errors="raise")
        return True
    except Exception:
        return False

# ---------------------------------
# Sidebar – File Upload
# ---------------------------------
st.sidebar.header("📂 Upload Dataset")
Ufile = st.sidebar.file_uploader(
    "Upload your file",
    type=["csv", "txt", "xlsx", "xls"],
    help="Supports CSV, TXT, and Excel files"
)

csv_sep = st.sidebar.text_input("CSV Delimiter", value=",")
csv_encoding = st.sidebar.text_input("CSV Encoding", value="utf-8")

sheet_name = None
if Ufile and os.path.splitext(Ufile.name)[1].lower() in [".xlsx", ".xls"]:
    try:
        xls = pd.ExcelFile(Ufile)
        Ufile.seek(0)
        sheet_name = st.sidebar.selectbox("Select Sheet", options=xls.sheet_names, index=0)
    except Exception:
        sheet_name = None

# ---------------------------------
# Main Content
# ---------------------------------
if Ufile:
    try:
        df = load_file(Ufile, sheet_name=sheet_name, encoding=csv_encoding, sep=csv_sep)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Dataset Preview
    st.subheader("👀 Quick Preview")
    st.write(df.head(10))

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    st.subheader("📋 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Rows", f"{len(df):,}")
    with c2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    with st.expander("🔍 Missing Values by Column"):
        na_table = df.isna().sum().to_frame("Missing Count")
        na_table["Missing %"] = (na_table["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(na_table)

    # -------------------------------
    # Summary Statistics
    # -------------------------------
    st.subheader("📊 Summary Statistics")
    if df.select_dtypes(include=np.number).shape[1] > 0:
        st.markdown("**Numeric Columns**")
        st.dataframe(df.describe().T)
    if df.select_dtypes(exclude=np.number).shape[1] > 0:
        st.markdown("**Categorical Columns**")
        nunique = df.select_dtypes(exclude=np.number).nunique().sort_values(ascending=False)
        st.dataframe(nunique.to_frame("Unique Values"))

    # -------------------------------
    # Column Analysis
    # -------------------------------
    st.subheader("🔎 Column Explorer")
    column = st.selectbox("Choose a column", df.columns, index=0)
    col = df[column]

    if is_datetime_series(col):
        if not pd.api.types.is_datetime64_any_dtype(col):
            df[column] = pd.to_datetime(col, errors="coerce")
        st.markdown(f"**{column} (Datetime)**")

        agg_target = st.selectbox(
            "Aggregate on:",
            options=["<count>"] + list(df.select_dtypes(include=np.number).columns),
            index=0
        )
        freq = st.selectbox("Resample Frequency", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], index=2)
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}

        tmp = df[[column]].copy()
        if agg_target != "<count>":
            tmp[agg_target] = df[agg_target]
        tmp = tmp.dropna(subset=[column]).sort_values(column).set_index(column)

        series = tmp.resample(freq_map[freq]).size() if agg_target == "<count>" else tmp[agg_target].resample(freq_map[freq]).mean()
        st.line_chart(series)

    elif pd.api.types.is_numeric_dtype(col):
        st.markdown(f"**{column} (Numeric)**")
        st.write(col.describe())

        bins = st.slider("Number of bins", 5, 100, 30)
        fig, ax = plt.subplots()
        col.dropna().hist(ax=ax, bins=bins)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        with st.expander("📦 Boxplot"):
            fig2, ax2 = plt.subplots()
            ax2.boxplot(col.dropna().values, vert=True)
            ax2.set_title(f"Boxplot of {column}")
            st.pyplot(fig2)

    else:
        st.markdown(f"**{column} (Categorical/Text)**")
        vc = col.astype("string").fillna("<NA>").value_counts().head(30)
        st.dataframe(vc.to_frame("Count"))

        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"Top Categories in {column}")
        st.pyplot(fig)

    # -------------------------------
    # Correlation Analysis
    # -------------------------------
    num = df.select_dtypes(include=np.number)
    if num.shape[1] >= 2:
        st.subheader("🔗 Correlation Analysis")
        corr = num.corr(numeric_only=True)
        st.dataframe(corr.style.format("{:.2f}"))

        with st.expander("Heatmap View"):
            fig, ax = plt.subplots(figsize=(min(12, 1 + 0.7 * corr.shape[1]), min(8, 1 + 0.5 * corr.shape[0])))
            cax = ax.imshow(corr.values, interpolation="nearest", cmap="coolwarm")
            ax.set_xticks(range(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(corr.shape[0])); ax.set_yticklabels(corr.index)
            ax.set_title("Correlation Heatmap")
            fig.colorbar(cax)
            st.pyplot(fig)

    # -------------------------------
    # Filter & Export
    # -------------------------------
    st.subheader("🧹 Filter & Export Data")
    with st.expander("Apply Filter (pandas query syntax)"):
        st.caption("Example: `price > 100 and region == 'East'`")
        q = st.text_input("Enter filter query")
        if q:
            try:
                filtered = df.query(q)
                st.write(filtered.head(20))
                st.success(f"Rows after filter: {len(filtered):,}")
                st.download_button("⬇️ Download Filtered Data", filtered.to_csv(index=False).encode("utf-8"), "filtered.csv", "text/csv")
            except Exception as e:
                st.error(f"Filter error: {e}")

    st.download_button("⬇️ Download Full Dataset", df.to_csv(index=False).encode("utf-8"), "full_dataset.csv", "text/csv")

else:
    st.info("📂 Please upload a CSV or Excel file to start exploring.")
