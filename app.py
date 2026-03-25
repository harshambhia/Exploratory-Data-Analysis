import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.3rem 0;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #7eb8f7;
    }
    .metric-card .label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        color: #7eb8f7;
        border-left: 3px solid #7eb8f7;
        padding-left: 12px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
    }
    .insight-box {
        background: #1a2744;
        border-left: 4px solid #f0a500;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #ddd;
    }
    .stAlert { border-radius: 8px; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data
def load_data(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        return None


def classify_columns(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64"]).columns.tolist()
    return numeric, categorical, datetime


def missing_summary(df):
    miss = df.isnull().sum()
    pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"Missing Values": miss, "Percentage (%)": pct})[miss > 0].sort_values("Percentage (%)", ascending=False)


def auto_insights(df, numeric_cols, categorical_cols):
    insights = []
    # High cardinality
    for c in categorical_cols:
        if df[c].nunique() > 50:
            insights.append(f"⚠️ **{c}** has {df[c].nunique()} unique values — consider grouping or encoding.")
    # Skewness
    for c in numeric_cols:
        sk = df[c].skew()
        if abs(sk) > 1:
            direction = "right" if sk > 0 else "left"
            insights.append(f"📐 **{c}** is {direction}-skewed (skew={sk:.2f}) — may need log/power transform.")
    # Constant columns
    for c in df.columns:
        if df[c].nunique() <= 1:
            insights.append(f"🚫 **{c}** has only 1 unique value — likely useless for modelling.")
    # High missing
    miss_pct = (df.isnull().mean() * 100)
    for c in miss_pct[miss_pct > 30].index:
        insights.append(f"🕳️ **{c}** is missing {miss_pct[c]:.1f}% of values.")
    # Potential ID columns
    for c in df.columns:
        if df[c].nunique() == len(df):
            insights.append(f"🔑 **{c}** appears to be a unique identifier (all values distinct).")
    return insights


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 EDA Agent")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "txt", "xlsx", "xls"],
        help="Supports CSV, TXT (delimited), Excel"
    )
    st.markdown("---")
    if uploaded_file:
        st.markdown("### ⚙️ Settings")
        show_raw = st.checkbox("Show raw data", value=True)
        sample_n = st.slider("Sample rows to display", 5, 100, 20)
        corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
        st.markdown("---")
    st.markdown("<small style='color:#555'>Built with Streamlit + Plotly</small>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 Exploratory Data Analysis Agent")
st.markdown("Upload a dataset in the sidebar to begin automated analysis.")

if not uploaded_file:
    st.info("👈 Upload a CSV, TXT, or Excel file in the sidebar to start.")
    st.stop()

df = load_data(uploaded_file)
if df is None:
    st.stop()

numeric_cols, categorical_cols, datetime_cols = classify_columns(df)

# ── Tab layout ────────────────────────────────────────────────────────────────
tabs = st.tabs(["📋 Overview", "📊 Distributions", "🔗 Correlations", "🗂️ Categoricals", "🕳️ Missing Data", "💡 Insights"])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [df.shape[0], df.shape[1], len(numeric_cols), len(categorical_cols), int(df.isnull().sum().sum())],
        ["Rows", "Columns", "Numeric", "Categorical", "Missing Cells"]
    ):
        col.markdown(f'<div class="metric-card"><div class="value">{val:,}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe(include="all").T.style.format(precision=2), use_container_width=True)

    if show_raw:
        st.markdown('<div class="section-header">Raw Data Sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(sample_n), use_container_width=True)

    st.markdown('<div class="section-header">Column Types</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ════════════════════════════════════════════════════════════
with tabs[1]:
    if not numeric_cols:
        st.warning("No numeric columns detected.")
    else:
        st.markdown('<div class="section-header">Numeric Distributions</div>', unsafe_allow_html=True)
        selected_num = st.multiselect("Choose columns", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])

        if selected_num:
            for col in selected_num:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram + KDE", "Box Plot"))

                # Histogram
                fig.add_trace(go.Histogram(x=df[col].dropna(), name=col, marker_color="#7eb8f7",
                                           opacity=0.75, nbinsx=40), row=1, col=1)
                # Box
                fig.add_trace(go.Box(y=df[col].dropna(), name=col, marker_color="#f0a500",
                                     boxmean="sd"), row=1, col=2)

                fig.update_layout(
                    title=f"Distribution of {col}",
                    template="plotly_dark",
                    height=320,
                    showlegend=False,
                    margin=dict(t=50, b=30)
                )
                st.plotly_chart(fig, use_container_width=True)

                sk = df[col].skew()
                ku = df[col].kurtosis()
                st.caption(f"Skewness: **{sk:.3f}** | Kurtosis: **{ku:.3f}** | Mean: **{df[col].mean():.3f}** | Std: **{df[col].std():.3f}**")
                st.markdown("---")


# ════════════════════════════════════════════════════════════
# TAB 3 — CORRELATIONS
# ════════════════════════════════════════════════════════════
with tabs[2]:
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
    else:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = df[numeric_cols].corr(method=corr_method)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template="plotly_dark",
            title=f"{corr_method.capitalize()} Correlation Matrix"
        )
        fig.update_layout(height=max(400, len(numeric_cols) * 35 + 100))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Top Correlated Pairs</div>', unsafe_allow_html=True)
        pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                     .stack()
                     .reset_index())
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["Abs Corr"] = pairs["Correlation"].abs()
        pairs = pairs.sort_values("Abs Corr", ascending=False).drop("Abs Corr", axis=1)
        st.dataframe(pairs.head(20).style.format({"Correlation": "{:.4f}"}), use_container_width=True)

        st.markdown('<div class="section-header">Scatter Plot Explorer</div>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        x_col = ca.selectbox("X axis", numeric_cols, index=0)
        y_col = cb.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        color_by = st.selectbox("Color by (optional)", ["None"] + categorical_cols)

        fig2 = px.scatter(
            df, x=x_col, y=y_col,
            color=None if color_by == "None" else color_by,
            template="plotly_dark",
            trendline="ols",
            opacity=0.65,
            title=f"{x_col} vs {y_col}"
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — CATEGORICALS
# ════════════════════════════════════════════════════════════
with tabs[3]:
    if not categorical_cols:
        st.warning("No categorical columns detected.")
    else:
        st.markdown('<div class="section-header">Categorical Analysis</div>', unsafe_allow_html=True)
        selected_cat = st.multiselect("Choose columns", categorical_cols, default=categorical_cols[:min(3, len(categorical_cols))])
        top_n = st.slider("Show top N categories", 5, 30, 10)

        for col in selected_cat:
            vc = df[col].value_counts().head(top_n).reset_index()
            vc.columns = [col, "Count"]

            fig = px.bar(vc, x=col, y="Count",
                         template="plotly_dark",
                         title=f"Top {top_n} values — {col}",
                         color="Count",
                         color_continuous_scale="Blues")
            fig.update_layout(height=320, margin=dict(t=50, b=40), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            pct_covered = vc["Count"].sum() / len(df) * 100
            st.caption(f"Top {top_n} categories cover **{pct_covered:.1f}%** of rows | Total unique values: **{df[col].nunique()}**")
            st.markdown("---")


# ════════════════════════════════════════════════════════════
# TAB 5 — MISSING DATA
# ════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Missing Value Summary</div>', unsafe_allow_html=True)
    miss_df = missing_summary(df)

    if miss_df.empty:
        st.success("✅ No missing values found in this dataset!")
    else:
        st.dataframe(miss_df.style.format({"Percentage (%)": "{:.2f}"}), use_container_width=True)

        fig = px.bar(
            miss_df.reset_index(), x="index", y="Percentage (%)",
            template="plotly_dark",
            title="Missing Value % by Column",
            color="Percentage (%)",
            color_continuous_scale="OrRd"
        )
        fig.update_layout(xaxis_title="Column", height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Missing Value Heatmap</div>', unsafe_allow_html=True)
        miss_matrix = df[miss_df.index.tolist()].isnull().astype(int)
        fig2 = px.imshow(
            miss_matrix.T,
            template="plotly_dark",
            color_continuous_scale=[[0, "#1e2a45"], [1, "#e05c5c"]],
            title="Row-level missingness (red = missing)",
            zmin=0, zmax=1
        )
        fig2.update_layout(height=max(200, len(miss_df) * 28 + 80))
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 6 — INSIGHTS
# ════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Automated Insights</div>', unsafe_allow_html=True)
    insights = auto_insights(df, numeric_cols, categorical_cols)

    if insights:
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    else:
        st.success("✅ No major issues detected. Dataset looks clean!")

    st.markdown('<div class="section-header">Quick Recommendations</div>', unsafe_allow_html=True)
    recs = []
    if df.duplicated().sum() > 0:
        recs.append(f"🔁 Found **{df.duplicated().sum()}** duplicate rows — consider removing them.")
    if df.isnull().any().any():
        recs.append("🕳️ Missing values present — consider imputation or removal strategies.")
    if len(numeric_cols) > 1:
        high_corr = (df[numeric_cols].corr().abs().where(
            np.triu(np.ones((len(numeric_cols), len(numeric_cols))), k=1).astype(bool)) > 0.9).stack()
        if high_corr.any():
            recs.append("🔗 Highly correlated features (>0.9) detected — consider feature selection.")
    if not recs:
        recs.append("✅ Dataset appears ready for modelling after standard preprocessing.")
    for r in recs:
        st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)
