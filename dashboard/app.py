import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# ------------------------------------------------------------
# AQI dashboard: clean, light interface with compact navigation
# ------------------------------------------------------------
st.set_page_config(
    page_title="AQI | Air Quality Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cleaned_aqi.csv"
MODEL_PATH = ROOT / "models" / "aqi_model.pkl"
RESULTS_CANDIDATES = [
    ROOT / "models" / "model_evaluation_results.csv",
    ROOT / "data" / "processed" / "model_evaluation_results.csv",
]

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f5f8fc; color: #172b4d; }
      [data-testid="stHeader"] { background: rgba(245,248,252,.92); }
      [data-testid="stSidebar"] { background: #ffffff; }
      .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; }
      h1, h2, h3 { color: #17365d; letter-spacing: -0.02em; }
      h1 { font-size: 2.05rem !important; }
      h2 { font-size: 1.35rem !important; }
      h3 { font-size: 1.05rem !important; }
      [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e2eaf3;
        padding: 16px 18px; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(24,55,90,.04);
      }
      [data-testid="stMetricLabel"] { color: #64748b; }
      [data-testid="stMetricValue"] { color: #17365d; }
      div[data-testid="stTabs"] button { font-weight: 600; }
      div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #087e8b; border-bottom-color: #087e8b;
      }
      .intro { color: #64748b; margin-top: -.55rem; margin-bottom: 1.3rem; }
      .section-note { color: #64748b; font-size: .9rem; }
      .status-strip {
        background: #eaf4ff; border: 1px solid #cfe4fa; color: #24496b;
        padding: 12px 15px; border-radius: 10px; margin: 8px 0 18px 0;
      }
      .footer { color: #8190a5; font-size: .82rem; border-top: 1px solid #e3eaf2; padding-top: 14px; margin-top: 28px; }
    </style>
    """,
    unsafe_allow_html=True,
)

def locate_first(paths):
    for p in paths:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner="Loading AQI data...")
def load_data(path_string):
    raw = pd.read_csv(path_string)
    required = {"Date", "City", "AQI"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    raw["AQI"] = pd.to_numeric(raw["AQI"], errors="coerce")
    raw["City"] = raw["City"].astype("string").str.strip()
    raw = raw.dropna(subset=["Date", "AQI", "City"])
    raw = raw[raw["City"].ne("")]

    daily = (
        raw.groupby(["City", "Date"], as_index=False)
        .agg(AQI=("AQI", "mean"))
        .sort_values(["City", "Date"])
        .reset_index(drop=True)
    )
    return raw, daily

def category(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"

def style_fig(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=52, b=12),
        font=dict(family="Arial", color="#334155"),
        title=dict(font=dict(size=16, color="#17365d")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#edf2f7", zeroline=False)
    return fig

st.title("Air Quality Analytics")
st.markdown(
    '<div class="intro">AQI trends, city comparisons and machine-learning estimates for Indian cities · Historical dataset: 2020–2025</div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error("The cleaned AQI dataset was not found.")
    st.write("Expected file: `data/processed/cleaned_aqi.csv`")
    st.stop()

try:
    raw_df, df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

years = sorted(df["Date"].dt.year.unique().tolist())
cities = sorted(df["City"].dropna().unique().tolist())
min_date, max_date = df["Date"].min().date(), df["Date"].max().date()

# Horizontal navigation makes the main sections visible immediately.
tabs = st.tabs(["Overview", "City insights", "Pollutants", "AQI prediction"])

# ------------------------------- OVERVIEW -------------------------------
with tabs[0]:
    st.subheader("At a glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cities covered", f"{len(cities):,}")
    k2.metric("Average AQI", f"{df['AQI'].mean():.1f}")
    k3.metric("Highest recorded AQI", f"{df['AQI'].max():.0f}")
    k4.metric("Lowest recorded AQI", f"{df['AQI'].min():.0f}")

    left, right = st.columns([1.35, 1])
    with left:
        yearly = df.assign(Year=df["Date"].dt.year).groupby("Year", as_index=False)["AQI"].mean()
        st.plotly_chart(
            style_fig(px.line(yearly, x="Year", y="AQI", markers=True, title="Average AQI by year"), 360),
            use_container_width=True,
        )
    with right:
        cat = df.assign(Category=df["AQI"].map(category)).groupby("Category").size().reindex(
            ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"], fill_value=0
        ).rename("Records").reset_index()
        st.plotly_chart(
            style_fig(px.bar(cat, x="Category", y="Records", title="AQI category distribution"), 360),
            use_container_width=True,
        )

    st.subheader("City snapshot")
    chosen_city = st.selectbox("Choose a city to preview", cities, key="overview_city")
    city_data = df[df["City"] == chosen_city]
    a, b, c = st.columns(3)
    a.metric(f"{chosen_city} · average AQI", f"{city_data['AQI'].mean():.1f}")
    b.metric("Maximum AQI", f"{city_data['AQI'].max():.0f}")
    c.metric("Latest available reading", f"{city_data.sort_values('Date').iloc[-1]['AQI']:.0f}")
    st.caption(f"Data covers {min_date.strftime('%d %b %Y')} to {max_date.strftime('%d %b %Y')}. Values are based on the supplied dataset, not live monitoring.")

# ------------------------------- CITY INSIGHTS -------------------------------
with tabs[1]:
    st.subheader("Explore a city's air quality")
    selected_city = st.selectbox("Select city", cities, key="city_page_select")
    city_df = df[df["City"] == selected_city].sort_values("Date").copy()
    x1, x2, x3 = st.columns(3)
    x1.metric("Average AQI", f"{city_df['AQI'].mean():.1f}")
    x2.metric("Maximum AQI", f"{city_df['AQI'].max():.0f}")
    x3.metric("Minimum AQI", f"{city_df['AQI'].min():.0f}")

    st.plotly_chart(
        style_fig(px.line(city_df, x="Date", y="AQI", title=f"Daily AQI trend · {selected_city}"), 380),
        use_container_width=True,
    )
    lcol, rcol = st.columns(2)
    with lcol:
        month_data = city_df.assign(Month=city_df["Date"].dt.month).groupby("Month", as_index=False)["AQI"].mean()
        month_data["Month"] = pd.to_datetime(month_data["Month"], format="%m").dt.strftime("%b")
        st.plotly_chart(style_fig(px.bar(month_data, x="Month", y="AQI", title="Average AQI by month"), 320), use_container_width=True)
    with rcol:
        year_data = city_df.assign(Year=city_df["Date"].dt.year).groupby("Year", as_index=False)["AQI"].mean()
        st.plotly_chart(style_fig(px.line(year_data, x="Year", y="AQI", markers=True, title="Average AQI by year"), 320), use_container_width=True)

    st.subheader("Compare cities")
    compare_cities = st.multiselect("Choose up to 6 cities", cities, default=[selected_city], max_selections=6)
    if compare_cities:
        comp = df[df["City"].isin(compare_cities)].assign(Year=lambda x: x["Date"].dt.year)
        comp = comp.groupby(["Year", "City"], as_index=False)["AQI"].mean()
        st.plotly_chart(style_fig(px.line(comp, x="Year", y="AQI", color="City", markers=True, title="Yearly AQI comparison"), 360), use_container_width=True)

# ------------------------------- POLLUTANTS -------------------------------
with tabs[2]:
    st.subheader("Pollutant and city comparisons")
    if "Pollutant" in raw_df.columns:
        pollutants = raw_df["Pollutant"].dropna().astype(str).str.strip()
        pollutants = pollutants[pollutants.ne("")]
        counts = pollutants.value_counts().head(15).rename_axis("Pollutant").reset_index(name="Records")
        st.plotly_chart(style_fig(px.bar(counts, x="Pollutant", y="Records", title="Most frequently reported prominent pollutants"), 340), use_container_width=True)
    else:
        st.info("The source dataset does not contain a Pollutant column.")

    col1, col2 = st.columns(2)
    city_means = df.groupby("City", as_index=False)["AQI"].mean()
    with col1:
        most = city_means.nlargest(10, "AQI").sort_values("AQI")
        st.plotly_chart(style_fig(px.bar(most, x="AQI", y="City", orientation="h", title="10 highest-average-AQI cities"), 390), use_container_width=True)
    with col2:
        cleanest = city_means.nsmallest(10, "AQI").sort_values("AQI", ascending=False)
        st.plotly_chart(style_fig(px.bar(cleanest, x="AQI", y="City", orientation="h", title="10 lowest-average-AQI cities"), 390), use_container_width=True)

    st.subheader("City-by-year AQI heatmap")
    heat_source = df.assign(Year=df["Date"].dt.year)
    heat = heat_source.pivot_table(index="City", columns="Year", values="AQI", aggfunc="mean")
    heat = heat.loc[heat.mean(axis=1).sort_values(ascending=False).head(30).index]
    st.plotly_chart(style_fig(px.imshow(heat, aspect="auto", labels=dict(color="Average AQI"), title="30 highest-average-AQI cities by year"), 600), use_container_width=True)

# ------------------------------- PREDICTION -------------------------------
with tabs[3]:
    st.subheader("Estimate AQI for a selected city")
    st.markdown(
        '<div class="status-strip">This feature estimates one next-step AQI value from the latest seven available readings. It is not a live AQI reading or a long-range forecast.</div>',
        unsafe_allow_html=True,
    )
    pred_city = st.selectbox("Select city", cities, key="prediction_city")
    history_df = df[df["City"] == pred_city].sort_values("Date")
    latest_date = history_df["Date"].max()
    st.caption(f"Latest available record for {pred_city}: {latest_date.date()}")
    default_date = (latest_date + pd.Timedelta(days=1)).date()
    pred_date = st.date_input("Prediction date", value=default_date, min_value=default_date, key="prediction_date")

    if st.button("Generate AQI estimate", type="primary", key="run_prediction"):
        if not MODEL_PATH.exists():
            st.error("The trained model file is missing. Add your compatible model at `models/aqi_model.pkl`.")
            st.info("Run the model-training notebook and save the fitted pipeline to the models folder.")
        else:
            try:
                with st.spinner("Loading model and generating estimate..."):
                    # Load only when needed. A broken/incompatible pickle won't stop analytics pages.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", InconsistentVersionWarning)
                        fitted_model = joblib.load(MODEL_PATH)

                    vals = history_df.loc[history_df["Date"] < pd.Timestamp(pred_date), "AQI"].to_numpy()
                    if len(vals) < 7:
                        st.error("At least seven previous AQI records are required.")
                    else:
                        features = pd.DataFrame([{
                            "City": pred_city,
                            "Year": pred_date.year,
                            "Month": pred_date.month,
                            "Day": pred_date.day,
                            "DayOfYear": pd.Timestamp(pred_date).dayofyear,
                            "DayOfWeek": pd.Timestamp(pred_date).dayofweek,
                            "Quarter": pd.Timestamp(pred_date).quarter,
                            "Previous_AQI": vals[-1],
                            "Previous_2_AQI": vals[-2],
                            "Previous_3_AQI": vals[-3],
                            "Previous_7_AQI": vals[-7],
                            "Average_3": float(np.mean(vals[-3:])),
                            "Average_7": float(np.mean(vals[-7:])),
                            "Recent_Change": float(vals[-1] - vals[-2]),
                        }])
                        estimate = max(0.0, float(fitted_model.predict(features)[0]))
                        r1, r2 = st.columns(2)
                        r1.metric("Estimated AQI", f"{estimate:.1f}")
                        r2.metric("Estimated category", category(estimate))
                        st.caption("This is a model estimate based on historical data. Actual air quality may differ.")
            except Exception as exc:
                st.error("The saved model could not be loaded or used with the current Python libraries.")
                st.warning(
                    "This usually happens when the model was saved with a different scikit-learn version. "
                    "A pickle file cannot always be repaired by changing the dashboard code."
                )
                with st.expander("Technical details"):
                    st.code(str(exc))
                st.markdown(
                    "**How to fix it:** open `notebook/05_Model_Training.ipynb`, install/use the same scikit-learn environment for training and dashboard, rerun the training cells, and save a fresh `aqi_model.pkl` into the repository's `models` folder."
                )

