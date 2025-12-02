# workspace/streamlit_template/pages/06_Demand_Forecasting.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Allow importing utils from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.models import ForecastingModels
from utils.ui_components import render_sidebar
from utils.load_css import load_css_for_page   # ← NEW FIX


# -----------------------------------------------------------
# IMPORTANT: MUST BE FIRST STREAMLIT CALL
# -----------------------------------------------------------
st.set_page_config(page_title="Demand Forecasting", page_icon="🔮", layout="wide")

# Load CSS safely (no FileNotFound error on Streamlit Cloud)
load_css_for_page(__file__)
# -----------------------------------------------------------


# Authentication check
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

st.title("🔮 Advanced Demand Forecasting")
st.markdown("Multi-level demand forecasting with promotional impact analysis")


# -----------------------------------------------------------
# DATA CHECK
# -----------------------------------------------------------
if 'uploaded_data' not in st.session_state and 'processed_data' not in st.session_state:
    st.warning("⚠️ No data found. Please upload data first!")
    st.markdown("[👆 Go to Upload Data page](1_📊_Upload_Data)")
    st.stop()

df = st.session_state.get('processed_data', st.session_state.get('uploaded_data'))


# -----------------------------------------------------------
# SIDEBAR CONFIG
# -----------------------------------------------------------
with st.sidebar:
    st.markdown("### Forecasting Configuration")

    forecast_type = st.selectbox(
        "Forecast Type",
        ["Total Demand", "By SKU", "By Store", "By Category", "By Region"]
    )

    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year"]
    )

    include_promotions = st.checkbox("Include Promotional Impact", value=True)
    include_seasonality = st.checkbox("Include Seasonality", value=True)
    include_trends = st.checkbox("Include Trend Analysis", value=True)


# -----------------------------------------------------------
# MAIN TABS
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Overview & Insights",
    "🎯 Forecast Generation",
    "📈 Advanced Analysis"
])

# ===========================================================
# TAB 1 — OVERVIEW
# ===========================================================
with tab1:
    st.markdown("### 📊 Demand Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_demand = df['Sales'].sum() if 'Sales' in df.columns else 0
        st.metric("Total Sales Value", f"₹{total_demand:,.0f}")

    with col2:
        avg_demand = df['Sales'].mean() if 'Sales' in df.columns else 0
        st.metric("Avg Daily Sales", f"₹{avg_demand:,.0f}")

    with col3:
        if 'Units_Sold' in df.columns:
            total_units = df['Units_Sold'].sum()
            st.metric("Total Units Sold", f"{total_units:,.0f}")
        else:
            peak_demand = df['Sales'].max() if 'Sales' in df.columns else 0
            st.metric("Peak Sales", f"₹{peak_demand:,.0f}")

    with col4:
        if 'Units_Sold' in df.columns:
            units_volatility = df['Units_Sold'].std() / df['Units_Sold'].mean() * 100
            st.metric("Units Volatility", f"{units_volatility:.1f}%")
        else:
            demand_volatility = (
                df['Sales'].std() / df['Sales'].mean() * 100
                if 'Sales' in df.columns else 0
            )
            st.metric("Sales Volatility", f"{demand_volatility:.1f}%")


    # ---------------------------
    # Trend Line Chart
    # ---------------------------
    st.markdown("#### Demand Patterns")

    if 'Date' in df.columns and 'Sales' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        daily = df.groupby('Date')['Sales'].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['Date'], y=daily['Sales'],
            mode='lines', name='Actual', line=dict(width=2)
        ))

        daily['MA_7'] = daily['Sales'].rolling(7).mean()
        daily['MA_30'] = daily['Sales'].rolling(30).mean()

        fig.add_trace(go.Scatter(
            x=daily['Date'], y=daily['MA_7'],
            name='7-Day MA', line=dict(dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=daily['Date'], y=daily['MA_30'],
            name='30-Day MA', line=dict(dash='dot')
        ))

        fig.update_layout(
            title="Demand Trend Analysis",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================
# TAB 2 — FORECAST GENERATION
# ===========================================================
with tab2:
    st.markdown("### 🎯 Generate Demand Forecast")

    col1, col2, col3 = st.columns(3)

    forecast_days_map = {
        "1 Week": 7, "2 Weeks": 14, "1 Month": 30,
        "3 Months": 90, "6 Months": 180, "1 Year": 365
    }
    days = forecast_days_map[forecast_horizon]

    with col1:
        st.info(f"Forecast for {days} days")

    with col2:
        confidence_level = st.selectbox("Confidence Level", ["80%", "90%", "95%", "99%"], index=2)

    with col3:
        aggregation = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly"])

    st.markdown("#### Select Forecasting Models")

    col1, col2, col3 = st.columns(3)

    use_ensemble = col1.checkbox("Ensemble", value=True)
    use_prophet  = col2.checkbox("Prophet", value=True)
    use_xgb      = col3.checkbox("XGBoost", value=True)


    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):

        if not any([use_ensemble, use_prophet, use_xgb]):
            st.error("Select at least one forecasting model")
            st.stop()

        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            base_df = df.groupby('Date')['Sales'].sum().reset_index()
            base_df.columns = ['ds', 'y']

            forecaster = ForecastingModels()
            results = {}

            # Prophet
            if use_prophet:
                results['Prophet'] = forecaster.train_prophet(
                    base_df, days, include_holidays=include_promotions
                )

            # XGBoost
            if use_xgb:
                results['XGBoost'] = forecaster.train_xgboost(base_df, days)

            # Ensemble
            if use_ensemble and len(results) > 0:
                ensemble_avg = None

                for model in results.values():
                    if ensemble_avg is None:
                        ensemble_avg = model["forecast"]["yhat"].copy()
                    else:
                        ensemble_avg += model["forecast"]["yhat"]

                ensemble_avg = ensemble_avg / len(results)

                ensemble_output = {
                    "forecast": pd.DataFrame({
                        "ds": list(results.values())[0]["forecast"]["ds"],
                        "yhat": ensemble_avg
                    }),
                    "metrics": {}
                }
                results["Ensemble"] = ensemble_output

            st.session_state.demand_forecast_results = results
            st.success("✅ Forecast generated successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

    # Display Results
    if 'demand_forecast_results' in st.session_state:
        results = st.session_state.demand_forecast_results

        st.markdown("#### 📊 Forecast Result Comparison")

        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange']

        for i, (name, res) in enumerate(results.items()):
            fig.add_trace(go.Scatter(
                x=res['forecast']['ds'],
                y=res['forecast']['yhat'],
                mode='lines',
                name=name,
                line=dict(width=2, color=colors[i % len(colors)])
            ))

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================
# TAB 3 — ADVANCED ANALYSIS
# ===========================================================
with tab3:
    st.markdown("### 📈 Advanced Analysis")
    st.caption("Promotional impact, units forecasting, and more…")


    # (Your remaining advanced analysis code can stay unchanged)
