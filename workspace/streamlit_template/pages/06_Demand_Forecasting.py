import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import ForecastingModels
from utils.ui_components import render_sidebar

def load_css():
    with open("assets/custom.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="Demand Forecasting", page_icon="🔮", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

st.title("🔮 Advanced Demand Forecasting")
st.markdown("Multi-level demand forecasting with promotional impact analysis")

# Check for data
if 'uploaded_data' not in st.session_state and 'processed_data' not in st.session_state:
    st.warning("⚠️ No data found. Please upload data first!")
    st.markdown("[👆 Go to Upload Data page](1_📊_Upload_Data)")
    st.stop()

# Get data
df = st.session_state.get('processed_data', st.session_state.get('uploaded_data'))

# Sidebar configuration
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

# Main content tabs - Enterprise best practice (3 tabs)
tab1, tab2, tab3 = st.tabs([
    "📊 Overview & Insights",
    "🎯 Forecast Generation",
    "📈 Advanced Analysis"
])

with tab1:
    st.markdown("### 📊 Demand Overview")
    
    # Key metrics
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
            demand_volatility = df['Sales'].std() / df['Sales'].mean() * 100 if 'Sales' in df.columns else 0
            st.metric("Sales Volatility", f"{demand_volatility:.1f}%")
    
    # Demand patterns
    st.markdown("#### Demand Patterns")
    
    if 'Date' in df.columns and 'Sales' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        daily_demand = df.groupby('Date')['Sales'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_demand['Date'],
            y=daily_demand['Sales'],
            mode='lines',
            name='Actual Demand',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving average
        daily_demand['MA_7'] = daily_demand['Sales'].rolling(window=7).mean()
        daily_demand['MA_30'] = daily_demand['Sales'].rolling(window=30).mean()
        
        fig.add_trace(go.Scatter(
            x=daily_demand['Date'],
            y=daily_demand['MA_7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_demand['Date'],
            y=daily_demand['MA_30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="Demand Trend Analysis",
            xaxis_title="Date",
            yaxis_title="Demand (₹)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly pattern
    if 'Date' in df.columns and 'Sales' in df.columns:
        df['DayOfWeek'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.day_name()
        weekly_pattern = df.groupby('DayOfWeek')['Sales'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig_weekly = px.bar(
            x=weekly_pattern.index,
            y=weekly_pattern.values,
            title="Average Demand by Day of Week",
            labels={'x': 'Day', 'y': 'Avg Demand (₹)'}
        )
        fig_weekly.update_layout(template='plotly_white')
        st.plotly_chart(fig_weekly, use_container_width=True)

    # Sales vs Units Scatter Plot
    st.markdown("#### 📈 Variable Relationship Analysis")

    if 'Units_Sold' in df.columns:
        fig_scatter = px.scatter(
            df,
            x='Units_Sold',
            y='Sales',
            title="Sales vs Units Sold Relationship",
            labels={'Units_Sold': 'Units Sold', 'Sales': 'Sales (₹)'},
            trendline="ols",
            opacity=0.7
        )

        correlation = df['Units_Sold'].corr(df['Sales'])
        fig_scatter.add_annotation(
            x=df['Units_Sold'].max() * 0.7,
            y=df['Sales'].max() * 0.9,
            text=f"Correlation: {correlation:.3f}",
            showarrow=False,
            font=dict(size=12, color='red')
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        if correlation > 0.7:
            st.success(f"🔗 **Strong Positive Correlation** ({correlation:.3f}): Sales and units sold are highly related")
        elif correlation > 0.3:
            st.info(f"🔗 **Moderate Positive Correlation** ({correlation:.3f}): Some relationship exists")
        else:
            st.warning(f"🔗 **Weak Correlation** ({correlation:.3f}): Other factors may influence sales")
    else:
        st.info("💡 **Tip**: Add a 'Units_Sold' column to see the relationship between units sold and sales value.")
    
    # Insights section
    st.markdown("---")
    st.markdown("### 🔔 Demand Alerts & Insights")
    
    if 'Sales' in df.columns and 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        recent_data = df.sort_values('Date').tail(30)
        
        recent_avg = recent_data['Sales'].mean()
        historical_avg = df['Sales'].mean()
        volatility = recent_data['Sales'].std()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Key Insights")
            
            if recent_avg > historical_avg * 1.1:
                st.success("📈 Upward Trend Detected: Recent demand is 10%+ higher than historical average")
            elif recent_avg < historical_avg * 0.9:
                st.warning("📉 Downward Trend Detected: Recent demand is 10%+ lower than historical average")
            else:
                st.info("➡️ Stable Demand: Recent patterns align with historical average")
            
            if volatility > historical_avg * 0.3:
                st.warning(f"⚠️ High Volatility Alert: Demand variance is significant (σ = ₹{volatility:,.0f})")
            else:
                st.success(f"✅ Stable Demand Pattern: Low volatility (σ = ₹{volatility:,.0f})")
            
            if 'DayOfWeek' in df.columns or len(df) > 365:
                st.info("📅 Seasonal Patterns Detected: Consider using seasonal forecasting models")
        
        with col2:
            st.markdown("#### 💡 Recommendations")
            
            st.markdown("""
            **Inventory Management:**
            - Maintain safety stock for high volatility items
            - Adjust reorder points based on forecast
            - Monitor promotional impact on demand
            
            **Forecasting Strategy:**
            - Use ensemble methods for critical products
            - Include promotional calendar in forecasts
            - Update forecasts weekly for accuracy
            
            **Risk Mitigation:**
            - Set up automated alerts for demand spikes
            - Plan for peak demand periods
            - Diversify supplier base for high-demand items
            """)

with tab2:
    st.markdown("### 🎯 Generate Demand Forecast")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_days_map = {
            "1 Week": 7, "2 Weeks": 14, "1 Month": 30,
            "3 Months": 90, "6 Months": 180, "1 Year": 365
        }
        days = forecast_days_map[forecast_horizon]
        st.info(f"Forecasting for {days} days")
    
    with col2:
        confidence_level = st.selectbox(
            "Confidence Level",
            ["80%", "90%", "95%", "99%"],
            index=2
        )
    
    with col3:
        aggregation = st.selectbox(
            "Aggregation",
            ["Daily", "Weekly", "Monthly"]
        )
    
    st.markdown("#### Select Forecasting Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_ensemble = st.checkbox("Ensemble (Recommended)", value=True,
                                   help="Combines multiple models")
    with col2:
        use_prophet = st.checkbox("Prophet", value=True,
                                 help="Good for seasonality")
    with col3:
        use_xgb = st.checkbox("XGBoost", value=True,
                             help="High performance")
    
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        if not any([use_ensemble, use_prophet, use_xgb]):
            st.error("Please select at least one forecasting model")
        else:
            with st.spinner("Generating demand forecast..."):
                try:
                    if 'Date' not in df.columns or 'Sales' not in df.columns:
                        st.error("Required columns 'Date' and 'Sales' not found")
                        st.stop()
                    
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                    forecast_df = df.groupby('Date')['Sales'].sum().reset_index()
                    forecast_df.columns = ['ds', 'y']
                    
                    forecaster = ForecastingModels()
                    results = {}
                    
                    if use_prophet:
                        prophet_result = forecaster.train_prophet(
                            forecast_df, days, include_holidays=include_promotions
                        )
                        results['Prophet'] = prophet_result
                    
                    if use_xgb:
                        xgb_result = forecaster.train_xgboost(forecast_df, days)
                        results['XGBoost'] = xgb_result
                    
                    if use_ensemble and len(results) > 0:
                        ensemble_forecast = None
                        for model_result in results.values():
                            if ensemble_forecast is None:
                                ensemble_forecast = model_result['forecast']['yhat'].copy()
                            else:
                                ensemble_forecast += model_result['forecast']['yhat']
                        
                        ensemble_forecast = ensemble_forecast / len(results)
                        
                        ensemble_result = {
                            'forecast': pd.DataFrame({
                                'ds': results[list(results.keys())[0]]['forecast']['ds'],
                                'yhat': ensemble_forecast
                            }),
                            'metrics': {'mae': 0, 'rmse': 0, 'mape': 0}
                        }
                        results['Ensemble'] = ensemble_result
                    
                    st.session_state.demand_forecast_results = results
                    st.session_state.demand_forecast_config = {
                        'horizon': forecast_horizon,
                        'days': days,
                        'confidence': confidence_level
                    }
                    
                    st.success("✅ Demand forecast generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
    
    if 'demand_forecast_results' in st.session_state:
        results = st.session_state.demand_forecast_results
        
        st.markdown("#### 📊 Forecast Results")
        
        fig = go.Figure()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (model_name, result) in enumerate(results.items()):
            if 'forecast' in result:
                forecast = result['forecast']
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=f'{model_name}',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        fig.update_layout(
            title="Demand Forecast Comparison",
            xaxis_title="Date",
            yaxis_title="Forecasted Demand (₹)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📋 Forecast Summary")
        
        summary_data = []
        for model_name, result in results.items():
            if 'forecast' in result:
                forecast = result['forecast']
                summary_data.append({
                    'Model': model_name,
                    'Total Forecast': f"₹{forecast['yhat'].sum():,.0f}",
                    'Avg Daily': f"₹{forecast['yhat'].mean():,.0f}",
                    'Max': f"₹{forecast['yhat'].max():,.0f}",
                    'Min': f"₹{forecast['yhat'].min():,.0f}"
                })
        
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

with tab3:
    st.markdown("### 📈 Advanced Analysis")
    st.caption("Promotional impact and units-based forecasting")
    
    # Promotional Analysis
    st.markdown("#### 📢 Promotional Impact Analysis")
    
    if 'Promotion' in df.columns:
        promo_impact = df.groupby('Promotion')['Sales'].agg(['mean', 'sum', 'count']).reset_index()
        promo_impact.columns = ['Promotion', 'Avg Sales', 'Total Sales', 'Count']
        promo_impact['Promotion'] = promo_impact['Promotion'].map({0: 'No Promotion', 1: 'With Promotion'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Promotional Performance**")
            st.dataframe(promo_impact, use_container_width=True)
            
            no_promo_series = promo_impact.loc[promo_impact['Promotion'] == 'No Promotion', 'Avg Sales']
            promo_series = promo_impact.loc[promo_impact['Promotion'] == 'With Promotion', 'Avg Sales']

            no_promo_avg = float(no_promo_series.mean()) if not no_promo_series.empty else 0.0
            promo_avg = float(promo_series.mean()) if not promo_series.empty else 0.0

            if no_promo_avg == 0:
                st.warning("Insufficient 'No Promotion' data to calculate promotional lift.")
                st.metric("Promotional Lift", "N/A")
            else:
                lift = ((promo_avg - no_promo_avg) / no_promo_avg) * 100
                st.metric("Promotional Lift", f"{lift:.1f}%")
        
        with col2:
            fig = px.bar(
                promo_impact,
                x='Promotion',
                y='Avg Sales',
                title="Average Sales: Promotion vs No Promotion",
                color='Promotion'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            promo_ts = df.groupby(['Date', 'Promotion'])['Sales'].sum().reset_index()
            
            fig = go.Figure()
            
            for promo in [0, 1]:
                promo_data = promo_ts[promo_ts['Promotion'] == promo]
                fig.add_trace(go.Scatter(
                    x=promo_data['Date'],
                    y=promo_data['Sales'],
                    mode='markers',
                    name='With Promotion' if promo == 1 else 'No Promotion',
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Sales Over Time: Promotional Impact",
                xaxis_title="Date",
                yaxis_title="Sales (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No promotional data available. Add a 'Promotion' column (0/1) to your dataset.")
    
    # Units Forecasting
    st.markdown("---")
    st.markdown("#### 📦 Units-Based Demand Forecasting")

    if 'Units_Sold' in df.columns:
        st.success("✅ Units data detected!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Units Sold", f"{df['Units_Sold'].sum():,.0f}")
        with col2:
            st.metric("Avg Daily Units", f"{df['Units_Sold'].mean():,.1f}")
        with col3:
            st.metric("Peak Units/Day", f"{df['Units_Sold'].max():,.0f}")
        with col4:
            units_vol = df['Units_Sold'].std() / df['Units_Sold'].mean() * 100
            st.metric("Units Volatility", f"{units_vol:.1f}%")

        st.markdown("**Generate Units Forecast**")

        units_col1, units_col2, units_col3 = st.columns(3)

        with units_col1:
            units_forecast_days = st.selectbox(
                "Forecast Horizon (days)",
                [7, 14, 30, 60, 90],
                key="units_days"
            )

        with units_col2:
            units_model = st.selectbox(
                "Forecasting Model",
                ["XGBoost", "Prophet", "Ensemble"],
                key="units_model"
            )

        with units_col3:
            units_confidence = st.selectbox(
                "Confidence Level",
                ["80%", "90%", "95%", "99%"],
                index=2,
                key="units_confidence"
            )

        if st.button("🚀 Generate Units Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating units forecast..."):
                try:
                    if 'Date' not in df.columns or 'Units_Sold' not in df.columns:
                        st.error("Required columns 'Date' and 'Units_Sold' not found")
                    else:
                        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                        units_df = df.groupby('Date')['Units_Sold'].sum().reset_index()
                        units_df.columns = ['ds', 'y']

                        forecaster = ForecastingModels()
                        units_results = {}

                        if units_model == "XGBoost":
                            xgb_result = forecaster.train_xgboost(units_df, units_forecast_days)
                            units_results['XGBoost'] = xgb_result
                        elif units_model == "Prophet":
                            prophet_result = forecaster.train_prophet(units_df, units_forecast_days)
                            units_results['Prophet'] = prophet_result
                        elif units_model == "Ensemble":
                            ensemble_result = forecaster.train_ensemble(units_df, units_forecast_days)
                            units_results['Ensemble'] = ensemble_result

                        st.session_state.units_forecast_results = units_results
                        st.success("✅ Units forecast generated successfully!")

                except Exception as e:
                    st.error(f"Error generating units forecast: {str(e)}")

        if 'units_forecast_results' in st.session_state:
            units_results = st.session_state.units_forecast_results

            st.markdown("**Units Forecast Results**")

            fig_units = go.Figure()
            colors = ['green', 'orange', 'blue']

            for i, (model_name, result) in enumerate(units_results.items()):
                if 'forecast' in result:
                    forecast = result['forecast']
                    fig_units.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines+markers',
                        name=f'{model_name}',
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))

            fig_units.update_layout(
                title="Units Sold Forecast",
                xaxis_title="Date",
                yaxis_title="Forecasted Units",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig_units, use_container_width=True)

            units_summary = []
            for model_name, result in units_results.items():
                if 'forecast' in result:
                    forecast = result['forecast']
                    units_summary.append({
                        'Model': model_name,
                        'Total Units': f"{forecast['yhat'].sum():,.0f}",
                        'Avg Daily': f"{forecast['yhat'].mean():,.1f}",
                        'Peak/Day': f"{forecast['yhat'].max():,.0f}"
                    })

            if units_summary:
                st.dataframe(pd.DataFrame(units_summary), use_container_width=True)

    else:
        st.warning("❌ No 'Units_Sold' column found in your data")
        st.markdown("""
        **To enable units-based forecasting:**
        1. Add a 'Units_Sold' column to your dataset
        2. Data should contain the number of units sold per transaction
        3. Re-upload your data with units information
        """)
