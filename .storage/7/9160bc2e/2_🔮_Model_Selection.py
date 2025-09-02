import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import ForecastingModels

st.set_page_config(page_title="Model Selection", page_icon="🔮", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

st.title("🔮 Model Selection & Forecasting")
st.markdown("Select forecasting models and generate predictions for your retail data")

# Check if data is available
if 'uploaded_data' not in st.session_state and 'processed_data' not in st.session_state:
    st.warning("⚠️ No data found. Please upload data first!")
    st.markdown("[👆 Go to Upload Data page](1_📊_Upload_Data)")
    st.stop()

# Get data
if 'processed_data' in st.session_state:
    df = st.session_state.processed_data
else:
    df = st.session_state.uploaded_data

# Data preparation
st.markdown("### 📊 Data Preparation")

col1, col2, col3 = st.columns(3)

with col1:
    # Aggregation level
    agg_level = st.selectbox(
        "Aggregation Level",
        ["Total", "By Store", "By Category", "By Store & Category"]
    )

with col2:
    # Forecast horizon
    forecast_days = st.selectbox(
        "Forecast Horizon (Days)",
        [30, 60, 90, 180]
    )

with col3:
    # Date column
    date_col = st.selectbox(
        "Date Column",
        [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
    )

# Model selection
st.markdown("### 🤖 Model Selection")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Classical Models")
    use_prophet = st.checkbox("Prophet (Seasonality Detection)", value=True)
    use_arima = st.checkbox("ARIMA (Classical Time Series)")

with col2:
    st.markdown("#### Machine Learning Models")
    use_rf = st.checkbox("Random Forest", value=True)
    use_lgb = st.checkbox("LightGBM (Recommended)", value=True)

# Advanced options
with st.expander("🔧 Advanced Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        train_split = st.slider("Training Data Split", 0.6, 0.9, 0.8)
        seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
    
    with col2:
        include_holidays = st.checkbox("Include Holidays", value=True)
        cross_validation = st.checkbox("Cross Validation", value=True)

# Run forecasting
if st.button("🚀 Run Forecasting", type="primary", use_container_width=True):
    if not any([use_prophet, use_arima, use_rf, use_lgb]):
        st.error("Please select at least one model!")
        st.stop()
    
    with st.spinner("Preparing data and training models..."):
        try:
            # Prepare data based on aggregation level
            if date_col not in df.columns:
                st.error("Date column not found!")
                st.stop()
            
            # Convert date column
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Aggregate data
            if agg_level == "Total":
                agg_df = df.groupby(date_col)['Sales'].sum().reset_index()
                agg_df.columns = ['ds', 'y']
            elif agg_level == "By Store" and 'Store' in df.columns:
                agg_df = df.groupby([date_col, 'Store'])['Sales'].sum().reset_index()
            elif agg_level == "By Category" and 'Category' in df.columns:
                agg_df = df.groupby([date_col, 'Category'])['Sales'].sum().reset_index()
            else:
                # Default to total
                agg_df = df.groupby(date_col)['Sales'].sum().reset_index()
                agg_df.columns = ['ds', 'y']
            
            # Initialize forecasting models
            forecaster = ForecastingModels()
            
            # Store results
            results = {}
            
            # Train and forecast with selected models
            if use_prophet:
                with st.spinner("Training Prophet model..."):
                    prophet_result = forecaster.train_prophet(
                        agg_df, forecast_days, seasonality_mode, include_holidays
                    )
                    results['Prophet'] = prophet_result
            
            if use_rf:
                with st.spinner("Training Random Forest model..."):
                    rf_result = forecaster.train_random_forest(
                        agg_df, forecast_days, train_split
                    )
                    results['Random Forest'] = rf_result
            
            if use_lgb:
                with st.spinner("Training LightGBM model..."):
                    lgb_result = forecaster.train_lightgbm(
                        agg_df, forecast_days, train_split
                    )
                    results['LightGBM'] = lgb_result
            
            # Store results in session state
            st.session_state.forecast_results = results
            st.session_state.forecast_config = {
                'agg_level': agg_level,
                'forecast_days': forecast_days,
                'date_col': date_col,
                'models_used': [name for name, use in [
                    ('Prophet', use_prophet),
                    ('Random Forest', use_rf),
                    ('LightGBM', use_lgb)
                ] if use]
            }
            
            st.success("✅ Forecasting completed successfully!")
            
        except Exception as e:
            st.error(f"Error during forecasting: {str(e)}")
            st.stop()

# Display results if available
if 'forecast_results' in st.session_state:
    results = st.session_state.forecast_results
    
    st.markdown("### 📈 Forecasting Results")
    
    # Model comparison metrics
    st.markdown("#### 🎯 Model Performance Comparison")
    
    metrics_data = []
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'MAE': round(metrics.get('mae', 0), 2),
                'RMSE': round(metrics.get('rmse', 0), 2),
                'MAPE': f"{round(metrics.get('mape', 0), 2)}%"
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Best model
        best_model = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
        st.success(f"🏆 Best performing model: **{best_model}** (lowest MAE)")
    
    # Forecast visualization
    st.markdown("#### 📊 Forecast Visualization")
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, result) in enumerate(results.items()):
        if 'forecast' in result:
            forecast_df = result['forecast']
            
            # Add historical data (if available)
            if 'historical' in result:
                hist_df = result['historical']
                fig.add_trace(go.Scatter(
                    x=hist_df['ds'],
                    y=hist_df['y'],
                    mode='lines',
                    name=f'{model_name} - Historical',
                    line=dict(color=colors[i % len(colors)], width=1),
                    opacity=0.7
                ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                name=f'{model_name} - Forecast',
                line=dict(color=colors[i % len(colors)], width=3, dash='dash')
            ))
            
            # Add confidence intervals if available
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
                    y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
                    fill='tonexty',
                    fillcolor=f'rgba({colors[i % len(colors)]}, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} - Confidence Interval',
                    showlegend=False
                ))
    
    fig.update_layout(
        title="Sales Forecast Comparison",
        xaxis_title="Date",
        yaxis_title="Sales",
        hovermode='x unified',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual model details
    st.markdown("#### 🔍 Individual Model Details")
    
    selected_model = st.selectbox(
        "Select model for detailed view",
        list(results.keys())
    )
    
    if selected_model in results:
        model_result = results[selected_model]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"##### {selected_model} - Metrics")
            if 'metrics' in model_result:
                metrics = model_result['metrics']
                for metric, value in metrics.items():
                    if metric == 'mape':
                        st.metric(metric.upper(), f"{value:.2f}%")
                    else:
                        st.metric(metric.upper(), f"{value:.2f}")
        
        with col2:
            st.markdown(f"##### {selected_model} - Forecast Summary")
            if 'forecast' in model_result:
                forecast = model_result['forecast']
                st.metric("Forecast Period", f"{len(forecast)} days")
                st.metric("Avg Daily Sales", f"{forecast['yhat'].mean():.2f}")
                st.metric("Total Forecast", f"{forecast['yhat'].sum():.2f}")

else:
    st.info("👆 Configure your models and click 'Run Forecasting' to see results")
    
    # Model information
    st.markdown("### 📚 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔮 Prophet
        - **Best for**: Seasonal patterns, holidays
        - **Pros**: Handles missing data, interpretable
        - **Cons**: Slower for large datasets
        
        #### 📈 ARIMA
        - **Best for**: Classical time series
        - **Pros**: Statistical foundation, interpretable
        - **Cons**: Requires stationary data
        """)
    
    with col2:
        st.markdown("""
        #### 🌳 Random Forest
        - **Best for**: Non-linear patterns
        - **Pros**: Robust, handles outliers
        - **Cons**: Less interpretable
        
        #### 🚀 LightGBM
        - **Best for**: High accuracy (recommended)
        - **Pros**: Fast, accurate, handles features
        - **Cons**: Requires feature engineering
        """)