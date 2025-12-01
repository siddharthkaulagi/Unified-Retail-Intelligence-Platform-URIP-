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
from utils.ui_components import render_sidebar


# --- PATHS ---
# Get the absolute path of the directory containing the script's parent
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_FILE = os.path.join(ASSETS_DIR, "custom.css")


def load_css():
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()



st.set_page_config(page_title="Model Selection", page_icon="🔮", layout="wide")
# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

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
    use_arima = st.checkbox("ARIMA (Classical Time Series)", value=True)

    with col2:
        st.markdown("#### Machine Learning Models")
        use_rf = st.checkbox("Random Forest", value=True)
        use_lgb = st.checkbox("LightGBM (Recommended)", value=True)
        use_xgb = st.checkbox("XGBoost", value=False)
        use_ensemble = st.checkbox("Ensemble Methods", value=False, help="Combines multiple models")


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
if st.button("🚀 Run Forecasting", type="primary", width='stretch'):
    if not any([use_arima, use_rf, use_lgb, use_xgb, use_ensemble]):
        st.error("Please select at least one model!")
        st.stop()
    
    with st.spinner("Preparing data and training models..."):
        try:
            # Prepare data based on aggregation level
            if date_col not in df.columns:
                st.error("Date column not found!")
                st.stop()
            
            # Convert date column
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            
            # Aggregate data
            # Aggregate data
            group_cols = []
            if agg_level == "Total":
                agg_df = df.groupby(date_col)['Sales'].sum().reset_index()
                agg_df.columns = ['ds', 'y']
            elif agg_level == "By Store" and 'Store' in df.columns:
                agg_df = df.groupby([date_col, 'Store'])['Sales'].sum().reset_index()
                group_cols = ['Store']
            elif agg_level == "By Category" and 'Category' in df.columns:
                agg_df = df.groupby([date_col, 'Category'])['Sales'].sum().reset_index()
                group_cols = ['Category']
            elif agg_level == "By Store & Category" and 'Store' in df.columns and 'Category' in df.columns:
                agg_df = df.groupby([date_col, 'Store', 'Category'])['Sales'].sum().reset_index()
                group_cols = ['Store', 'Category']
            else:
                # Default to total
                agg_df = df.groupby(date_col)['Sales'].sum().reset_index()
                agg_df.columns = ['ds', 'y']
            
            # Initialize forecasting models
            forecaster = ForecastingModels()
            
            # Store results
            results = {}

            # Helper function to train models with groups
            def train_model_with_groups(train_func, df, group_cols, **kwargs):
                if not group_cols:
                    # No grouping, just train directly
                    return train_func(df, **kwargs)
                
                # Grouped training
                all_forecasts = []
                all_historical = []
                metrics_list = []
                
                # Get unique groups
                groups = df.groupby(group_cols)
                
                for name, group in groups:
                    # Prepare group data
                    group_df = group.copy()
                    
                    # Skip if not enough data
                    if len(group_df) < 5:
                        print(f"Skipping group {name}: Insufficient data ({len(group_df)} rows)")
                        continue

                    # Rename columns for model
                    group_df = group_df.rename(columns={date_col: 'ds', 'Sales': 'y'})
                    
                    # Train model for this group
                    try:
                        res = train_func(group_df, **kwargs)
                        
                        # Add group info back to results
                        if 'forecast' in res:
                            f = res['forecast'].copy()
                            if isinstance(name, tuple):
                                for i, col in enumerate(group_cols):
                                    f[col] = name[i]
                            else:
                                f[group_cols[0]] = name
                            all_forecasts.append(f)
                            
                        if 'historical' in res:
                            h = res['historical'].copy()
                            if isinstance(name, tuple):
                                for i, col in enumerate(group_cols):
                                    h[col] = name[i]
                            else:
                                h[group_cols[0]] = name
                            all_historical.append(h)
                            
                        if 'metrics' in res:
                            metrics_list.append(res['metrics'])
                            
                    except Exception as e:
                        print(f"Error training group {name}: {e}")
                        continue
                
                # Combine results
                if not all_forecasts:
                    return None
                    
                combined_forecast = pd.concat(all_forecasts, ignore_index=True)
                combined_historical = pd.concat(all_historical, ignore_index=True)
                
                # Calculate global metrics by averaging group metrics
                # This avoids the issue of non-overlapping forecast/history dates
                global_metrics = {}
                if metrics_list:
                    # Get all metric keys (mae, rmse, etc.)
                    metric_keys = metrics_list[0].keys()
                    
                    for key in metric_keys:
                        # Filter out None values
                        values = [m[key] for m in metrics_list if m.get(key) is not None]
                        if values:
                            global_metrics[key] = sum(values) / len(values)
                        else:
                            global_metrics[key] = 0.0
                else:
                    global_metrics = {'mae': 0, 'rmse': 0, 'mape': 0, 'r2': 0}
                
                return {
                    'model': 'Grouped Model',
                    'forecast': combined_forecast,
                    'historical': combined_historical,
                    'metrics': global_metrics
                }
            
            # Train and forecast with selected models
            if use_arima:
                with st.spinner("Training ARIMA model..."):
                    arima_result = train_model_with_groups(
                        forecaster.train_arima,
                        agg_df, group_cols,
                        forecast_days=forecast_days
                    )
                    if arima_result:
                        results['ARIMA'] = arima_result

            
            if use_rf:
                with st.spinner("Training Random Forest model..."):
                    rf_result = train_model_with_groups(
                        forecaster.train_random_forest,
                        agg_df, group_cols,
                        forecast_days=forecast_days,
                        train_split=train_split
                    )
                    if rf_result:
                        results['Random Forest'] = rf_result
            
            if use_lgb:
                with st.spinner("Training LightGBM model..."):
                    lgb_result = train_model_with_groups(
                        forecaster.train_lightgbm,
                        agg_df, group_cols,
                        forecast_days=forecast_days,
                        train_split=train_split
                    )
                    if lgb_result:
                        results['LightGBM'] = lgb_result
            if use_xgb:
                with st.spinner("Training XGBoost model..."):
                    xgb_result = train_model_with_groups(
                        forecaster.train_xgboost,
                        agg_df, group_cols,
                        forecast_days=forecast_days,
                        train_split=train_split
                    )
                    if xgb_result:
                        results['XGBoost'] = xgb_result

            if use_ensemble:
                with st.spinner("Training Ensemble model..."):
                    ensemble_result = train_model_with_groups(
                        forecaster.train_ensemble,
                        agg_df, group_cols,
                        forecast_days=forecast_days,
                        train_split=train_split
                    )
                    if ensemble_result:
                        results['Ensemble'] = ensemble_result

            
            if not results:
                st.error("❌ No models could be trained. This usually happens when there is insufficient data for the selected aggregation level. Please try a different aggregation or check your data.")
                st.stop()

            # Store results in session state
            st.session_state.forecast_results = results
            st.session_state.forecast_config = {
                'agg_level': agg_level,
                'forecast_days': forecast_days,
                'date_col': date_col,
                'models_used': [name for name, use in [
                    ('ARIMA', use_arima),
                    ('Random Forest', use_rf),
                    ('LightGBM', use_lgb),
                    ('XGBoost', use_xgb),
                    ('Ensemble Methods', use_ensemble)
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
                'MAPE': f"{round(metrics.get('mape', 0), 2)}%" if metrics.get('mape') is not None else 'N/A',
                'R2': round(metrics.get('r2', float('nan')), 4)
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, width='stretch')
        
        # Best model
        best_model = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
        st.success(f"🏆 Best performing model: **{best_model}** (lowest MAE)")
    
    # Forecast visualization
    st.markdown("#### 📊 Forecast Visualization")
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    color_map = {
        'blue': '0, 0, 255',
        'red': '255, 0, 0',
        'green': '0, 128, 0',
        'orange': '255, 165, 0',
        'purple': '128, 0, 128'
    }
    
    # First, add actual sales data (only once, not per model)
    first_model = list(results.keys())[0]
    if 'historical' in results[first_model]:
        hist_df = results[first_model]['historical'].copy()
        
        # Limit to last 9 months for cleaner visualization
        if len(hist_df) > 270:  # ~9 months
            hist_df = hist_df.tail(270)
        
        # Add actual sales line
        fig.add_trace(go.Scatter(
            x=hist_df['ds'],
            y=hist_df['y'],
            mode='lines',
            name='Actual Sales',
            line=dict(color='black', width=2),
            opacity=0.8
        ))
    
    # Then add fitted values and forecasts for each model
    for i, (model_name, result) in enumerate(results.items()):
        if 'forecast' in result:
            forecast_df = result['forecast']
            
            # Add fitted values (model predictions on historical data)
            if 'fitted_values' in result:
                fitted_df = result['fitted_values']
                
                # Limit fitted values to last 9 months as well
                if len(fitted_df) > 270:
                    fitted_df = fitted_df.tail(270)
                
                fig.add_trace(go.Scatter(
                    x=fitted_df['ds'],
                    y=fitted_df['yhat'],
                    mode='lines',
                    name=f'{model_name} - Fitted',
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.6
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
                    fillcolor=f'rgba({color_map[colors[i % len(colors)]]}, 0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{model_name} - Confidence Interval',
                    showlegend=False
                ))
    
    fig.update_layout(
        title="Sales Forecast Comparison (Actual vs Fitted vs Forecast)",
        xaxis_title="Date",
        yaxis_title="Sales (₹)",
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, width='stretch')
    
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
                # Display selected metrics including R2
                st.metric('MAE', f"{metrics.get('mae', 0):.2f}")
                st.metric('RMSE', f"{metrics.get('rmse', 0):.2f}")
                mape_val = metrics.get('mape', None)
                st.metric('MAPE', f"{mape_val:.2f}%" if mape_val is not None else 'N/A')
                r2_val = metrics.get('r2', None)
                st.metric('R2', f"{r2_val:.4f}" if r2_val is not None else 'N/A')
        
        with col2:
            st.markdown(f"##### {selected_model} - Forecast Summary")
            if 'forecast' in model_result:
                forecast = model_result['forecast']
                st.metric("Forecast Period", f"{len(forecast)} days")
                st.metric("Avg Daily Sales", f"₹{forecast['yhat'].mean():.2f}")
                st.metric("Total Forecast", f"₹{forecast['yhat'].sum():.2f}")

else:
    st.info("👆 Configure your models and click 'Run Forecasting' to see results")
    
    # Model information
    st.markdown("### 📚 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""

        #### ⚡ XGBoost
        - **Best for**: High-performance predictive modeling
        - **Pros**: Extremely fast, handles missing data, regularization reduces overfitting
        - **Cons**: Complex tuning, less interpretable than simpler models
        
        #### 📈 ARIMA
        - **Best for**: Classical time series
        - **Pros**: Statistical foundation, interpretable
        - **Cons**: Requires stationary data

        #### 🔮 Prophet
        - **Best for**: Seasonal & holiday-driven trends
        - **Pros**: Auto-seasonality, minimal tuning
        - **Cons**: Weaker on complex nonlinear patterns

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

        #### 🤝 Ensemble Models
        - **Best for**: Combining model strengths for stability
        - **Pros**: High accuracy, reduces variance & overfitting
        - **Cons**: More complex and resource-heavy 

        """)
