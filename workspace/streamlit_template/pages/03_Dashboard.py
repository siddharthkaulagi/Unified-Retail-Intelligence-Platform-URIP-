import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os
try:
    from statsmodels.tsa.seasonal import STL
    stl_available = True
except Exception:
    stl_available = False

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


st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
# Sidebar
render_sidebar()

st.title("📈 Forecast Dashboard")
st.markdown("Interactive dashboard for analyzing forecasting results and insights")

# Check if forecast results are available
if 'forecast_results' not in st.session_state:
    st.warning("⚠️ No forecast results found. Please run forecasting first!")
    st.markdown("[👆 Go to Model Selection page](2_🔮_Model_Selection)")
    st.stop()

results = st.session_state.forecast_results
config = st.session_state.get('forecast_config', {})

# Dashboard filters
st.markdown("### 🎛️ Dashboard Filters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_models = st.multiselect(
        "Select Models",
        list(results.keys()),
        default=[]
    )

with col2:
    chart_type = st.selectbox(
        "Chart Type",
        ["Line Chart", "Bar Chart"]
    )

with col3:
    show_confidence = st.checkbox("Show Confidence Intervals", value=False)

with col4:
    show_metrics = st.checkbox("Show Metrics", value=False)

# Key Metrics Overview
st.markdown("### 📊 Key Metrics Overview")

if show_metrics:
    metrics_cols = st.columns(len(selected_models))
    
    for i, model_name in enumerate(selected_models):
        if model_name in results and 'metrics' in results[model_name]:
         with metrics_cols[i]:
            st.markdown(f"#### {model_name}")
            metrics = results[model_name]['metrics']

            mae = metrics.get('mae') or 0
            rmse = metrics.get('rmse') or 0
            mape = metrics.get('mape') or 0
            r2 = metrics.get('r2') or 0 

            st.metric("MAE", f"{mae:.4f}")
            st.metric("RMSE", f"{rmse:.4f}")
            if mape is not None:
                st.metric("MAPE", f"{mape:.3f}%")
            else:
                st.metric("MAPE", "N/A")
            if r2 is not None:
                st.metric("R²", f"{r2:.4f}")
            else:
                st.metric("R²", "N/A")



# Main Forecast Visualization
st.markdown("### 📈 Forecast Visualization")

# Create comprehensive visualization with Actual, Fitted, and Forecast
if selected_models:
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
    first_model = selected_models[0]
    if first_model in results and 'historical' in results[first_model]:
        hist_df = results[first_model]['historical'].copy()
        
        # Convert date column
        if 'ds' in hist_df.columns:
            hist_df['ds'] = pd.to_datetime(hist_df['ds'], errors='coerce')
        if 'y' in hist_df.columns:
            hist_df['y'] = pd.to_numeric(hist_df['y'], errors='coerce')
        
        # Limit to last 10 months for cleaner visualization
        if len(hist_df) > 300:  # ~10 months
            hist_df = hist_df.tail(300)
        
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
    for i, model_name in enumerate(selected_models):
        if model_name in results:
            result = results[model_name]
            
            # Add fitted values (model predictions on historical data)
            if 'fitted_values' in result:
                fitted_df = result['fitted_values'].copy()
                
                # Convert date column
                if 'ds' in fitted_df.columns:
                    fitted_df['ds'] = pd.to_datetime(fitted_df['ds'], errors='coerce')
                if 'yhat' in fitted_df.columns:
                    fitted_df['yhat'] = pd.to_numeric(fitted_df['yhat'], errors='coerce')
                
                # Limit fitted values to last 10 months as well
                if len(fitted_df) > 300:
                    fitted_df = fitted_df.tail(300)
                
                fig.add_trace(go.Scatter(
                    x=fitted_df['ds'],
                    y=fitted_df['yhat'],
                    mode='lines',
                    name=f'{model_name} - Fitted',
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.6
                ))
            
            # Add forecast
            if 'forecast' in result:
                forecast_df = result['forecast'].copy()
                
                # Convert date column
                if 'ds' in forecast_df.columns:
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'], errors='coerce')
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    if col in forecast_df.columns:
                        forecast_df[col] = pd.to_numeric(forecast_df[col], errors='coerce')
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name=f'{model_name} - Forecast',
                    line=dict(color=colors[i % len(colors)], width=3, dash='dash')
                ))
                
                # Add confidence intervals if available and selected
                if show_confidence and 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
                        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
                        fill='tonexty',
                        fillcolor=f'rgba({color_map[colors[i % len(colors)]]}, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{model_name} - Confidence Interval',
                        showlegend=False
                    ))
    
    # Update layout
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical details
    if first_model in results and 'historical' in results[first_model]:
        historical_data = results[first_model]['historical']
        if 'y' in historical_data.columns:
            st.markdown("#### 📊 Statistical Details")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                mean_val = historical_data['y'].mean()
                st.metric("Historical Mean", f"₹{mean_val:,.0f}")

            with col2:
                std_val = historical_data['y'].std()
                st.metric("Standard Deviation", f"₹{std_val:,.0f}")

            with col3:
                # Calculate outliers
                Q1 = historical_data['y'].quantile(0.25)
                Q3 = historical_data['y'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_count = len(historical_data[(historical_data['y'] < lower_bound) | (historical_data['y'] > upper_bound)])
                st.metric("Outliers Detected", outliers_count)

            with col4:
                if first_model in results and 'forecast' in results[first_model]:
                    forecast_data = results[first_model]['forecast']
                    if 'yhat' in forecast_data.columns:
                        forecast_mean = forecast_data['yhat'].mean()
                        st.metric("Forecast Mean", f"₹{forecast_mean:,.0f}")

            # Category TreeMap
            st.markdown("#### 📦 Sales by Category")
            
            data_df = st.session_state.get('uploaded_data')
            if data_df is not None and 'Category' in data_df.columns and 'Sales' in data_df.columns:
                category_sales = data_df.groupby('Category')['Sales'].sum().reset_index()
                
                fig_tree = px.treemap(
                    category_sales,
                    path=['Category'],
                    values='Sales',
                    title="Category Sales Distribution",
                    color='Sales',
                    color_continuous_scale='Blues'
                )
                
                fig_tree.update_layout(height=400)
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("Category data not available")



# Model Comparison Analysis
st.markdown("### 🔍 Model Comparison Analysis")

col1, col2 = st.columns(2)

with col1:
    # Metrics comparison
    st.markdown("#### Performance Metrics Comparison")
    
    metrics_data = []
    for model_name in selected_models:
        if model_name in results and 'metrics' in results[model_name]:
            metrics = results[model_name]['metrics']
            metrics_data.append({
                'Model': model_name,
                'MAE': metrics.get('mae', 0),
                'RMSE': metrics.get('rmse', 0),
                'MAPE': metrics.get('mape', 0),
                'R2': round(metrics.get('r2', 0), 4) if metrics.get('r2') is not None else 'N/A'
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)

        # Display metrics table instead of chart for cleaner look
        st.dataframe(metrics_df, width='stretch', hide_index=True)

with col2:
    # Forecast summary statistics
    st.markdown("#### Forecast Summary Statistics")
    
    summary_data = []
    for model_name in selected_models:
        if model_name in results and 'forecast' in results[model_name]:
            forecast_df = results[model_name]['forecast']
            summary_data.append({
                'Model': model_name,
                'Avg Daily Sales': forecast_df['yhat'].mean(),
                'Total Forecast': forecast_df['yhat'].sum(),
                'Min Forecast': forecast_df['yhat'].min(),
                'Max Forecast': forecast_df['yhat'].max(),
                'Std Dev': forecast_df['yhat'].std()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Format numbers
        for col in ['Avg Daily Sales', 'Total Forecast', 'Min Forecast', 'Max Forecast', 'Std Dev']:
            summary_df[col] = summary_df[col].round(2)
        
        st.dataframe(summary_df, width='stretch')


st.markdown("### 💼 Business Insights & Recommendations")

insights_col1, insights_col2 = st.columns(2)

with insights_col1:
    st.markdown("#### 📈 Key Insights")
    
    if results:
        # Find best performing model
        best_model = None
        best_mae = float('inf')
        
        for model_name, result in results.items():
            if 'metrics' in result:
                mae = result['metrics'].get('mae', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
        
        if best_model:
            st.success(f"🏆 **Best Model**: {best_model} (MAE: {best_mae:.2f})")
            
            # Get forecast summary
            if 'forecast' in results[best_model]:
                forecast_df = results[best_model]['forecast']
                total_forecast = forecast_df['yhat'].sum()
                avg_daily = forecast_df['yhat'].mean()
                
                st.info(f"📊 **Forecast Summary** ({config.get('forecast_days', 30)} days)")
                st.write(f"- Total Predicted Sales: ₹{total_forecast:,.2f}")
                st.write(f"- Average Daily Sales: ₹{avg_daily:,.2f}")
                
                # Calculate growth
                if 'historical' in results[best_model]:
                    hist_df = results[best_model]['historical']
                    recent_avg = hist_df['y'].tail(30).mean()
                    growth = ((avg_daily - recent_avg) / recent_avg) * 100
                    
                    if growth > 0:
                        st.write(f"- Predicted Growth: +{growth:.1f}% vs recent avg")
                    else:
                        st.write(f"- Predicted Change: {growth:.1f}% vs recent avg")

with insights_col2:
    st.markdown("#### 🎯 Recommendations")
    
    recommendations = [
        "📦 **Inventory Planning**: Use forecast to optimize stock levels",
        "🛒 **Procurement**: Plan purchases based on predicted demand",
        "👥 **Staffing**: Adjust workforce according to sales forecasts",
        "📊 **Performance**: Monitor actual vs predicted for model improvement",
        "🎯 **Marketing**: Plan campaigns during predicted high-demand periods"
    ]
    
    for rec in recommendations:
        st.write(rec)
    
    # Model-specific recommendations
    if best_model:
        st.markdown("#### 🔧 Model-Specific Notes")
        
        if "Prophet" in best_model:
            st.write("✅ Good for seasonal patterns and holiday effects")
        elif "LightGBM" in best_model:
            st.write("✅ Excellent for complex patterns and feature interactions")
        elif "Random Forest" in best_model:
            st.write("✅ Robust and interpretable for business decisions")
        
        st.write("💡 Consider ensemble methods for improved accuracy")

# Advanced Analytics - STL Decomposition
if stl_available and results:
    with st.expander("🔬 Advanced Analytics: Sales Decomposition (Trend, Seasonality, Residuals)"):
        st.markdown("#### STL Decomposition Analysis")
        st.caption("Understand sales patterns by separating Trend (long-term direction), Seasonality (repeating patterns), and Residuals (noise)")
        
        # Get historical data from best model
        best_model = None
        best_mae = float('inf')
        for model_name, result in results.items():
            if 'metrics' in result:
                mae = result['metrics'].get('mae', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
        
        if best_model and 'historical' in results[best_model]:
            hist_df = results[best_model]['historical'].copy()
            
            # Prepare data for STL
            if 'ds' in hist_df.columns and 'y' in hist_df.columns:
                hist_df['ds'] = pd.to_datetime(hist_df['ds'])
                hist_df = hist_df.set_index('ds').sort_index()
                
                # Need at least 2 complete seasonal periods (assume weekly seasonality = 14 days minimum)
                if len(hist_df) >= 14:
                    try:
                        # Perform STL decomposition
                        stl = STL(hist_df['y'], seasonal=7, trend=None)
                        result_stl = stl.fit()
                        
                        # Create subplot with 4 rows
                        fig_stl = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=('Original Sales', 'Trend Component', 'Seasonal Component', 'Residual Component'),
                            vertical_spacing=0.08
                        )
                        
                        # Original time series
                        fig_stl.add_trace(
                            go.Scatter(x=hist_df.index, y=hist_df['y'], mode='lines', name='Original',
                                      line=dict(color='#1f77b4', width=2)),
                            row=1, col=1
                        )
                        
                        # Trend
                        fig_stl.add_trace(
                            go.Scatter(x=hist_df.index, y=result_stl.trend, mode='lines', name='Trend',
                                      line=dict(color='#ff7f0e', width=2)),
                            row=2, col=1
                        )
                        
                        # Seasonal
                        fig_stl.add_trace(
                            go.Scatter(x=hist_df.index, y=result_stl.seasonal, mode='lines', name='Seasonal',
                                      line=dict(color='#2ca02c', width=2)),
                            row=3, col=1
                        )
                        
                        # Residual
                        fig_stl.add_trace(
                            go.Scatter(x=hist_df.index, y=result_stl.resid, mode='lines', name='Residual',
                                      line=dict(color='#d62728', width=1)),
                            row=4, col=1
                        )
                        
                        fig_stl.update_layout(
                            height=800,
                            showlegend=False,
                            template='plotly_white',
                            title_text="STL Decomposition: Separating Sales into Components"
                        )
                        
                        fig_stl.update_xaxes(title_text="Date", row=4, col=1)
                        fig_stl.update_yaxes(title_text="Sales (₹)", row=1, col=1)
                        fig_stl.update_yaxes(title_text="Trend (₹)", row=2, col=1)
                        fig_stl.update_yaxes(title_text="Seasonal Effect (₹)", row=3, col=1)
                        fig_stl.update_yaxes(title_text="Residuals (₹)", row=4, col=1)
                        
                        st.plotly_chart(fig_stl, use_container_width=True)
                        
                        # Insights
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            trend_direction = "increasing" if result_stl.trend.iloc[-1] > result_stl.trend.iloc[0] else "decreasing"
                            st.info(f"📈 **Trend**: Sales are {trend_direction} over time")
                        with col2:
                            seasonal_strength = result_stl.seasonal.std() / hist_df['y'].std() * 100
                            st.info(f"🔄 **Seasonality Strength**: {seasonal_strength:.1f}%")
                        with col3:
                            residual_variance = result_stl.resid.std() / hist_df['y'].mean() * 100
                            st.info(f"🎲 **Noise Level**: {residual_variance:.1f}%")
                    
                    except Exception as e:
                        st.warning(f"Could not perform STL decomposition: {str(e)}")
                else:
                    st.info("Need at least 14 days of historical data for decomposition analysis")



# Export options
st.markdown("### 📤 Export Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Chart Data", width='stretch'):
        # Prepare export data
        export_data = []
        for model_name in selected_models:
            if model_name in results and 'forecast' in results[model_name]:
                forecast_df = results[model_name]['forecast'].copy()
                forecast_df['Model'] = model_name
                export_data.append(forecast_df)
        
        if export_data:
            combined_df = pd.concat(export_data, ignore_index=True)
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"forecast_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

with col2:
    if st.button("📈 Export Metrics", width='stretch'):
        metrics_export = []
        for model_name in selected_models:
            if model_name in results and 'metrics' in results[model_name]:
                metrics = results[model_name]['metrics']
                metrics['Model'] = model_name
                metrics_export.append(metrics)
        
        if metrics_export:
            metrics_df = pd.DataFrame(metrics_export)
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download Metrics CSV",
                data=csv,
                file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

with col3:
    st.info("📋 Full reports available in Reports page")
