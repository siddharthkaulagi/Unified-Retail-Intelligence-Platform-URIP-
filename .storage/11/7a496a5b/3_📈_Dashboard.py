import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

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
        default=list(results.keys())
    )

with col2:
    chart_type = st.selectbox(
        "Chart Type",
        ["Line Chart", "Area Chart", "Bar Chart"]
    )

with col3:
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)

with col4:
    show_metrics = st.checkbox("Show Metrics", value=True)

# Key Metrics Overview
st.markdown("### 📊 Key Metrics Overview")

if show_metrics:
    metrics_cols = st.columns(len(selected_models))
    
    for i, model_name in enumerate(selected_models):
        if model_name in results and 'metrics' in results[model_name]:
            with metrics_cols[i]:
                st.markdown(f"#### {model_name}")
                metrics = results[model_name]['metrics']
                
                st.metric("MAE", f"{metrics.get('mae', 0):.2f}")
                st.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
                st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

# Main Forecast Visualization
st.markdown("### 📈 Forecast Visualization")

fig = go.Figure()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i, model_name in enumerate(selected_models):
    if model_name not in results:
        continue
        
    result = results[model_name]
    color = colors[i % len(colors)]
    
    # Add historical data
    if 'historical' in result:
        hist_df = result['historical']
        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(
                x=hist_df['ds'],
                y=hist_df['y'],
                mode='lines',
                name=f'{model_name} - Historical',
                line=dict(color=color, width=2),
                opacity=0.7
            ))
        elif chart_type == "Area Chart":
            fig.add_trace(go.Scatter(
                x=hist_df['ds'],
                y=hist_df['y'],
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                name=f'{model_name} - Historical',
                line=dict(color=color),
                opacity=0.6
            ))
    
    # Add forecast
    if 'forecast' in result:
        forecast_df = result['forecast']
        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                name=f'{model_name} - Forecast',
                line=dict(color=color, width=3, dash='dash')
            ))
        elif chart_type == "Area Chart":
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                name=f'{model_name} - Forecast',
                line=dict(color=color, dash='dash'),
                opacity=0.6
            ))
        elif chart_type == "Bar Chart":
            fig.add_trace(go.Bar(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                name=f'{model_name} - Forecast',
                marker_color=color,
                opacity=0.7
            ))
        
        # Add confidence intervals
        if show_confidence and 'yhat_lower' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
                y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
                fill='tonexty',
                fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{model_name} - Confidence',
                showlegend=False
            ))

fig.update_layout(
    title="Sales Forecast Dashboard",
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode='x unified',
    height=600,
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)

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
                'MAPE': metrics.get('mape', 0)
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create metrics comparison chart
        fig_metrics = go.Figure()
        
        fig_metrics.add_trace(go.Bar(
            name='MAE',
            x=metrics_df['Model'],
            y=metrics_df['MAE'],
            yaxis='y'
        ))
        
        fig_metrics.add_trace(go.Bar(
            name='RMSE',
            x=metrics_df['Model'],
            y=metrics_df['RMSE'],
            yaxis='y'
        ))
        
        fig_metrics.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Error Value",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)

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
        
        st.dataframe(summary_df, use_container_width=True)

# Detailed Analysis
st.markdown("### 🔬 Detailed Analysis")

analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Trend Analysis", "Seasonality", "Residuals"])

with analysis_tab1:
    st.markdown("#### Trend Analysis")
    
    # Calculate trends for each model
    trend_data = []
    for model_name in selected_models:
        if model_name in results and 'forecast' in results[model_name]:
            forecast_df = results[model_name]['forecast']
            if len(forecast_df) > 1:
                # Calculate trend (slope)
                x = np.arange(len(forecast_df))
                y = forecast_df['yhat'].values
                trend = np.polyfit(x, y, 1)[0]
                
                trend_data.append({
                    'Model': model_name,
                    'Trend (Daily)': trend,
                    'Trend (%)': (trend / y.mean()) * 100 if y.mean() != 0 else 0
                })
    
    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        fig_trend = px.bar(
            trend_df, 
            x='Model', 
            y='Trend (Daily)',
            title="Daily Trend Comparison",
            color='Trend (Daily)',
            color_continuous_scale='RdYlGn'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.dataframe(trend_df, use_container_width=True)

with analysis_tab2:
    st.markdown("#### Seasonality Analysis")
    
    # Analyze weekly patterns
    seasonality_data = []
    for model_name in selected_models:
        if model_name in results and 'forecast' in results[model_name]:
            forecast_df = results[model_name]['forecast'].copy()
            forecast_df['day_of_week'] = pd.to_datetime(forecast_df['ds']).dt.day_name()
            
            weekly_avg = forecast_df.groupby('day_of_week')['yhat'].mean()
            
            for day, avg_sales in weekly_avg.items():
                seasonality_data.append({
                    'Model': model_name,
                    'Day': day,
                    'Avg Sales': avg_sales
                })
    
    if seasonality_data:
        seasonality_df = pd.DataFrame(seasonality_data)
        
        fig_seasonality = px.line(
            seasonality_df,
            x='Day',
            y='Avg Sales',
            color='Model',
            title="Weekly Seasonality Pattern"
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_seasonality.update_xaxis(categoryorder='array', categoryarray=day_order)
        
        st.plotly_chart(fig_seasonality, use_container_width=True)

with analysis_tab3:
    st.markdown("#### Residual Analysis")
    st.info("Residual analysis helps identify model accuracy and potential issues")
    
    # Show residuals for models with historical data
    for model_name in selected_models:
        if model_name in results and 'historical' in results[model_name]:
            st.markdown(f"##### {model_name} Residuals")
            
            # This would require actual vs predicted on historical data
            # For now, show a placeholder
            st.info(f"Residual analysis for {model_name} - Feature available in full version")

# Business Insights
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
                st.write(f"- Total Predicted Sales: ${total_forecast:,.2f}")
                st.write(f"- Average Daily Sales: ${avg_daily:,.2f}")
                
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

# Export options
st.markdown("### 📤 Export Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Chart Data", use_container_width=True):
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
    if st.button("📈 Export Metrics", use_container_width=True):
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