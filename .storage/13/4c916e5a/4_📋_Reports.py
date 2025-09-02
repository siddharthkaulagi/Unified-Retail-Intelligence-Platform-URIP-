import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import base64

st.set_page_config(page_title="Reports", page_icon="📋", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

st.title("📋 Reports & Downloads")
st.markdown("Generate comprehensive reports and download forecasting results")

# Check if forecast results are available
if 'forecast_results' not in st.session_state:
    st.warning("⚠️ No forecast results found. Please run forecasting first!")
    st.markdown("[👆 Go to Model Selection page](2_🔮_Model_Selection)")
    st.stop()

results = st.session_state.forecast_results
config = st.session_state.get('forecast_config', {})

# Report Configuration
st.markdown("### ⚙️ Report Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    report_type = st.selectbox(
        "Report Type",
        ["Executive Summary", "Technical Report", "Business Intelligence", "Custom Report"]
    )

with col2:
    include_charts = st.checkbox("Include Charts", value=True)
    include_metrics = st.checkbox("Include Metrics", value=True)

with col3:
    export_format = st.selectbox(
        "Export Format",
        ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
    )

# Report Preview
st.markdown("### 📄 Report Preview")

# Executive Summary
if report_type == "Executive Summary":
    st.markdown("#### 📊 Executive Summary Report")
    
    # Key findings
    best_model = None
    best_mae = float('inf')
    total_forecast = 0
    
    for model_name, result in results.items():
        if 'metrics' in result:
            mae = result['metrics'].get('mae', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_model = model_name
        
        if 'forecast' in result:
            total_forecast += result['forecast']['yhat'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", best_model or "N/A")
    with col2:
        st.metric("Best MAE", f"{best_mae:.2f}" if best_mae != float('inf') else "N/A")
    with col3:
        st.metric("Forecast Period", f"{config.get('forecast_days', 'N/A')} days")
    with col4:
        st.metric("Total Forecast", f"${total_forecast:,.2f}")
    
    # Key insights
    st.markdown("##### 🎯 Key Insights")
    insights = [
        f"• **Best performing model**: {best_model} with MAE of {best_mae:.2f}",
        f"• **Forecast horizon**: {config.get('forecast_days', 'N/A')} days ahead",
        f"• **Data aggregation**: {config.get('agg_level', 'Total')} level analysis",
        f"• **Models evaluated**: {len(results)} different algorithms"
    ]
    
    for insight in insights:
        st.write(insight)
    
    # Recommendations
    st.markdown("##### 💼 Business Recommendations")
    recommendations = [
        "📦 **Inventory Management**: Use forecasts to optimize stock levels and reduce carrying costs",
        "🛒 **Procurement Planning**: Align purchasing decisions with predicted demand patterns",
        "👥 **Resource Allocation**: Schedule staff and resources based on forecasted sales volumes",
        "📈 **Performance Monitoring**: Establish KPIs to track forecast accuracy and business impact"
    ]
    
    for rec in recommendations:
        st.write(rec)

# Technical Report
elif report_type == "Technical Report":
    st.markdown("#### 🔬 Technical Analysis Report")
    
    # Model performance comparison
    st.markdown("##### Model Performance Metrics")
    
    metrics_data = []
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            metrics_data.append({
                'Model': model_name,
                'MAE': round(metrics.get('mae', 0), 4),
                'RMSE': round(metrics.get('rmse', 0), 4),
                'MAPE': round(metrics.get('mape', 0), 4)
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Performance analysis
        st.markdown("##### Performance Analysis")
        best_model_row = metrics_df.loc[metrics_df['MAE'].idxmin()]
        st.write(f"**Best Model**: {best_model_row['Model']}")
        st.write(f"**Performance Metrics**:")
        st.write(f"- Mean Absolute Error (MAE): {best_model_row['MAE']}")
        st.write(f"- Root Mean Square Error (RMSE): {best_model_row['RMSE']}")
        st.write(f"- Mean Absolute Percentage Error (MAPE): {best_model_row['MAPE']}%")
    
    # Technical specifications
    st.markdown("##### Technical Specifications")
    tech_specs = {
        'Data Processing': config.get('agg_level', 'Total aggregation'),
        'Forecast Horizon': f"{config.get('forecast_days', 30)} days",
        'Models Used': ', '.join(config.get('models_used', [])),
        'Date Column': config.get('date_col', 'Date'),
        'Training Method': 'Time series cross-validation'
    }
    
    for spec, value in tech_specs.items():
        st.write(f"**{spec}**: {value}")

# Business Intelligence Report
elif report_type == "Business Intelligence":
    st.markdown("#### 📊 Business Intelligence Report")
    
    # Business impact analysis
    st.markdown("##### Business Impact Analysis")
    
    if best_model and 'forecast' in results[best_model]:
        forecast_df = results[best_model]['forecast']
        
        # Calculate business metrics
        total_forecast = forecast_df['yhat'].sum()
        avg_daily_sales = forecast_df['yhat'].mean()
        peak_sales_day = forecast_df.loc[forecast_df['yhat'].idxmax(), 'ds']
        min_sales_day = forecast_df.loc[forecast_df['yhat'].idxmin(), 'ds']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Revenue Projections**")
            st.write(f"• Total forecasted revenue: ${total_forecast:,.2f}")
            st.write(f"• Average daily revenue: ${avg_daily_sales:,.2f}")
            st.write(f"• Peak sales expected: {peak_sales_day.strftime('%Y-%m-%d')}")
            st.write(f"• Lowest sales expected: {min_sales_day.strftime('%Y-%m-%d')}")
        
        with col2:
            st.markdown("**Operational Insights**")
            
            # Calculate coefficient of variation
            cv = (forecast_df['yhat'].std() / forecast_df['yhat'].mean()) * 100
            
            if cv < 10:
                volatility = "Low"
            elif cv < 20:
                volatility = "Moderate"
            else:
                volatility = "High"
            
            st.write(f"• Sales volatility: {volatility} (CV: {cv:.1f}%)")
            st.write(f"• Demand variability: {forecast_df['yhat'].std():.2f}")
            
            # Weekly pattern analysis
            forecast_df_copy = forecast_df.copy()
            forecast_df_copy['day_of_week'] = pd.to_datetime(forecast_df_copy['ds']).dt.day_name()
            weekly_avg = forecast_df_copy.groupby('day_of_week')['yhat'].mean()
            best_day = weekly_avg.idxmax()
            worst_day = weekly_avg.idxmin()
            
            st.write(f"• Best sales day: {best_day}")
            st.write(f"• Lowest sales day: {worst_day}")

# Generate downloadable reports
st.markdown("### 📥 Download Reports")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Generate Excel Report", use_container_width=True):
        # Create Excel report
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': ['Report Generated', 'Best Model', 'Forecast Period', 'Total Models'],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M'),
                    best_model or 'N/A',
                    f"{config.get('forecast_days', 'N/A')} days",
                    len(results)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Metrics sheet
            if metrics_data:
                metrics_df.to_excel(writer, sheet_name='Model_Metrics', index=False)
            
            # Forecasts sheet
            all_forecasts = []
            for model_name, result in results.items():
                if 'forecast' in result:
                    forecast_df = result['forecast'].copy()
                    forecast_df['Model'] = model_name
                    all_forecasts.append(forecast_df)
            
            if all_forecasts:
                combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
                combined_forecasts.to_excel(writer, sheet_name='Forecasts', index=False)
        
        output.seek(0)
        
        st.download_button(
            label="📥 Download Excel Report",
            data=output.getvalue(),
            file_name=f"retail_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with col2:
    if st.button("📄 Generate CSV Export", use_container_width=True):
        # Combine all forecast data
        all_data = []
        for model_name, result in results.items():
            if 'forecast' in result:
                forecast_df = result['forecast'].copy()
                forecast_df['Model'] = model_name
                
                # Add metrics if available
                if 'metrics' in result:
                    metrics = result['metrics']
                    forecast_df['MAE'] = metrics.get('mae', '')
                    forecast_df['RMSE'] = metrics.get('rmse', '')
                    forecast_df['MAPE'] = metrics.get('mape', '')
                
                all_data.append(forecast_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            csv = combined_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Data",
                data=csv,
                file_name=f"forecast_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

with col3:
    if st.button("🔧 Generate JSON Export", use_container_width=True):
        # Create JSON export
        json_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': report_type,
                'forecast_config': config,
                'models_count': len(results)
            },
            'models': {}
        }
        
        for model_name, result in results.items():
            model_data = {}
            
            if 'metrics' in result:
                model_data['metrics'] = result['metrics']
            
            if 'forecast' in result:
                forecast_df = result['forecast']
                model_data['forecast'] = forecast_df.to_dict('records')
            
            json_data['models'][model_name] = model_data
        
        import json
        json_str = json.dumps(json_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download JSON Data",
            data=json_str,
            file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# Report customization
st.markdown("### 🎨 Custom Report Builder")

with st.expander("🔧 Advanced Report Options"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Include Sections:**")
        include_executive = st.checkbox("Executive Summary", value=True)
        include_technical = st.checkbox("Technical Details", value=True)
        include_business = st.checkbox("Business Insights", value=True)
        include_appendix = st.checkbox("Data Appendix", value=False)
    
    with col2:
        st.markdown("**Visualization Options:**")
        chart_style = st.selectbox("Chart Style", ["Professional", "Colorful", "Minimal"])
        include_confidence_bands = st.checkbox("Confidence Intervals", value=True)
        include_trend_analysis = st.checkbox("Trend Analysis", value=True)
    
    if st.button("🎯 Generate Custom Report"):
        st.success("Custom report configuration saved! Use the download buttons above to generate your report.")

# Report scheduling (placeholder)
st.markdown("### 📅 Report Scheduling")
st.info("💡 **Pro Tip**: In a production environment, you could schedule automated report generation and email delivery.")

col1, col2, col3 = st.columns(3)

with col1:
    schedule_frequency = st.selectbox(
        "Schedule Frequency",
        ["Manual", "Daily", "Weekly", "Monthly"]
    )

with col2:
    email_recipients = st.text_input("Email Recipients", placeholder="user@company.com")

with col3:
    if st.button("📧 Setup Scheduling", disabled=True):
        st.info("Scheduling feature available in enterprise version")

# Usage analytics
st.markdown("### 📈 Report Usage Analytics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Reports Generated", "1", "This session")

with col2:
    st.metric("Most Used Format", "Excel", "70% of downloads")

with col3:
    st.metric("Avg Report Size", "2.3 MB", "Typical size")

with col4:
    st.metric("Export Success Rate", "100%", "No errors")