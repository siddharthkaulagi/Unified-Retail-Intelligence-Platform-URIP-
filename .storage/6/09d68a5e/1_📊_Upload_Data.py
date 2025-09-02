import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(page_title="Upload Data", page_icon="📊", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

st.title("📊 Upload & Validate Data")
st.markdown("Upload your retail sales dataset for forecasting analysis")

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your retail sales data with columns: Date, Store, Category, Sales"
)

# Sample data option
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Use Sample Data", type="secondary"):
        if os.path.exists("sample_retail_data.csv"):
            st.session_state.uploaded_data = pd.read_csv("sample_retail_data.csv")
            st.success("Sample data loaded successfully!")
        else:
            st.error("Sample data not found. Please upload your own data.")

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.session_state.uploaded_data = df
        st.success(f"File uploaded successfully! Shape: {df.shape}")
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

# Display data if available
if 'uploaded_data' in st.session_state:
    df = st.session_state.uploaded_data
    
    # Data validation
    st.markdown("### 🔍 Data Validation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values, 
                 delta="⚠️" if missing_values > 0 else "✅")
    
    # Column validation
    required_columns = ['Date', 'Sales']
    optional_columns = ['Store', 'Category', 'Units_Sold', 'Promotion']
    
    st.markdown("#### Column Validation")
    validation_results = []
    
    for col in required_columns:
        if col in df.columns:
            validation_results.append({"Column": col, "Status": "✅ Found", "Type": "Required"})
        else:
            validation_results.append({"Column": col, "Status": "❌ Missing", "Type": "Required"})
    
    for col in optional_columns:
        if col in df.columns:
            validation_results.append({"Column": col, "Status": "✅ Found", "Type": "Optional"})
    
    validation_df = pd.DataFrame(validation_results)
    st.dataframe(validation_df, use_container_width=True)
    
    # Data preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Data summary
    st.markdown("### 📈 Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Numerical Summary")
        if 'Sales' in df.columns:
            st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Visualizations
    if 'Date' in df.columns and 'Sales' in df.columns:
        st.markdown("### 📊 Data Visualization")
        
        # Convert date column
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Time series plot
            if 'Store' in df.columns:
                fig = px.line(df.groupby(['Date', 'Store'])['Sales'].sum().reset_index(),
                             x='Date', y='Sales', color='Store',
                             title="Sales Trend by Store")
            else:
                daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
                fig = px.line(daily_sales, x='Date', y='Sales',
                             title="Daily Sales Trend")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sales distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df, x='Sales', nbins=50,
                                       title="Sales Distribution")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                if 'Category' in df.columns:
                    category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
                    fig_bar = px.bar(x=category_sales.index, y=category_sales.values,
                                    title="Sales by Category")
                    fig_bar.update_xaxis(title="Category")
                    fig_bar.update_yaxis(title="Total Sales")
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        except Exception as e:
            st.warning(f"Could not parse dates: {str(e)}")
    
    # Data preprocessing options
    st.markdown("### 🔧 Data Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        handle_missing = st.selectbox(
            "Handle Missing Values",
            ["Keep as is", "Forward fill", "Backward fill", "Drop rows"]
        )
    
    with col2:
        date_format = st.selectbox(
            "Date Format",
            ["Auto-detect", "YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"]
        )
    
    if st.button("Apply Preprocessing", type="primary"):
        processed_df = df.copy()
        
        # Handle missing values
        if handle_missing == "Forward fill":
            processed_df = processed_df.fillna(method='ffill')
        elif handle_missing == "Backward fill":
            processed_df = processed_df.fillna(method='bfill')
        elif handle_missing == "Drop rows":
            processed_df = processed_df.dropna()
        
        # Store processed data
        st.session_state.processed_data = processed_df
        st.success("Data preprocessing completed!")
        
        # Show changes
        if len(processed_df) != len(df):
            st.info(f"Data shape changed from {df.shape} to {processed_df.shape}")

else:
    st.info("👆 Please upload a dataset or use the sample data to get started.")
    
    # Show expected format
    st.markdown("### 📋 Expected Data Format")
    st.markdown("""
    Your dataset should contain the following columns:
    
    **Required:**
    - `Date`: Date of sales (YYYY-MM-DD format preferred)
    - `Sales`: Sales amount (numerical)
    
    **Optional:**
    - `Store`: Store identifier
    - `Category`: Product category
    - `Units_Sold`: Number of units sold
    - `Promotion`: Promotion indicator (0/1)
    """)
    
    # Sample format
    sample_format = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Store': ['Store_A', 'Store_A', 'Store_B'],
        'Category': ['Electronics', 'Clothing', 'Electronics'],
        'Sales': [1250.50, 890.25, 1100.75],
        'Units_Sold': [25, 18, 22],
        'Promotion': [0, 1, 0]
    })
    
    st.dataframe(sample_format, use_container_width=True)