import streamlit as st
import pandas as pd
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

def login_page():
    """Login and signup page"""
    st.title("🛒 Retail Sales Forecasting Platform")
    st.markdown("### Industrial Engineering Solution for Supply Chain Optimization")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary", use_container_width=True):
                # Simplified authentication - in production, use proper authentication
                if username and password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Please enter both username and password")
        
        with tab2:
            st.subheader("Create New Account")
            new_username = st.text_input("Username", key="signup_username")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            
            if st.button("Sign Up", type="primary", use_container_width=True):
                if new_username and new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        st.session_state.authenticated = True
                        st.session_state.username = new_username
                        st.success("Account created successfully!")
                        st.rerun()
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

def main_app():
    """Main application after authentication"""
    # Sidebar
    with st.sidebar:
        st.title("🛒 Retail Forecasting")
        st.markdown(f"Welcome, **{st.session_state.username}**!")
        
        # Navigation
        st.markdown("### Navigation")
        
        # Theme toggle
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark"):
                st.session_state.theme = "dark"
        with theme_col2:
            if st.button("☀️ Light"):
                st.session_state.theme = "light"
        
        st.markdown("---")
        
        # Logout
        if st.button("🚪 Logout", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
    
    # Main content
    st.title("🏠 Home Dashboard")
    st.markdown("### Retail Sales Forecasting & Supply Chain Optimization")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Dataset", "retail_sales.csv", "2 days ago")
    
    with col2:
        st.metric("Last Forecast", "LightGBM Model", "1 day ago")
    
    with col3:
        st.metric("Forecast Accuracy", "94.2%", "2.1%")
    
    with col4:
        st.metric("Active Models", "3", "1")
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Getting Started
        1. **📊 Upload Data** - Upload your retail sales dataset
        2. **🔮 Select Models** - Choose forecasting algorithms
        3. **📈 View Dashboard** - Analyze forecast results
        4. **📋 Generate Reports** - Download insights and forecasts
        """)
    
    with col2:
        st.markdown("""
        #### Supported Models
        - **Prophet** - Seasonality & holidays detection
        - **Random Forest** - Tree-based ensemble method
        - **LightGBM** - Gradient boosting (recommended)
        - **ARIMA/SARIMA** - Classical time series
        """)
    
    # Sample data info
    st.markdown("### 📋 Sample Dataset Information")
    
    # Create sample data if it doesn't exist
    if not os.path.exists("sample_retail_data.csv"):
        create_sample_data()
    
    if os.path.exists("sample_retail_data.csv"):
        sample_df = pd.read_csv("sample_retail_data.csv")
        st.dataframe(sample_df.head(10), use_container_width=True)
        st.caption(f"Sample dataset with {len(sample_df)} records showing retail sales data")

def create_sample_data():
    """Create sample retail sales data"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
    categories = ['Electronics', 'Clothing', 'Groceries', 'Home']
    
    data = []
    for date in date_range:
        for store in stores:
            for category in categories:
                # Add seasonality and trend
                base_sales = 1000 + np.sin(date.timetuple().tm_yday * 2 * np.pi / 365) * 200
                trend = (date - start_date).days * 0.1
                noise = np.random.normal(0, 100)
                
                sales = max(0, base_sales + trend + noise)
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Store': store,
                    'Category': category,
                    'Sales': round(sales, 2),
                    'Units_Sold': int(sales / 50),
                    'Promotion': np.random.choice([0, 1], p=[0.8, 0.2])
                })
    
    df = pd.DataFrame(data)
    df.to_csv("sample_retail_data.csv", index=False)

# Main application logic
if not st.session_state.authenticated:
    login_page()
else:
    main_app()