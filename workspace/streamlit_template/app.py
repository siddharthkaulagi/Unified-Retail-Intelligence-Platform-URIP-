import streamlit as st
import pandas as pd
from datetime import datetime
import os
from utils.database import session_manager
from utils.ui_components import render_sidebar


# --- PATHS ---
# Get the absolute path of the directory containing the script
# This makes the app runnable from any directory
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_FILE = os.path.join(ASSETS_DIR, "custom.css")
LOGO_FILE = os.path.join(ASSETS_DIR, "urip_logo.png")
SAMPLE_DATA_FILE = os.path.join(BASE_DIR, "sample_retail_data.csv")


# Page configuration
st.set_page_config(
    page_title="ML-Driven Retail Supply Chain Optimization Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed by default for landing page
)

def load_css():
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'theme' not in st.session_state:
    st.session_state.theme = "light"
if 'auth_page' not in st.session_state:
    st.session_state.auth_page = "landing" # Default to landing page

def landing_page():
    """
    Modern Landing Page for the application.
    Showcases features and provides entry points for Login/Signup.
    """
    # Hero Section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div style='padding-top: 50px;'>
            <h1 style='font-size: 3.5rem; font-weight: 800; background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                Unified Retail Intelligence Platform
            </h1>
            <h3 style='font-size: 1.5rem; color: #555; margin-bottom: 30px;'>
                Transform your retail operations with ML-driven forecasting, planning, and geospatial analytics.
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Call to Action Buttons
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🚀 Get Started", type="primary", use_container_width=True):
                st.session_state.auth_page = "signup"
                st.rerun()
        with c2:
            if st.button("🔑 Sign In", type="secondary", use_container_width=True):
                st.session_state.auth_page = "login"
                st.rerun()
                
    with col2:
        # Display a hero image or a nice graphic
        # Using the logo for now, but could be a dashboard screenshot
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=400)
        else:
            st.markdown("🛒", unsafe_allow_html=True) # Fallback icon

    st.markdown("---")

    # Features Section
    st.markdown("<h2 style='text-align: center; margin-bottom: 50px;'>Why Choose URIP?</h2>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    
    with f1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 250px;'>
            <h1 style='font-size: 3rem;'>📈</h1>
            <h3 style='color: #1e293b;'>Demand Forecasting</h3>
            <p style='color: #475569;'>Predict future sales with high accuracy using advanced ML models like Prophet and XGBoost.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 250px;'>
            <h1 style='font-size: 3rem;'>🗺️</h1>
            <h3 style='color: #1e293b;'>Store Location GIS</h3>
            <p style='color: #475569;'>Identify optimal store locations using geospatial data, competitor analysis, and demographic scoring.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 250px;'>
            <h1 style='font-size: 3rem;'>🤖</h1>
            <h3 style='color: #1e293b;'>AI Assistant</h3>
            <p style='color: #475569;'>Get real-time insights and answers to your business questions with our Gemini-powered chatbot.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    f4, f5, f6 = st.columns(3)
    
    with f4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 250px;'>
            <h1 style='font-size: 3rem;'>📦</h1>
            <h3 style='color: #1e293b;'>Inventory Analytics</h3>
            <p style='color: #475569;'>Optimize stock levels with ABC, XYZ, and FSN analysis to reduce carrying costs and stockouts.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f5:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 250px;'>
            <h1 style='font-size: 3rem;'>🏭</h1>
            <h3 style='color: #1e293b;'>Facility Layout</h3>
            <p style='color: #475569;'>Design efficient warehouse and store layouts using the Activity Relationship Chart (ARC) method.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with f6:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; height: 250px;'>
            <h1 style='font-size: 3rem;'>👥</h1>
            <h3 style='color: #1e293b;'>CRM Analytics</h3>
            <p style='color: #475569;'>Understand your customers better with RFM segmentation, Churn Prediction, and CLV analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>© 2024 Unified Retail Intelligence Platform. All rights reserved.</p>", unsafe_allow_html=True)


def login_page(default_tab="Login"):
    """Login and signup page"""
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.auth_page = "landing"
        st.rerun()

    # Logo and title on the same line
    logo_col, title_col = st.columns([1, 8])
    with logo_col:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=150)
    with title_col:
        st.markdown("<h1 style='margin-top: 30px;'>URIP-Unified Retail Intelligence Platform</h1>", unsafe_allow_html=True)
        st.markdown("ML-Driven Retail Supply Chain Optimization Platform for Forecasting, Planning & Decision Support", unsafe_allow_html=True)

    
    col2, col3 = st.columns([1, 2])
    
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary", use_container_width=True):
                if username and password:
                    from utils.database import user_manager, session_manager
                    
                    # Attempt authentication
                    success, message, user_data = user_manager.authenticate_user(
                        username, password,
                        ip_address=None,
                        user_agent=None
                    )
                    
                    if success and user_data:
                        # Create session
                        session_token = session_manager.create_session(
                            user_data['id'],
                            ip_address=None,
                            user_agent=None
                        )
                        
                        # Set session state
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_data['id']
                        st.session_state.username = user_data['username']
                        st.session_state.user_email = user_data['email']
                        st.session_state.user_role = user_data['role']
                        st.session_state.session_token = session_token
                        
                        st.success(f"Welcome back, {user_data['username']}!")
                        st.rerun()
                    else:
                        st.error(message)
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
                        from utils.database import user_manager, session_manager
                        
                        # Create user
                        success, message = user_manager.create_user(
                            new_username, new_email, new_password, "", ""
                        )
                        
                        if success:
                            st.success(message)
                            st.info("You can now login with your credentials!")
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")
            
            # If user came here via "Sign Up" button, show a hint
            if default_tab == "Sign Up":
                st.caption("👈 Click the 'Sign Up' tab above if not already selected.")



def main_app():
    """Main application after authentication"""
    # Sidebar
    render_sidebar()

    
    # Main content
    # Logo and title on the same line
    logo_col, title_col = st.columns([1, 8])
    with logo_col:
        if os.path.exists(LOGO_FILE):
            st.image(LOGO_FILE, width=150)
    with title_col:
        st.markdown("<h1 style='margin-top: 30px;'>Unified Retail Intelligence Platform (URIP)</h1>", unsafe_allow_html=True)
    # --- 1. Welcome Header ---
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: #1E3D59;'>Welcome to URIP – Unified Retail Intelligence Platform</h2>
        <p style='font-size: 18px; color: #555;'>A complete decision-support dashboard for retail forecasting, inventory analytics & store intelligence.</p>
        <p style='font-size: 14px; color: #888;'>Choose a module to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # --- 2. Main Feature Cards ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**📈 Sales & Demand Forecasting**\n\n"
                "• Predict daily/weekly/monthly sales\n"
                "• Compare ML models (ARIMA, Prophet, XGBoost)\n"
                "• View accuracy metrics (MAPE, RMSE)")
        
        st.info("**🏬 Store Layout Insights**\n\n"
                "• SKU placement suggestions\n"
                "• Basic movement-based optimization\n"
                "• Department relationship scoring")

    with col2:
        st.success("**📦 Inventory Analytics**\n\n"
                   "• ABC / XYZ / FSN Classification\n"
                   "• Identify high-value & volatile SKUs\n"
                   "• Optimize stock levels")
        
        st.success("**👥 CRM & Customer Insights**\n\n"
                   "• RFM Customer Segmentation\n"
                   "• Identify Champions & At-Risk customers\n"
                   "• Targeted marketing strategies")

    with col3:
        st.warning("**🗺️ GIS Store Mapping**\n\n"
                   "• Visualize store performance\n"
                   "• Ward-wise geospatial insights\n"
                   "• Location intelligence for expansion")
        
        st.warning("**📄 Reports & Downloads**\n\n"
                   "• Generate full project reports\n"
                   "• Download forecasts & analytics\n"
                   "• Auto-generated executive summaries")

    st.markdown("---")

    # --- 3. System Architecture Diagram ---
    st.subheader("🏗️ System Architecture")
    st.markdown("""
    <div style='border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 10px; background-color: #f9f9f9;'>
        <h4>&lt; System Architecture Diagram &gt;</h4>
        <p>Data Ingestion ➔ Preprocessing ➔ ML Models ➔ Inventory Analytics ➔ GIS Module ➔ Dashboard ➔ Reports</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # # --- 4. Quick Statistics ---
    # st.subheader("📊 Live System Stats")
    # stat1, stat2, stat3, stat4 = st.columns(4)
    # stat1.metric("SKUs Analyzed", "1,245", "+12%")
    # stat2.metric("Stores Mapped", "50", "Bangalore")
    # stat3.metric("Forecast Accuracy", "92.4%", "Ensemble")
    # stat4.metric("Records Processed", "1.2M+", "Real-time")
    
    # st.markdown("---")

    # --- 5. Tech Stack Highlights ---
    st.markdown("**🚀 Powered By:**")
    st.markdown("Python | Streamlit | Scikit-Learn | Prophet | XGBoost | LightGBM | GeoPandas | Folium | Pandas | SQLite")
    
    st.markdown("---")

    # --- 6. Get Started Buttons ---
    st.markdown("### 🚀 Get Started")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("⬆️ Upload New Data", use_container_width=True):
            st.switch_page("pages/01_Upload_Data.py")
    with btn_col2:
        if st.button("📈 Open Forecasting Dashboard", use_container_width=True):
            st.switch_page("pages/06_Demand_Forecasting.py")



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
    df.to_csv(SAMPLE_DATA_FILE, index=False)

# Main application logic - Session-based authentication
def check_session():
    """Check if user has a valid session"""
    if 'session_token' in st.session_state:
        user_data = session_manager.validate_session(st.session_state.session_token)
        if user_data:
            # Update session state with user data
            st.session_state.authenticated = True
            st.session_state.user_id = user_data['id']
            st.session_state.username = user_data['username']
            st.session_state.user_email = user_data['email']
            st.session_state.user_role = user_data['role']
            return True

    # Clear invalid session
    st.session_state.authenticated = False
    for key in ['user_id', 'username', 'user_email', 'user_role', 'session_token']:
        if key in st.session_state:
            del st.session_state[key]
    return False

# Check session on app start
if check_session():
    main_app()
else:
    # Routing logic for unauthenticated users
    if st.session_state.auth_page == "landing":
        landing_page()
    elif st.session_state.auth_page == "login":
        login_page(default_tab="Login")
    elif st.session_state.auth_page == "signup":
        login_page(default_tab="Sign Up")
    else:
        landing_page()
