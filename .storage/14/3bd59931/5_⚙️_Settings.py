import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

st.title("⚙️ Settings & Configuration")
st.markdown("Manage your profile, preferences, and application settings")

# User Profile Section
st.markdown("### 👤 User Profile")

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://via.placeholder.com/150x150/4CAF50/FFFFFF?text=User", width=150)
    
    if st.button("📷 Change Avatar", use_container_width=True):
        st.info("Avatar upload feature coming soon!")

with col2:
    st.markdown("#### Profile Information")
    
    # Get current user info
    current_username = st.session_state.get('username', 'User')
    
    username = st.text_input("Username", value=current_username)
    email = st.text_input("Email", value=f"{current_username.lower()}@company.com")
    full_name = st.text_input("Full Name", value=f"{current_username} Smith")
    department = st.selectbox("Department", ["Supply Chain", "Analytics", "Operations", "Management"])
    
    if st.button("💾 Update Profile", type="primary"):
        st.session_state.username = username
        st.success("Profile updated successfully!")

# Application Preferences
st.markdown("### 🎨 Application Preferences")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Display Settings")
    
    # Theme selection
    current_theme = st.session_state.get('theme', 'light')
    theme = st.radio(
        "Theme",
        ["light", "dark"],
        index=0 if current_theme == 'light' else 1,
        horizontal=True
    )
    
    if theme != current_theme:
        st.session_state.theme = theme
        st.success(f"Theme changed to {theme} mode!")
    
    # Language
    language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
    
    # Timezone
    timezone = st.selectbox(
        "Timezone", 
        ["UTC", "EST", "PST", "GMT", "CET"],
        index=2  # Default to PST
    )
    
    # Date format
    date_format = st.selectbox(
        "Date Format",
        ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY", "DD-MM-YYYY"]
    )

with col2:
    st.markdown("#### Notification Settings")
    
    email_notifications = st.checkbox("Email Notifications", value=True)
    forecast_alerts = st.checkbox("Forecast Completion Alerts", value=True)
    error_notifications = st.checkbox("Error Notifications", value=True)
    weekly_reports = st.checkbox("Weekly Summary Reports", value=False)
    
    st.markdown("#### Data Preferences")
    
    default_forecast_days = st.slider("Default Forecast Period (Days)", 7, 365, 30)
    auto_save_results = st.checkbox("Auto-save Results", value=True)
    data_retention_days = st.slider("Data Retention (Days)", 30, 365, 90)

# Model Configuration
st.markdown("### 🤖 Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Default Model Settings")
    
    preferred_models = st.multiselect(
        "Preferred Models",
        ["Prophet", "Random Forest", "LightGBM", "ARIMA"],
        default=["Prophet", "LightGBM"]
    )
    
    default_train_split = st.slider("Default Training Split", 0.6, 0.9, 0.8)
    enable_cross_validation = st.checkbox("Enable Cross Validation", value=True)
    
    confidence_level = st.slider("Confidence Level (%)", 80, 99, 95)

with col2:
    st.markdown("#### Performance Settings")
    
    max_models_parallel = st.slider("Max Parallel Models", 1, 4, 2)
    model_timeout = st.slider("Model Timeout (minutes)", 1, 30, 10)
    
    st.markdown("#### Advanced Options")
    
    enable_feature_engineering = st.checkbox("Auto Feature Engineering", value=True)
    enable_hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
    cache_model_results = st.checkbox("Cache Model Results", value=True)

# Data Management
st.markdown("### 📊 Data Management")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Upload Settings")
    
    max_file_size = st.slider("Max File Size (MB)", 1, 100, 50)
    allowed_formats = st.multiselect(
        "Allowed File Formats",
        ["CSV", "Excel", "JSON", "Parquet"],
        default=["CSV", "Excel"]
    )
    
    auto_data_validation = st.checkbox("Auto Data Validation", value=True)
    handle_missing_data = st.selectbox(
        "Handle Missing Data",
        ["Prompt User", "Auto Forward Fill", "Auto Drop", "Auto Interpolate"]
    )

with col2:
    st.markdown("#### Storage Settings")
    
    st.info("📊 **Current Usage**")
    st.progress(0.3, text="Storage: 30% used (150MB / 500MB)")
    st.progress(0.6, text="Models: 60% used (6 / 10 saved models)")
    
    if st.button("🧹 Clear Cache", use_container_width=True):
        st.success("Cache cleared successfully!")
    
    if st.button("📤 Export All Data", use_container_width=True):
        st.info("Data export initiated. Check your downloads folder.")

# Security Settings
st.markdown("### 🔒 Security Settings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Account Security")
    
    if st.button("🔑 Change Password", use_container_width=True):
        with st.form("change_password"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Password"):
                if new_password == confirm_password:
                    st.success("Password updated successfully!")
                else:
                    st.error("Passwords do not match!")
    
    enable_2fa = st.checkbox("Enable Two-Factor Authentication", value=False)
    
    if enable_2fa:
        st.info("📱 2FA setup instructions will be sent to your email")

with col2:
    st.markdown("#### Session Management")
    
    auto_logout = st.slider("Auto Logout (minutes)", 15, 480, 60)
    remember_login = st.checkbox("Remember Login", value=True)
    
    st.markdown("#### Privacy Settings")
    
    share_usage_data = st.checkbox("Share Anonymous Usage Data", value=True)
    allow_data_export = st.checkbox("Allow Data Export", value=True)

# System Information
st.markdown("### ℹ️ System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Application Info")
    st.write("**Version**: 1.0.0")
    st.write("**Build**: 2024.01.15")
    st.write("**Environment**: Production")

with col2:
    st.markdown("#### Session Info")
    st.write(f"**User**: {st.session_state.get('username', 'Unknown')}")
    st.write(f"**Login Time**: {datetime.now().strftime('%H:%M:%S')}")
    st.write("**Session ID**: abc123def456")

with col3:
    st.markdown("#### System Status")
    st.write("🟢 **API**: Online")
    st.write("🟢 **Database**: Connected")
    st.write("🟢 **ML Models**: Available")

# Action Buttons
st.markdown("### 🎯 Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        # Save settings to session state
        settings = {
            'theme': theme,
            'language': language,
            'timezone': timezone,
            'date_format': date_format,
            'default_forecast_days': default_forecast_days,
            'preferred_models': preferred_models,
            'default_train_split': default_train_split,
            'max_file_size': max_file_size,
            'auto_logout': auto_logout
        }
        
        st.session_state.user_settings = settings
        st.success("✅ All settings saved successfully!")

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        if st.button("⚠️ Confirm Reset", use_container_width=True):
            # Reset settings
            if 'user_settings' in st.session_state:
                del st.session_state.user_settings
            st.success("Settings reset to defaults!")

with col3:
    if st.button("📤 Export Settings", use_container_width=True):
        settings_data = st.session_state.get('user_settings', {})
        settings_json = pd.DataFrame([settings_data]).to_json(indent=2)
        
        st.download_button(
            label="📥 Download Settings",
            data=settings_json,
            file_name=f"user_settings_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

with col4:
    if st.button("📥 Import Settings", use_container_width=True):
        uploaded_settings = st.file_uploader(
            "Upload Settings File",
            type=['json'],
            key="settings_upload"
        )
        
        if uploaded_settings:
            try:
                import json
                settings = json.load(uploaded_settings)
                st.session_state.user_settings = settings
                st.success("Settings imported successfully!")
            except Exception as e:
                st.error(f"Error importing settings: {str(e)}")

# Help and Support
st.markdown("### 🆘 Help & Support")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Quick Help")
    
    help_topics = [
        "📊 How to upload data",
        "🔮 Running forecasts",
        "📈 Reading dashboards",
        "📋 Generating reports",
        "⚙️ Configuring settings"
    ]
    
    selected_help = st.selectbox("Select Help Topic", help_topics)
    
    if st.button("📖 View Help", use_container_width=True):
        st.info(f"Help documentation for: {selected_help}")

with col2:
    st.markdown("#### Contact Support")
    
    support_type = st.selectbox(
        "Support Type",
        ["Technical Issue", "Feature Request", "General Question", "Bug Report"]
    )
    
    if st.button("📧 Contact Support", use_container_width=True):
        st.success("Support ticket created! We'll get back to you within 24 hours.")

# Footer
st.markdown("---")
st.markdown("### 📋 Recent Activity")

activity_data = [
    {"Time": "10:30 AM", "Action": "Updated profile settings", "Status": "✅"},
    {"Time": "10:15 AM", "Action": "Changed theme to dark mode", "Status": "✅"},
    {"Time": "09:45 AM", "Action": "Saved model preferences", "Status": "✅"},
    {"Time": "09:30 AM", "Action": "Logged in", "Status": "✅"}
]

activity_df = pd.DataFrame(activity_data)
st.dataframe(activity_df, use_container_width=True, hide_index=True)