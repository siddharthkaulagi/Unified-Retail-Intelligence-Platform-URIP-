# workspace/streamlit_template/pages/05_Settings.py
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils.ui_components import render_sidebar
from utils.feature_flags import feature_flags

# IMPORTANT: set_page_config must be called before any other Streamlit commands.
st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

def load_css():
    """
    Load CSS relative to this file's location (robust on Streamlit Cloud).
    pages/05_Settings.py -> parent = pages -> parent.parent = streamlit_template
    assets/custom.css should be at: streamlit_template/assets/custom.css
    """
    css_path = Path(__file__).resolve().parent.parent / "assets" / "custom.css"

    if not css_path.exists():
        # Do not raise — show a helpful warning in-app and continue.
        st.info(f"Custom CSS not found at `{css_path}` — continuing without custom styles.")
        return

    try:
        with css_path.open("r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning("Failed to load custom CSS — continuing without it.")
        # Optionally log exception to console (useful for Streamlit logs)
        st.write(f"CSS load error: {e}")

# load css after set_page_config
load_css()

# --- rest of your page ---
# Check if this feature is enabled
if not feature_flags.is_enabled('settings'):
    st.warning("⚠️ This feature is currently disabled.")
    st.info("Contact your administrator if you need access to this page.")
    st.stop()

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

st.title("⚙️ Settings & Configuration")
st.caption("Manage your account and system preferences")

st.markdown("---")

# User Profile Section - Clean Card UI
st.markdown("### 👤 User Profile")

# Get current user info
current_username = st.session_state.get('username', 'User')

col1, col2 = st.columns([2, 1])

with col1:
    username = st.text_input("Username", value=current_username, help="Your display name")
    email = st.text_input("Email", value=f"{current_username.lower()}@company.com", disabled=True, help="Read-only")

with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    if st.button("💾 Update Profile", type="primary", use_container_width=True):
        st.session_state.username = username
        st.success("✅ Profile updated successfully!")
        st.rerun()

st.markdown("---")

# System Information - Clean 3-Column Layout
st.markdown("### ℹ️ System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Application Version", "1.0.0", delta="Latest")
    st.caption("📅 Build: 2025.01.15")

with col2:
    st.metric("Active User", current_username)
    st.caption(f"🕐 Session: {datetime.now().strftime('%H:%M:%S')}")

with col3:
    st.metric("System Status", "Online", delta="All systems operational")
    st.caption("🟢 All services running")

st.markdown("---")

# System Status - Clean Status Grid
st.markdown("### 📊 Service Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.success("**Application**")
    st.caption("🟢 Online")

with status_col2:
    st.success("**Database**")
    st.caption("🟢 Connected")

with status_col3:
    st.success("**ML Models**")
    st.caption("🟢 Available")

with status_col4:
    st.success("**AI Services**")
    st.caption("🟢 Active")

st.markdown("---")

# Action Buttons - Aligned and Prominent
st.markdown("### 🎯 Quick Actions")

action_col1, action_col2, action_col3 = st.columns(3)

with action_col1:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        settings = {
            'username': st.session_state.get('username', 'User'),
            'theme': st.session_state.get('app_theme', 'light'),
            'language': st.session_state.get('app_language', 'English'),
        }
        st.session_state.user_settings = settings
        st.success("✅ All settings saved!")

with action_col2:
    if st.button("🔄 Reset Settings", use_container_width=True):
        st.session_state.app_theme = 'light'
        st.session_state.app_language = 'English'
        if 'user_settings' in st.session_state:
            del st.session_state.user_settings
        st.success("✅ Reset to defaults!")
        st.rerun()

with action_col3:
    if st.button("🚪 Logout", type="secondary", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.success("👋 Logged out successfully!")
        st.rerun()

st.markdown("---")

# Recent Activity - Clean Table
st.markdown("### 📋 Recent Activity")

activity_data = [
    {"Time": datetime.now().strftime('%H:%M:%S'), "Action": "Viewed Settings page", "Status": "✅ Success"},
    {"Time": datetime.now().strftime('%H:%M:%S'), "Action": f"Logged in as {current_username}", "Status": "✅ Success"},
]

activity_df = pd.DataFrame(activity_data)
st.dataframe(activity_df, use_container_width=True, hide_index=True)

# Footer Note
st.markdown("---")
st.caption("💡 **Note:** Display preferences (Theme & Language) are accessible from the sidebar.")
