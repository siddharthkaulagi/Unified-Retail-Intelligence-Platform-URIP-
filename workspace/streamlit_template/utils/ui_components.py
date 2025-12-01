import streamlit as st
from utils.database import session_manager

def render_sidebar():
    """
    Renders the standard application sidebar with navigation and user info.
    This centralizes the sidebar logic to ensure consistency across all pages.
    """
    with st.sidebar:
        st.title("ML-Driven Retail Supply Chain Optimization Platform")
        
        # User Welcome
        username = st.session_state.get('username', 'Guest')
        st.markdown(f"Welcome, **{username}**!")
        
        # Navigation Section
        st.markdown("### Navigation")

        # Navigation Section
        st.markdown("### Navigation")

        # AI Chatbot Button
        st.markdown("### 🤖 AI Assistant")
        if st.button("💬 Open AI Chat", width='stretch', type="primary"):
            st.switch_page("pages/09_AI_Chatbot.py")

        # GIS Analytics Button
        st.markdown("### 🏪 Store Location GIS")
        if st.button("🗺️ GIS Analytics", width='stretch', type="secondary"):
            st.switch_page("pages/10_Store_Location_GIS.py")
        
        # Theme Toggle
        st.markdown("### 🎨 Appearance")
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark"):
                st.session_state.theme = "dark"
                st.markdown("""
                <style>
                .stApp { background-color: #1a1a1a; color: white; }
                </style>
                """, unsafe_allow_html=True)
        with theme_col2:
            if st.button("☀️ Light"):
                st.session_state.theme = "light"
                st.markdown("""
                <style>
                .stApp { background-color: white; color: black; }
                </style>
                """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Logout Button
        if st.button("🚪 Logout", type="secondary"):
            if 'session_token' in st.session_state:
                session_manager.destroy_session(st.session_state.session_token)

            # Clear session state
            st.session_state.authenticated = False
            for key in ['user_id', 'username', 'user_email', 'user_role', 'session_token']:
                if key in st.session_state:
                    del st.session_state[key]

            st.success("Logged out successfully!")
            st.rerun()
