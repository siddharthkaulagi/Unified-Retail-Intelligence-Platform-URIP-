from pathlib import Path
import streamlit as st

def load_css_for_page(page_file: str):
    css_path = Path(page_file).resolve().parent.parent / "assets" / "custom.css"

    if not css_path.exists():
        st.info(f"⚠️ CSS not found at: {css_path}")
        return

    with css_path.open("r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
