#!/usr/bin/env python
"""Test GIS page syntax"""

try:
    print("Testing GIS page syntax...")
    exec(open('pages/10_🏪_Store_Location_GIS.py').read())
    print("✅ GIS page syntax check passed!")
except SyntaxError as e:
    print(f"❌ Syntax Error: {e}")
    print(f"Line {e.lineno}: {e.text}")
except Exception as e:
    print(f"❌ Other Error: {e}")
    # This is expected since streamlit functions won't work outside the app
