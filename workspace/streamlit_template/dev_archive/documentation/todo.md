# Retail Sales Forecasting Web Application - MVP Implementation

## Core Features to Implement
1. **Multi-page Streamlit app** with sidebar navigation
2. **User authentication** (simplified session-based)
3. **Data upload and validation** (CSV/Excel support)
4. **Multiple forecasting models** (Prophet, Random Forest, LightGBM)
5. **Interactive dashboard** with Plotly charts
6. **Report generation** (Excel export)
7. **Light/Dark mode toggle**

## Files to Create
1. `app.py` - Main Streamlit application with multi-page setup
2. `pages/1_📊_Upload_Data.py` - Data upload and validation page
3. `pages/2_🔮_Model_Selection.py` - Model selection and forecasting page
4. `pages/3_📈_Dashboard.py` - Interactive forecast dashboard
5. `pages/4_📋_Reports.py` - Report generation and downloads
6. `pages/5_⚙️_Settings.py` - Settings and profile page
7. `utils/auth.py` - Authentication utilities
8. `utils/data_processor.py` - Data preprocessing functions
9. `utils/models.py` - Forecasting model implementations
10. `utils/report_generator.py` - Report generation utilities

## MVP Simplifications
- Use session state for user authentication (no database)
- Implement 3 core models: Prophet, Random Forest, LightGBM
- Focus on store-level forecasting
- Excel export only (no PDF for MVP)
- Sample retail dataset included

## Key Libraries Needed
- streamlit, plotly, pandas, numpy
- scikit-learn, lightgbm, prophet
- openpyxl for Excel export