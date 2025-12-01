import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from utils.ui_components import render_sidebar


# --- PATHS ---
# Get the absolute path of the directory containing the script's parent
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_FILE = os.path.join(ASSETS_DIR, "custom.css")
SAMPLE_DATA_FILE = os.path.join(BASE_DIR, "sample_retail_data.csv")


def load_css():
    if os.path.exists(CSS_FILE):
        with open(CSS_FILE) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="Upload Data", page_icon="⬆️", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

st.title("⬆️ Upload & Validate Data")
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
        if os.path.exists(SAMPLE_DATA_FILE):
            st.session_state.uploaded_data = pd.read_csv(SAMPLE_DATA_FILE)
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
    st.dataframe(validation_df)
    
    # Data preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(20))

    # Data summary
    st.markdown("### 📑 Data Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Numerical Summary")
        if 'Sales' in df.columns:
            st.dataframe(df.describe())

    with col2:
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count()
        })
        st.dataframe(dtype_df)
    
    # Visualizations
    if 'Date' in df.columns and 'Sales' in df.columns:
        st.markdown("### 📶 Data Visualization")
        
        # Convert date column
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # Sales trend with moving averages (Last 9 Months)
            if 'Date' in df.columns:
                daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
                
                # Filter for last 9 months (approx 270 days) for cleaner visualization
                last_date = daily_sales['Date'].max()
                start_date = last_date - pd.Timedelta(days=390)
                plot_data = daily_sales[daily_sales['Date'] >= start_date].copy()
                
                # Calculate MAs on the full dataset first, then filter
                daily_sales['7_day_MA'] = daily_sales['Sales'].rolling(window=7).mean()
                daily_sales['30_day_MA'] = daily_sales['Sales'].rolling(window=30).mean()
                
                # Update plot data with calculated MAs
                plot_data = daily_sales[daily_sales['Date'] >= start_date]

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Sales'],
                                             mode='lines', name='Daily Sales', opacity=0.3, line=dict(color='gray')))
                fig_trend.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['7_day_MA'],
                                             mode='lines', name='7-Day Moving Average', line=dict(width=2, color='#1f77b4')))
                fig_trend.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['30_day_MA'],
                                             mode='lines', name='30-Day Moving Average', line=dict(width=2, color='#ff7f0e')))

                fig_trend.update_layout(title="Sales Trend (Last 9 Months Preview)",
                                      xaxis_title="Date", yaxis_title="Sales ($)",
                                      template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
                st.caption("ℹ️ Displaying last 9 months only for clarity. Full dataset will be used for training.")

            # Top performing categories analysis
            if 'Category' in df.columns and 'Store' in df.columns:
                category_store = df.groupby(['Category', 'Store'])['Sales'].sum().reset_index()
                # Get top store for each category
                top_category_store = category_store.loc[category_store.groupby('Category')['Sales'].idxmax()]

                fig_top = px.scatter(top_category_store, x='Category', y='Sales', size='Sales',
                                   color='Store', title="Best Performing Store per Category",
                                   template="plotly_white")
                st.plotly_chart(fig_top, use_container_width=True)

            # Sales distribution
            col1, col2 = st.columns(2)

            with col1:
                fig_hist = px.histogram(
                    df, x='Sales', 
                    nbins=30,  # Reduced from 50 for smoother distribution
                    title="Sales Distribution",
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist.update_layout(
                    bargap=0.1,  # Small gap between bars for cleaner look
                    template='plotly_white'
                )
                fig_hist.update_xaxes(title_text="Sales (₹)")
                fig_hist.update_yaxes(title_text="Frequency")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                if 'Category' in df.columns:
                    category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
                    fig_pie = px.pie(
                        names=category_sales.index, 
                        values=category_sales.values,
                        title="Sales Distribution by Category",
                        hole=0.3# Creates a donut chart for modern look
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)


        
        except Exception as e:
            st.warning(f"Could not parse dates: {str(e)}")
    
    # Enhanced data preprocessing options
    st.markdown("### 🔧 Data Preprocessing")

    # Create tabs for different preprocessing options
    preprocessing_tab1, preprocessing_tab2, preprocessing_tab3 = st.tabs([
        "🧹 Data Cleaning",
        "⚖️ Scaling & Encoding",
        "🔬 Feature Engineering"
    ])

    with preprocessing_tab1:
        st.markdown("#### Data Cleaning Options")

        col1, col2 = st.columns(2)

        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["Keep as is", "Forward fill", "Backward fill", "Drop rows", "Interpolate"]
            )

            outlier_handling = st.selectbox(
                "Outlier Handling",
                ["Keep as is", "Remove outliers", "Cap outliers", "Transform outliers"]
            )

        with col2:
            date_format = st.selectbox(
                "Date Format",
                ["Auto-detect", "YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"]
            )

            duplicate_handling = st.selectbox(
                "Handle Duplicates",
                ["Keep as is", "Remove duplicates", "Keep first occurrence"]
            )

    with preprocessing_tab2:
        st.markdown("#### Scaling & Encoding Options")

        col1, col2 = st.columns(2)

        with col1:
            scaling_method = st.selectbox(
                "Scaling Method",
                ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
            )

            categorical_encoding = st.selectbox(
                "Categorical Encoding",
                ["None", "Label Encoding", "One-Hot Encoding"]
            )

        with col2:
            # Data split ratio slider
            data_split_ratio = st.slider(
                "Train-Test Split Ratio",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Percentage of data to use for training (remaining for testing)"
            )

            st.metric("Training Data", f"{data_split_ratio:.0%}")
            st.metric("Testing Data", f"{1-data_split_ratio:.0%}")

    with preprocessing_tab3:
        st.markdown("#### Feature Engineering Options")

        col1, col2 = st.columns(2)

        with col1:
            lag_features = st.multiselect(
                "Lag Features",
                ["1-day lag", "7-day lag", "30-day lag"],
                default=["1-day lag", "7-day lag"]
            )

            rolling_features = st.multiselect(
                "Rolling Statistics",
                ["7-day rolling mean", "30-day rolling mean", "7-day rolling std"],
                default=["7-day rolling mean"]
            )

        with col2:
            datetime_features = st.multiselect(
                "Date-time Features",
                ["Day of week", "Month", "Quarter", "Is weekend", "Day of year"],
                default=["Day of week", "Month"]
            )

            interaction_features = st.multiselect(
                "Interaction Features",
                ["Store-Category interaction", "Category-Promotion interaction"],
                default=[]
            )
    
    if st.button("Apply Preprocessing", type="primary"):
        processed_df = df.copy()

        # Handle missing values
        if handle_missing == "Forward fill":
            processed_df = processed_df.ffill()
        elif handle_missing == "Backward fill":
            processed_df = processed_df.bfill()
        elif handle_missing == "Drop rows":
            processed_df = processed_df.dropna()
        elif handle_missing == "Interpolate":
            processed_df = processed_df.interpolate()

        # Handle duplicates
        if duplicate_handling == "Remove duplicates":
            processed_df = processed_df.drop_duplicates()
        elif duplicate_handling == "Keep first occurrence":
            processed_df = processed_df.drop_duplicates(keep='first')

        # Handle outliers
        if outlier_handling == "Remove outliers" and 'Sales' in processed_df.columns:
            Q1 = processed_df['Sales'].quantile(0.25)
            Q3 = processed_df['Sales'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df = processed_df[
                (processed_df['Sales'] >= lower_bound) &
                (processed_df['Sales'] <= upper_bound)
            ]
        elif outlier_handling == "Cap outliers" and 'Sales' in processed_df.columns:
            Q1 = processed_df['Sales'].quantile(0.25)
            Q3 = processed_df['Sales'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            processed_df['Sales'] = processed_df['Sales'].clip(lower=lower_bound, upper=upper_bound)

        # Apply scaling
        if scaling_method != "None" and 'Sales' in processed_df.columns:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif scaling_method == "RobustScaler":
                scaler = RobustScaler()

            processed_df['Sales'] = scaler.fit_transform(processed_df[['Sales']])

        # Apply categorical encoding
        if categorical_encoding != "None":
            categorical_columns = ['Store', 'Category']
            existing_cat_cols = [col for col in categorical_columns if col in processed_df.columns]

            if categorical_encoding == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in existing_cat_cols:
                    processed_df[col] = le.fit_transform(processed_df[col])

            elif categorical_encoding == "One-Hot Encoding":
                processed_df = pd.get_dummies(processed_df, columns=existing_cat_cols)

        # Feature engineering
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'], dayfirst=True, errors='coerce')

            # Lag features
            if "1-day lag" in lag_features:
                processed_df['Sales_lag_1'] = processed_df['Sales'].shift(1)
            if "7-day lag" in lag_features:
                processed_df['Sales_lag_7'] = processed_df['Sales'].shift(7)
            if "30-day lag" in lag_features:
                processed_df['Sales_lag_30'] = processed_df['Sales'].shift(30)

            # Rolling features
            if "7-day rolling mean" in rolling_features:
                processed_df['Sales_roll_7_mean'] = processed_df['Sales'].rolling(window=7).mean()
            if "30-day rolling mean" in rolling_features:
                processed_df['Sales_roll_30_mean'] = processed_df['Sales'].rolling(window=30).mean()
            if "7-day rolling std" in rolling_features:
                processed_df['Sales_roll_7_std'] = processed_df['Sales'].rolling(window=7).std()

            # Date-time features
            if "Day of week" in datetime_features:
                processed_df['Day_of_week'] = processed_df['Date'].dt.dayofweek
            if "Month" in datetime_features:
                processed_df['Month'] = processed_df['Date'].dt.month
            if "Quarter" in datetime_features:
                processed_df['Quarter'] = processed_df['Date'].dt.quarter
            if "Is weekend" in datetime_features:
                processed_df['Is_weekend'] = (processed_df['Date'].dt.dayofweek >= 5).astype(int)
            if "Day of year" in datetime_features:
                processed_df['Day_of_year'] = processed_df['Date'].dt.dayofyear

            # Interaction features
            if "Store-Category interaction" in interaction_features and 'Store' in processed_df.columns and 'Category' in processed_df.columns:
                processed_df['Store_Category'] = processed_df['Store'].astype(str) + '_' + processed_df['Category'].astype(str)
            if "Category-Promotion interaction" in interaction_features and 'Category' in processed_df.columns and 'Promotion' in processed_df.columns:
                processed_df['Category_Promotion'] = processed_df['Category'].astype(str) + '_Promo_' + processed_df['Promotion'].astype(str)

        # Store preprocessing configuration
        st.session_state.preprocessing_config = {
            'handle_missing': handle_missing,
            'outlier_handling': outlier_handling,
            'scaling_method': scaling_method,
            'categorical_encoding': categorical_encoding,
            'data_split_ratio': data_split_ratio,
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'datetime_features': datetime_features,
            'interaction_features': interaction_features
        }

        # Store processed data
        st.session_state.processed_data = processed_df
        
        # === DETAILED PREPROCESSING REPORT (IN EXPANDER) ===
        st.success("✅ Data preprocessing completed successfully!")
        
        with st.expander("📊 View Detailed Preprocessing Report", expanded=False):
            st.markdown("### 📊 Preprocessing Report")
            
            # 1. Shape Changes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", len(df), delta=None)
            with col2:
                rows_change = len(processed_df) - len(df)
                st.metric("Processed Rows", len(processed_df), delta=rows_change)
            with col3:
                cols_change = len(processed_df.columns) - len(df.columns)
                st.metric("Total Columns", len(processed_df.columns), delta=cols_change)
            
            # 2. Data Quality Improvements
            st.markdown("#### 🧹 Data Quality Changes")
            quality_col1, quality_col2 = st.columns(2)
            
            with quality_col1:
                # Missing values
                original_missing = df.isnull().sum().sum()
                processed_missing = processed_df.isnull().sum().sum()
                missing_handled = original_missing - processed_missing
                
                st.markdown(f"""
                **Missing Values:**
                - Before: `{original_missing:,}` missing values
                - After: `{processed_missing:,}` missing values
                - **Handled: {missing_handled:,} values** {'✅' if missing_handled > 0 else ''}
                """)
            
            with quality_col2:
                # Duplicates
                original_duplicates = df.duplicated().sum()
                processed_duplicates = processed_df.duplicated().sum()
                duplicates_removed = original_duplicates - processed_duplicates
                
                st.markdown(f"""
                **Duplicate Rows:**
                - Before: `{original_duplicates:,}` duplicates
                - After: `{processed_duplicates:,}` duplicates
                - **Removed: {duplicates_removed:,} rows** {'✅' if duplicates_removed > 0 else ''}
                """)
            
            # 2.5 Data Quality Visualization
            if original_missing > 0 or original_duplicates > 0:
                st.markdown("#### 📈 Data Quality Improvement Visualization")
                quality_viz_col1, quality_viz_col2 = st.columns(2)
                
                with quality_viz_col1:
                    # Missing values chart
                    fig_missing = go.Figure(data=[
                        go.Bar(name='Before', x=['Missing Values'], y=[original_missing], marker_color='indianred'),
                        go.Bar(name='After', x=['Missing Values'], y=[processed_missing], marker_color='lightseagreen')
                    ])
                    fig_missing.update_layout(
                        title="Missing Values: Before vs After",
                        yaxis_title="Count",
                        barmode='group',
                        height=300
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
                
                with quality_viz_col2:
                    # Duplicates chart
                    fig_dup = go.Figure(data=[
                        go.Bar(name='Before', x=['Duplicates'], y=[original_duplicates], marker_color='indianred'),
                        go.Bar(name='After', x=['Duplicates'], y=[processed_duplicates], marker_color='lightseagreen')
                    ])
                    fig_dup.update_layout(
                        title="Duplicate Rows: Before vs After",
                        yaxis_title="Count",
                        barmode='group',
                        height=300
                    )
                    st.plotly_chart(fig_dup, use_container_width=True)
            
            # 3. Outlier Handling (if applicable)
            if outlier_handling != "Keep as is" and 'Sales' in df.columns:
                st.markdown("#### 📉 Outlier Handling")
                outlier_col1, outlier_col2 = st.columns(2)
                
                with outlier_col1:
                    Q1 = df['Sales'].quantile(0.25)
                    Q3 = df['Sales'].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers_count = len(df[(df['Sales'] < lower_bound) | (df['Sales'] > upper_bound)])
                    
                    st.markdown(f"""
                    **Sales Outliers:**
                    - Outliers detected: `{outliers_count:,}` rows
                    - Lower bound: `₹{lower_bound:,.2f}`
                    - Upper bound: `₹{upper_bound:,.2f}`
                    """)
                    
                    if outlier_handling == "Remove outliers":
                        st.markdown(f"""
                        **Action Taken:**
                        - Method: Remove outliers
                        - **Rows removed: {outliers_count:,}** ❌
                        """)
                    elif outlier_handling == "Cap outliers":
                        st.markdown(f"""
                        **Action Taken:**
                        - Method: Cap to bounds
                        - **Values capped: {outliers_count:,}** 📌
                        """)
                
                with outlier_col2:
                    # Outlier visualization
                    fig_outlier = go.Figure()
                    fig_outlier.add_trace(go.Box(y=df['Sales'], name='Original', marker_color='indianred'))
                    fig_outlier.add_trace(go.Box(y=processed_df['Sales'], name='Processed', marker_color='lightseagreen'))
                    fig_outlier.update_layout(
                        title="Sales Distribution: Outliers Comparison",
                        yaxis_title="Sales (₹)",
                        height=300
                    )
                    st.plotly_chart(fig_outlier, use_container_width=True)
            
            # 4. Feature Engineering Summary
            new_columns = list(set(processed_df.columns) - set(df.columns))
            if new_columns:
                st.markdown("#### 🔬 New Features Created")
                st.markdown(f"**Total new features: {len(new_columns)}**")
                
                # Categorize features
                lag_cols = [col for col in new_columns if 'lag' in col.lower()]
                roll_cols = [col for col in new_columns if 'roll' in col.lower()]
                datetime_cols = [col for col in new_columns if any(x in col.lower() for x in ['day', 'month', 'quarter', 'weekend', 'year'])]
                interaction_cols = [col for col in new_columns if '_' in col and col not in lag_cols + roll_cols + datetime_cols]
                other_cols = [col for col in new_columns if col not in lag_cols + roll_cols + datetime_cols + interaction_cols]
                
                feature_col1, feature_col2, feature_col3 = st.columns(3)
                
                with feature_col1:
                    if lag_cols:
                        st.markdown(f"**Lag Features ({len(lag_cols)}):**")
                        for col in lag_cols:
                            st.write(f"- `{col}`")
                    
                    if roll_cols:
                        st.markdown(f"**Rolling Features ({len(roll_cols)}):**")
                        for col in roll_cols:
                            st.write(f"- `{col}`")
                
                with feature_col2:
                    if datetime_cols:
                        st.markdown(f"**DateTime Features ({len(datetime_cols)}):**")
                        for col in datetime_cols:
                            st.write(f"- `{col}`")
                
                with feature_col3:
                    if interaction_cols:
                        st.markdown(f"**Interaction Features ({len(interaction_cols)}):**")
                        for col in interaction_cols:
                            st.write(f"- `{col}`")
                    
                    if other_cols:
                        st.markdown(f"**Other Features ({len(other_cols)}):**")
                        for col in other_cols:
                            st.write(f"- `{col}`")
            
            # 5. Data Preview: Before vs After
            st.markdown("---")
            st.markdown("#### 👁️ Data Preview: Before vs After")
            
            preview_col1, preview_col2 = st.columns(2)
            
            with preview_col1:
                st.markdown("**Original Data (First 5 rows):**")
                st.dataframe(df.head())
            
            with preview_col2:
                st.markdown("**Processed Data (First 5 rows):**")
                st.dataframe(processed_df.head())
            
            # 6. Statistical Summary Comparison
            if 'Sales' in df.columns and 'Sales' in processed_df.columns:
                st.markdown("---")
                st.markdown("#### 📈 Sales Statistics Comparison")
                
                stat_col1, stat_col2 = st.columns([1, 2])
                
                with stat_col1:
                    stats_comparison = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        'Original': [
                            f"₹{df['Sales'].mean():,.2f}",
                            f"₹{df['Sales'].median():,.2f}",
                            f"₹{df['Sales'].std():,.2f}",
                            f"₹{df['Sales'].min():,.2f}",
                            f"₹{df['Sales'].max():,.2f}"
                        ],
                        'Processed': [
                            f"₹{processed_df['Sales'].mean():,.2f}",
                            f"₹{processed_df['Sales'].median():,.2f}",
                            f"₹{processed_df['Sales'].std():,.2f}",
                            f"₹{processed_df['Sales'].min():,.2f}",
                            f"₹{processed_df['Sales'].max():,.2f}"
                        ]
                    })
                    st.dataframe(stats_comparison, hide_index=True)
                
                with stat_col2:
                    # Sales distribution comparison
                    fig_sales_dist = go.Figure()
                    fig_sales_dist.add_trace(go.Histogram(x=df['Sales'], name='Original', opacity=0.7, marker_color='indianred', nbinsx=30))
                    fig_sales_dist.add_trace(go.Histogram(x=processed_df['Sales'], name='Processed', opacity=0.7, marker_color='lightseagreen', nbinsx=30))
                    fig_sales_dist.update_layout(
                        title="Sales Distribution: Before vs After",
                        xaxis_title="Sales (₹)",
                        yaxis_title="Frequency",
                        barmode='overlay',
                        height=400
                    )
                    st.plotly_chart(fig_sales_dist, use_container_width=True)



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
    
    st.dataframe(sample_format)
