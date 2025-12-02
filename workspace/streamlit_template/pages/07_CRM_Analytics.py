# workspace/streamlit_template/pages/07_CRM_Analytics.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os

# allow importing utils from parent folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.ui_components import render_sidebar
from utils.load_css import load_css_for_page   # shared loader (robust)

# -----------------------------------------------------------
# IMPORTANT: set_page_config must be the first Streamlit call
# -----------------------------------------------------------
st.set_page_config(page_title="CRM Analytics", page_icon="🤝", layout="wide")

# load css safely (no FileNotFoundError)
load_css_for_page(__file__)
# -----------------------------------------------------------

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

st.title("🤝 Customer Relationship Management (CRM)")
st.markdown("Customer analytics, segmentation, and lifetime value analysis")

# Check for data
if 'uploaded_data' not in st.session_state and 'processed_data' not in st.session_state:
    st.warning("⚠️ No data found. Please upload data first!")
    st.markdown("[👆 Go to Upload Data page](1_📊_Upload_Data)")
    st.stop()

# Get data
df = st.session_state.get('processed_data', st.session_state.get('uploaded_data')).copy()

# Main tabs - Reorganized for better UX
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Customer Overview & Insights",
    "🎯 RFM & Segmentation",
    "💰 Value Analysis",
    "⚠️ Churn Prediction"
])

with tab1:
    st.markdown("### 📊 Customer Overview & Insights")
    st.caption("Customer base analysis and key performance metrics")
    
    # Create synthetic customer ID if not present
    if 'Customer_ID' not in df.columns and 'Store' in df.columns:
        # Generate synthetic customer data for demo
        df['Customer_ID'] = df.index % 1000  # Simulate 1000 customers
    
    if 'Customer_ID' in df.columns:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            unique_customers = df['Customer_ID'].nunique()
            st.metric("Total Customers", f"{unique_customers:,}")
        
        with col2:
            if 'Sales' in df.columns:
                avg_purchase = df.groupby('Customer_ID')['Sales'].sum().mean()
                st.metric("Avg Customer Value", f"₹{avg_purchase:,.0f}")
        
        with col3:
            if 'Sales' in df.columns:
                purchases_per_customer = len(df) / unique_customers
                st.metric("Avg Purchases/Customer", f"{purchases_per_customer:.1f}")
        
        with col4:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                last_30_days = df[df['Date'] >= (df['Date'].max() - timedelta(days=30))]
                active_customers = last_30_days['Customer_ID'].nunique()
                st.metric("Active Customers (30d)", f"{active_customers:,}")
        
        # Customer distribution
        st.markdown("#### Customer Value Distribution")
        
        if 'Sales' in df.columns:
            customer_value = df.groupby('Customer_ID')['Sales'].sum().reset_index()
            customer_value.columns = ['Customer_ID', 'Total_Value']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    customer_value,
                    x='Total_Value',
                    nbins=50,
                    title="Distribution of Customer Total Value",
                    labels={'Total_Value': 'Customer Lifetime Value (₹)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top customers
                top_customers = customer_value.nlargest(10, 'Total_Value')
                fig = px.bar(
                    top_customers,
                    x='Customer_ID',
                    y='Total_Value',
                    title="Top 10 Customers by Value",
                    labels={'Customer_ID': 'Customer ID', 'Total_Value': 'Total Value (₹)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Customer cohort analysis
        if 'Date' in df.columns and 'Sales' in df.columns:
            st.markdown("#### Customer Acquisition Cohort")
            
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            df['YearMonth'] = df['Date'].dt.to_period('M')
            
            # First purchase date per customer
            first_purchase = df.groupby('Customer_ID')['Date'].min().reset_index()
            first_purchase.columns = ['Customer_ID', 'First_Purchase']
            first_purchase['Cohort'] = first_purchase['First_Purchase'].dt.to_period('M')
            
            df_cohort = df.merge(first_purchase[['Customer_ID', 'Cohort']], on='Customer_ID')
            
            cohort_data = df_cohort.groupby('Cohort')['Customer_ID'].nunique().reset_index()
            cohort_data.columns = ['Cohort', 'Customers']
            cohort_data['Cohort'] = cohort_data['Cohort'].astype(str)
            
            fig = px.line(
                cohort_data.tail(12),
                x='Cohort',
                y='Customers',
                title="Monthly Customer Acquisition",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Customer ID column not found. Please add a 'Customer_ID' column to your dataset for customer analytics.")

with tab2:
    st.markdown("### 🎯 RFM & Customer Segmentation")
    st.caption("Behavioral segmentation using RFM and machine learning clustering")
    
    # RFM Analysis Section
    st.markdown("#### 📊 RFM Analysis")
    st.markdown("**Recency, Frequency, Monetary** - Segment customers based on purchase behavior")
    
    if 'Customer_ID' not in df.columns:
        df['Customer_ID'] = df.index % 1000
    
    if 'Customer_ID' in df.columns and 'Date' in df.columns and 'Sales' in df.columns:
        # Calculate RFM metrics
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        snapshot_date = df['Date'].max() + timedelta(days=1)
        
        rfm = df.groupby('Customer_ID').agg({
            'Date': lambda x: (snapshot_date - x.max()).days,  # Recency
            'Sales': ['count', 'sum']  # Frequency and Monetary
        }).reset_index()
        
        rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
        
        # Calculate RFM scores (1-5) with error handling
        try:
            # Check if we have enough unique values for quintiles
            if rfm['Recency'].nunique() >= 5:
                rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            else:
                # Use fewer bins if not enough unique values
                n_bins = min(5, rfm['Recency'].nunique())
                rfm['R_Score'] = pd.qcut(rfm['Recency'], q=n_bins, labels=list(range(n_bins, 0, -1)), duplicates='drop')

            if rfm['Frequency'].nunique() >= 5:
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            else:
                n_bins = min(5, rfm['Frequency'].nunique())
                rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=n_bins, labels=list(range(1, n_bins+1)), duplicates='drop')

            if rfm['Monetary'].nunique() >= 5:
                rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            else:
                n_bins = min(5, rfm['Monetary'].nunique())
                rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=n_bins, labels=list(range(1, n_bins+1)), duplicates='drop')

        except ValueError as e:
            # Fallback method if qcut fails
            st.warning(f"Using alternative binning method due to data distribution: {str(e)}")

            # Use percentile-based scoring as fallback
            rfm['R_Score'] = pd.cut(rfm['Recency'], bins=5, labels=[5, 4, 3, 2, 1])
            rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=5, labels=[1, 2, 3, 4, 5])
            rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=5, labels=[1, 2, 3, 4, 5])
        
        # Convert to numeric
        rfm['R_Score'] = pd.to_numeric(rfm['R_Score'])
        rfm['F_Score'] = pd.to_numeric(rfm['F_Score'])
        rfm['M_Score'] = pd.to_numeric(rfm['M_Score'])
        
        rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
        
        # Segment customers
        def segment_customer(row):
            if row['RFM_Score'] >= 13:
                return 'Champions'
            elif row['RFM_Score'] >= 11:
                return 'Loyal Customers'
            elif row['RFM_Score'] >= 9:
                return 'Potential Loyalists'
            elif row['RFM_Score'] >= 7:
                return 'At Risk'
            else:
                return 'Lost'
        
        rfm['Segment'] = rfm.apply(segment_customer, axis=1)
        
        # Store RFM results
        st.session_state.rfm_data = rfm
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_recency = rfm['Recency'].mean()
            st.metric("Avg Recency", f"{avg_recency:.0f} days")
        
        with col2:
            avg_frequency = rfm['Frequency'].mean()
            st.metric("Avg Frequency", f"{avg_frequency:.1f} orders")
        
        with col3:
            avg_monetary = rfm['Monetary'].mean()
            st.metric("Avg Monetary", f"₹{avg_monetary:,.0f}")
        
        # Segment distribution
        st.markdown("#### Customer Segments")
        
        col1, col2 = st.columns(2)
        
        with col1:
            segment_counts = rfm['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            fig = px.pie(
                segment_counts,
                values='Count',
                names='Segment',
                title="Customer Segment Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_value = rfm.groupby('Segment')['Monetary'].sum().reset_index()
            segment_value.columns = ['Segment', 'Total_Value']
            
            fig = px.bar(
                segment_value,
                x='Segment',
                y='Total_Value',
                title="Total Value by Segment",
                color='Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM scatter plot
        st.markdown("#### RFM Analysis - Recency vs Monetary")
        
        fig = px.scatter(
            rfm,
            x='Recency',
            y='Monetary',
            size='Frequency',
            color='Segment',
            title="Customer Segmentation (RFM)",
            labels={
                'Recency': 'Days Since Last Purchase',
                'Monetary': 'Total Spent (₹)',
                'Frequency': 'Purchase Count'
            },
            hover_data=['Customer_ID', 'RFM_Score']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed segment info
        st.markdown("#### Segment Details")
        
        segment_details = rfm.groupby('Segment').agg({
            'Customer_ID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'RFM_Score': 'mean'
        }).round(2).reset_index()
        
        segment_details.columns = ['Segment', 'Count', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary (₹)', 'Avg RFM Score']
        
        st.dataframe(segment_details, use_container_width=True)
        
        # Download RFM data
        csv = rfm.to_csv(index=False)
        st.download_button(
            label="📥 Download RFM Analysis",
            data=csv,
            file_name="rfm_analysis.csv",
            mime="text/csv"
        )
    else:
        st.info("Required columns not found. Ensure your data has 'Customer_ID', 'Date', and 'Sales' columns.")
    
    # Machine Learning Segmentation Section
    st.markdown("---")
    st.markdown("#### 👥 ML-Based Customer Segmentation")
    st.markdown("Advanced clustering analysis using machine learning")
    
    if 'Customer_ID' not in df.columns:
        df['Customer_ID'] = df.index % 1000
    
    if 'Customer_ID' in df.columns and 'Sales' in df.columns:
        # Prepare features for clustering
        customer_features = df.groupby('Customer_ID').agg({
            'Sales': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        
        customer_features.columns = ['Customer_ID', 'Total_Sales', 'Avg_Sales', 'Purchase_Count', 'Sales_Std']
        customer_features['Sales_Std'] = customer_features['Sales_Std'].fillna(0)
        
        # Clustering configuration
        col1, col2 = st.columns(2)
        
        with col1:
            n_clusters = st.slider("Number of Segments", 3, 8, 4)
        
        with col2:
            clustering_method = st.selectbox(
                "Clustering Method",
                ["K-Means", "Custom RFM-based"]
            )
        
        if st.button("🎯 Run Segmentation", type="primary"):
            with st.spinner("Performing customer segmentation..."):
                # Prepare features
                features = customer_features[['Total_Sales', 'Avg_Sales', 'Purchase_Count', 'Sales_Std']].copy()
                
                # Standardize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                customer_features['Cluster'] = kmeans.fit_predict(features_scaled)
                
                # Store results
                st.session_state.segmentation_results = customer_features
                
                st.success(f"✅ Segmentation complete! {n_clusters} segments identified.")
        
        # Display results
        if 'segmentation_results' in st.session_state:
            seg_results = st.session_state.segmentation_results
            
            # Cluster statistics
            st.markdown("#### Segment Statistics")
            
            cluster_stats = seg_results.groupby('Cluster').agg({
                'Customer_ID': 'count',
                'Total_Sales': 'mean',
                'Avg_Sales': 'mean',
                'Purchase_Count': 'mean'
            }).round(2).reset_index()
            
            cluster_stats.columns = ['Segment', 'Customers', 'Avg Total Sales', 'Avg Per-Purchase', 'Avg Purchases']
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(
                    seg_results,
                    x='Total_Sales',
                    y='Purchase_Count',
                    color='Cluster',
                    size='Avg_Sales',
                    title="Customer Segments: Total Sales vs Purchase Frequency",
                    labels={
                        'Total_Sales': 'Total Sales (₹)',
                        'Purchase_Count': 'Number of Purchases',
                        'Cluster': 'Segment'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                cluster_size = seg_results['Cluster'].value_counts().reset_index()
                cluster_size.columns = ['Segment', 'Count']
                
                fig = px.pie(
                    cluster_size,
                    values='Count',
                    names='Segment',
                    title="Customer Distribution Across Segments"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Customer data not available for segmentation.")

with tab3:
    st.markdown("### 💰 Customer Value Analysis")
    st.caption("Customer Lifetime Value (CLV) predictions and high-value customer identification")
    
    if 'Customer_ID' not in df.columns:
        df['Customer_ID'] = df.index % 1000
    
    if 'Customer_ID' in df.columns and 'Date' in df.columns and 'Sales' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate CLV metrics
        customer_lifetime = df.groupby('Customer_ID').agg({
            'Sales': 'sum',
            'Date': ['min', 'max', 'count']
        }).reset_index()
        
        customer_lifetime.columns = ['Customer_ID', 'Total_Value', 'First_Purchase', 'Last_Purchase', 'Purchase_Count']
        
        # Calculate customer lifespan in days
        customer_lifetime['Lifespan_Days'] = (customer_lifetime['Last_Purchase'] - customer_lifetime['First_Purchase']).dt.days
        customer_lifetime['Lifespan_Days'] = customer_lifetime['Lifespan_Days'].apply(lambda x: max(x, 1))
        
        # Simple CLV calculation
        customer_lifetime['Avg_Purchase_Value'] = customer_lifetime['Total_Value'] / customer_lifetime['Purchase_Count']
        customer_lifetime['Purchase_Frequency'] = customer_lifetime['Purchase_Count'] / customer_lifetime['Lifespan_Days'] * 365
        
        # Predict CLV (simple model: current value + projected future value)
        customer_lifetime['Predicted_Annual_Value'] = customer_lifetime['Avg_Purchase_Value'] * customer_lifetime['Purchase_Frequency']
        customer_lifetime['Predicted_3Year_CLV'] = customer_lifetime['Predicted_Annual_Value'] * 3
        
        # Store CLV results
        st.session_state.clv_results = customer_lifetime
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_clv = customer_lifetime['Predicted_3Year_CLV'].mean()
            st.metric("Avg 3-Year CLV", f"₹{avg_clv:,.0f}")
        
        with col2:
            top_10_pct = customer_lifetime['Predicted_3Year_CLV'].quantile(0.9)
            st.metric("Top 10% CLV Threshold", f"₹{top_10_pct:,.0f}")
        
        with col3:
            total_clv = customer_lifetime['Predicted_3Year_CLV'].sum()
            st.metric("Total Predicted CLV", f"₹{total_clv:,.0f}")
        
        with col4:
            high_value_count = (customer_lifetime['Predicted_3Year_CLV'] >= top_10_pct).sum()
            st.metric("High-Value Customers", f"{high_value_count}")
        
        # CLV distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                customer_lifetime,
                x='Predicted_3Year_CLV',
                nbins=50,
                title="Distribution of Predicted 3-Year CLV",
                labels={'Predicted_3Year_CLV': 'Predicted 3-Year CLV (₹)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top customers by CLV
            top_clv = customer_lifetime.nlargest(10, 'Predicted_3Year_CLV')
            fig = px.bar(
                top_clv,
                x='Customer_ID',
                y='Predicted_3Year_CLV',
                title="Top 10 Customers by Predicted CLV",
                labels={'Customer_ID': 'Customer ID', 'Predicted_3Year_CLV': 'Predicted 3-Year CLV (₹)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # CLV segments
        st.markdown("#### CLV-Based Customer Segments")
        
        customer_lifetime['CLV_Segment'] = pd.qcut(
            customer_lifetime['Predicted_3Year_CLV'].fillna(0),
            q=4,
            labels=['Low Value', 'Medium Value', 'High Value', 'Very High Value']
        )
        
        clv_segment_stats = customer_lifetime.groupby('CLV_Segment').agg({
            'Customer_ID': 'count',
            'Total_Value': 'sum',
            'Predicted_3Year_CLV': 'mean'
        }).reset_index()
        
        clv_segment_stats.columns = ['Segment', 'Customer Count', 'Current Total Value', 'Avg Predicted CLV']
        
        st.dataframe(clv_segment_stats, use_container_width=True)
        
        # Scatter plot: Purchase Frequency vs CLV
        fig = px.scatter(
            customer_lifetime,
            x='Purchase_Frequency',
            y='Predicted_3Year_CLV',
            color='CLV_Segment',
            size='Total_Value',
            title="Purchase Frequency vs Predicted CLV",
            labels={
                'Purchase_Frequency': 'Annual Purchase Frequency',
                'Predicted_3Year_CLV': 'Predicted 3-Year CLV (₹)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Required data not available for CLV calculation.")

with tab4:
    st.markdown("### ⚠️ Churn Prediction & Retention")
    st.caption("Identify at-risk customers and recommended retention strategies")
    
    if 'Customer_ID' not in df.columns:
        df['Customer_ID'] = df.index % 1000
    
    if 'Customer_ID' in df.columns and 'Date' in df.columns and 'Sales' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate churn indicators
        snapshot_date = df['Date'].max()
        
        churn_data = df.groupby('Customer_ID').agg({
            'Date': 'max',
            'Sales': ['sum', 'mean', 'count']
        }).reset_index()
        
        churn_data.columns = ['Customer_ID', 'Last_Purchase', 'Total_Sales', 'Avg_Sales', 'Purchase_Count']
        
        # Days since last purchase
        churn_data['Days_Since_Purchase'] = (snapshot_date - churn_data['Last_Purchase']).dt.days
        
        # Define churn threshold (e.g., no purchase in 90 days)
        churn_threshold = st.slider("Churn Threshold (days)", 30, 180, 90)
        
        churn_data['Churn_Risk'] = churn_data['Days_Since_Purchase'].apply(
            lambda x: 'High Risk' if x > churn_threshold 
            else ('Medium Risk' if x > churn_threshold/2 else 'Low Risk')
        )
        
        # Store churn results
        st.session_state.churn_results = churn_data
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            high_risk = (churn_data['Churn_Risk'] == 'High Risk').sum()
            st.metric("High Risk Customers", f"{high_risk}")
        
        with col2:
            medium_risk = (churn_data['Churn_Risk'] == 'Medium Risk').sum()
            st.metric("Medium Risk Customers", f"{medium_risk}")
        
        with col3:
            low_risk = (churn_data['Churn_Risk'] == 'Low Risk').sum()
            st.metric("Low Risk Customers", f"{low_risk}")
        
        with col4:
            churn_rate = (high_risk / len(churn_data)) * 100 if len(churn_data) > 0 else 0
            st.metric("Churn Risk Rate", f"{churn_rate:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = churn_data['Churn_Risk'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            
            fig = px.pie(
                risk_counts,
                values='Count',
                names='Risk Level',
                title="Customer Churn Risk Distribution",
                color='Risk Level',
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange',
                    'High Risk': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                churn_data,
                x='Days_Since_Purchase',
                y='Total_Sales',
                color='Churn_Risk',
                size='Purchase_Count',
                title="Churn Risk: Recency vs Customer Value",
                labels={
                    'Days_Since_Purchase': 'Days Since Last Purchase',
                    'Total_Sales': 'Total Customer Value (₹)'
                },
                color_discrete_map={
                    'Low Risk': 'green',
                    'Medium Risk': 'orange',
                    'High Risk': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk customers
        st.markdown("#### High-Risk Customers Requiring Attention")
        
        high_risk_customers = churn_data[churn_data['Churn_Risk'] == 'High Risk'].nlargest(10, 'Total_Sales')
        
        st.dataframe(
            high_risk_customers[['Customer_ID', 'Days_Since_Purchase', 'Total_Sales', 'Purchase_Count']],
            use_container_width=True
        )
        
        # Retention strategies
        st.markdown("#### 💡 Recommended Retention Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **For High-Risk Customers:**
            - Send personalized re-engagement emails
            - Offer exclusive discounts or promotions
            - Request feedback on their experience
            - Provide loyalty rewards or incentives
            """)
        
        with col2:
            st.markdown("""
            **For Medium-Risk Customers:**
            - Send reminder emails about new products
            - Offer birthday or anniversary discounts
            - Invite to exclusive events or sales
            - Share relevant content and updates
            """)
    else:
        st.info("Required data not available for churn prediction.")
