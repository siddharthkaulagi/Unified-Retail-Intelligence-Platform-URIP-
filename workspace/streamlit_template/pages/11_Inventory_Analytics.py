import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import render_sidebar

# --- HELPER FUNCTIONS ---

def load_css():
    """Load custom CSS styles."""
    try:
        with open("assets/custom.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def compute_abc(df, item_col, value_col):
    """
    Perform ABC Analysis based on the selected value column.
    Returns a dataframe with 'Class', 'Cumulative Percentage', etc.
    """
    # Group by item and sum value
    abc_df = df.groupby(item_col)[value_col].sum().reset_index()
    
    # Sort descending
    abc_df = abc_df.sort_values(value_col, ascending=False)
    
    # Calculate cumulative metrics
    abc_df['Cumulative Value'] = abc_df[value_col].cumsum()
    total_value = abc_df[value_col].sum()
    abc_df['Cumulative Percentage'] = (abc_df['Cumulative Value'] / total_value) * 100
    
    # Classification Logic
    def classify(pct):
        if pct <= 80: return 'A'
        elif pct <= 95: return 'B'
        else: return 'C'
        
    abc_df['Class'] = abc_df['Cumulative Percentage'].apply(classify)
    return abc_df

def compute_xyz(df, item_col, value_col, date_col):
    """
    Perform XYZ Analysis based on demand volatility (Coefficient of Variation).
    Returns a dataframe with 'XYZ_Class', 'CV', etc.
    """
    if date_col == 'None' or date_col not in df.columns:
        return None

    # Ensure date is datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])

    # Calculate statistics per item
    # We aggregate value_col (e.g. Sales) by item and date first to get daily/transactional demand
    # Then compute std/mean of those demands
    # Note: This assumes multiple transactions per item. If only one, std is NaN.
    
    # Group by Item and Date to get daily demand (if multiple entries per day)
    daily_demand = df.groupby([item_col, date_col])[value_col].sum().reset_index()
    
    xyz_stats = daily_demand.groupby(item_col)[value_col].agg(['mean', 'std']).reset_index()
    
    # Handle items with single transaction (std is NaN) -> treat as Volatile (Z) or Stable (X)? 
    # Usually single transaction implies low frequency, maybe Z. Let's fill NaN std with 0 for now but CV will be 0.
    # Better approach: fillna(0) for std implies perfectly stable if only 1 data point.
    xyz_stats['std'] = xyz_stats['std'].fillna(0)
    
    # Calculate Coefficient of Variation (CV)
    # Avoid division by zero
    xyz_stats['CV'] = xyz_stats.apply(lambda row: row['std'] / row['mean'] if row['mean'] > 0 else 0, axis=1)
    
    def classify(cv):
        if cv <= 0.5: return 'X'     # Stable
        elif cv <= 1.0: return 'Y'   # Variable
        else: return 'Z'             # Volatile
        
    xyz_stats['XYZ_Class'] = xyz_stats['CV'].apply(classify)
    return xyz_stats

def compute_fsn(df, item_col, date_col):
    """
    Perform FSN Analysis based on Recency (Days since last sale).
    F (Fast): <= 30 days
    S (Slow): 31 - 90 days
    N (Non-moving): > 90 days
    """
    if date_col == 'None' or date_col not in df.columns:
        return None
        
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Find last sale date for the whole dataset (reference date)
    ref_date = df[date_col].max()
    
    # Find last sale date per item
    last_sales = df.groupby(item_col)[date_col].max().reset_index()
    last_sales.rename(columns={date_col: 'Last_Sale_Date'}, inplace=True)
    
    last_sales['Days_Since_Last_Sale'] = (ref_date - last_sales['Last_Sale_Date']).dt.days
    
    def classify(days):
        if days <= 30: return 'F'
        elif days <= 90: return 'S'
        else: return 'N'
        
    last_sales['FSN_Class'] = last_sales['Days_Since_Last_Sale'].apply(classify)
    return last_sales

# --- MAIN PAGE CONFIG ---

load_css()
st.set_page_config(page_title="Inventory Analytics", page_icon="📦", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

render_sidebar()

st.title("📦 Inventory Analytics Cockpit")
st.markdown("Strategic inventory optimization using **ABC** (Value), **XYZ** (Volatility), and **FSN** (Movement) analysis.")

# --- DATA LOADING ---
if 'processed_data' in st.session_state:
    df = st.session_state.processed_data
elif 'uploaded_data' in st.session_state:
    df = st.session_state.uploaded_data
else:
    st.warning("⚠️ No data found. Please upload data first!")
    st.markdown("[👆 Go to Upload Data page](1_📊_Upload_Data)")
    st.stop()

# --- CONFIGURATION ---
with st.expander("⚙️ Analysis Configuration", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    # Identify potential columns
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'Date' in col]
    
    with col1:
        item_col = st.selectbox(
            "Item/Category Column",
            cat_cols,
            index=0 if cat_cols else 0
        )
    
    with col2:
        # Allow user to choose metric basis (e.g. Quantity vs Sales)
        metric_basis = st.selectbox(
            "Metric Basis",
            ["Revenue / Sales", "Quantity", "Margin/Profit"],
            index=0
        )
        
    with col3:
        # Map metric basis to actual column
        # Try to auto-select based on name
        default_idx = 0
        if metric_basis == "Revenue / Sales" and 'Sales' in df.columns:
            default_idx = list(df.columns).index('Sales')
        elif metric_basis == "Quantity" and 'Units_Sold' in df.columns:
            default_idx = list(df.columns).index('Units_Sold')
            
        value_col = st.selectbox(
            f"Select Column for {metric_basis}",
            num_cols,
            index=default_idx if num_cols else 0
        )
    
    with col4:
        date_col = st.selectbox(
            "Date Column (Required for XYZ/FSN)",
            date_cols + ['None'],
            index=0 if date_cols else -1
        )

# --- COMPUTATIONS ---

# 1. ABC Analysis
abc_df = compute_abc(df, item_col, value_col)

# 2. XYZ Analysis
xyz_df = compute_xyz(df, item_col, value_col, date_col)

# 3. FSN Analysis
fsn_df = compute_fsn(df, item_col, date_col)

# Merge Results
# Start with ABC as base (contains all items)
full_analysis = abc_df.copy()

if xyz_df is not None:
    full_analysis = pd.merge(full_analysis, xyz_df, on=item_col, how='left')
    full_analysis['XYZ_Class'] = full_analysis['XYZ_Class'].fillna('Unknown')
    full_analysis['CV'] = full_analysis['CV'].fillna(0)

if fsn_df is not None:
    full_analysis = pd.merge(full_analysis, fsn_df, on=item_col, how='left')
    full_analysis['FSN_Class'] = full_analysis['FSN_Class'].fillna('Unknown')
    full_analysis['Days_Since_Last_Sale'] = full_analysis['Days_Since_Last_Sale'].fillna(-1)

# --- TABS LAYOUT ---
tab_overview, tab_abc, tab_xyz, tab_fsn, tab_matrix = st.tabs([
    "📊 Overview",
    "🅰️🅱️©️ ABC Analysis",
    "📉 XYZ Volatility",
    "🚚 FSN Movement",
    "🧩 Strategy Matrix"
])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    st.markdown("### 🏢 Inventory Health Overview")
    
    # KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    total_skus = len(full_analysis)
    total_value = full_analysis[value_col].sum()
    
    # Class A Value Share
    class_a_val = full_analysis[full_analysis['Class'] == 'A'][value_col].sum()
    class_a_pct = (class_a_val / total_value) * 100 if total_value > 0 else 0
    
    # Fast Moving Count (if FSN exists)
    fast_count = len(full_analysis[full_analysis['FSN_Class'] == 'F']) if 'FSN_Class' in full_analysis.columns else 0
    fast_pct = (fast_count / total_skus) * 100 if total_skus > 0 else 0
    
    with kpi1:
        st.metric("Total SKUs", f"{total_skus:,}")
    with kpi2:
        st.metric(f"Total {metric_basis}", f"₹{total_value:,.0f}")
    with kpi3:
        st.metric("Value in Class A", f"{class_a_pct:.1f}%", "Top 80% Value")
    with kpi4:
        if 'FSN_Class' in full_analysis.columns:
            st.metric("Fast Moving Items", f"{fast_count} ({fast_pct:.1f}%)", "Active Stock")
        else:
            st.metric("Fast Moving Items", "N/A", "Select Date Col")

    st.markdown("---")
    
    # High-level Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Value Distribution by ABC Class")
        fig_abc_bar = px.bar(
            full_analysis.groupby('Class')[value_col].sum().reset_index(),
            x='Class', y=value_col,
            color='Class',
            color_discrete_map={'A': '#2ecc71', 'B': '#f1c40f', 'C': '#e74c3c'},
            text_auto='.2s',
            title=f"Total {metric_basis} by Class"
        )
        st.plotly_chart(fig_abc_bar, use_container_width=True)
        
    with col2:
        if 'XYZ_Class' in full_analysis.columns:
            st.subheader("Volatility Distribution (XYZ)")
            fig_xyz_pie = px.pie(
                full_analysis, names='XYZ_Class', values=value_col,
                title=f"{metric_basis} Share by Volatility",
                color='XYZ_Class',
                color_discrete_map={'X': '#3498db', 'Y': '#9b59b6', 'Z': '#e67e22', 'Unknown': 'gray'}
            )
            st.plotly_chart(fig_xyz_pie, use_container_width=True)
        else:
            st.info("Select a Date column to see Volatility insights.")

# --- TAB 2: ABC ANALYSIS ---
with tab_abc:
    st.markdown("### 🅰️🅱️©️ Pareto Analysis")
    st.markdown("""
    **The Pareto Principle (80/20 Rule):** 
    - **Class A:** Top 80% of value (Vital few)
    - **Class B:** Next 15% of value
    - **Class C:** Bottom 5% of value (Trivial many)
    """)
    
    # Detailed Metrics
    col1, col2, col3 = st.columns(3)
    
    a_items = full_analysis[full_analysis['Class'] == 'A']
    b_items = full_analysis[full_analysis['Class'] == 'B']
    c_items = full_analysis[full_analysis['Class'] == 'C']
    
    with col1:
        st.metric("Class A (High Value)", f"{len(a_items)} Items", f"{a_items[value_col].sum()/total_value:.1%} of Value")
    with col2:
        st.metric("Class B (Moderate)", f"{len(b_items)} Items", f"{b_items[value_col].sum()/total_value:.1%} of Value")
    with col3:
        st.metric("Class C (Low Value)", f"{len(c_items)} Items", f"{c_items[value_col].sum()/total_value:.1%} of Value")
        
    # Pareto Chart
    fig_pareto = go.Figure()

    fig_pareto.add_trace(go.Bar(
        x=full_analysis[item_col],
        y=full_analysis[value_col],
        name=metric_basis,
        marker_color=['#2ecc71' if c == 'A' else '#f1c40f' if c == 'B' else '#e74c3c' for c in full_analysis['Class']]
    ))

    fig_pareto.add_trace(go.Scatter(
        x=full_analysis[item_col],
        y=full_analysis['Cumulative Percentage'],
        name='Cumulative %',
        yaxis='y2',
        mode='lines',
        line=dict(color='black', width=2)
    ))

    fig_pareto.update_layout(
        title='Pareto Chart: Value Contribution by SKU',
        yaxis=dict(title=metric_basis),
        yaxis2=dict(title='Cumulative Percentage', overlaying='y', side='right', range=[0, 105]),
        showlegend=False,
        height=500,
        hovermode="x unified"
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
    
    # Top Items Table
    st.subheader("🏆 Top Performing Items (Class A)")
    st.dataframe(
        a_items[[item_col, value_col, 'Cumulative Percentage']].head(50).style.format({
            value_col: "{:,.2f}",
            'Cumulative Percentage': "{:.2f}%"
        }),
        use_container_width=True
    )

# --- TAB 3: XYZ ANALYSIS ---
with tab_xyz:
    if xyz_df is not None:
        st.markdown("### 📉 Demand Volatility Analysis (XYZ)")
        st.markdown("""
        Classifies items based on the **Coefficient of Variation (CV)** of their demand.
        - **X (Stable):** CV ≤ 0.5 (Easy to forecast, low safety stock)
        - **Y (Variable):** 0.5 < CV ≤ 1.0 (Needs attention)
        - **Z (Volatile):** CV > 1.0 (Hard to forecast, high safety stock)
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        x_cnt = len(full_analysis[full_analysis['XYZ_Class'] == 'X'])
        y_cnt = len(full_analysis[full_analysis['XYZ_Class'] == 'Y'])
        z_cnt = len(full_analysis[full_analysis['XYZ_Class'] == 'Z'])
        
        with col1: st.metric("Class X (Stable)", f"{x_cnt} Items")
        with col2: st.metric("Class Y (Variable)", f"{y_cnt} Items")
        with col3: st.metric("Class Z (Volatile)", f"{z_cnt} Items")
        
        # Scatter Plot
        st.subheader("Demand Stability Map")
        fig_scatter = px.scatter(
            full_analysis,
            x='mean', # From XYZ calculation, need to ensure it's in full_analysis if we merged properly
            y='CV',
            color='XYZ_Class',
            size=value_col,
            hover_data=[item_col],
            title="Mean Demand vs. Volatility (CV)",
            color_discrete_map={'X': '#3498db', 'Y': '#9b59b6', 'Z': '#e67e22'},
            labels={'mean': 'Average Daily Demand', 'CV': 'Coefficient of Variation'}
        )
        # Add threshold lines
        fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="X Limit (0.5)")
        fig_scatter.add_hline(y=1.0, line_dash="dash", line_color="orange", annotation_text="Y Limit (1.0)")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    else:
        st.info("💡 Please select a valid **Date Column** in the configuration above to enable XYZ Analysis.")

# --- TAB 4: FSN ANALYSIS ---
with tab_fsn:
    if fsn_df is not None:
        st.markdown("### 🚚 FSN Movement Analysis")
        st.markdown("""
        Classifies items based on **Recency of Consumption** (Days since last sale).
        - **F (Fast):** Sold within last 30 days
        - **S (Slow):** Sold within 31-90 days
        - **N (Non-moving):** No sales in >90 days (Dead stock candidate)
        """)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        f_items = full_analysis[full_analysis['FSN_Class'] == 'F']
        s_items = full_analysis[full_analysis['FSN_Class'] == 'S']
        n_items = full_analysis[full_analysis['FSN_Class'] == 'N']
        
        n_value = n_items[value_col].sum()
        
        with col1: st.metric("Fast Moving (F)", f"{len(f_items)} Items")
        with col2: st.metric("Slow Moving (S)", f"{len(s_items)} Items")
        with col3: st.metric("Non-Moving (N)", f"{len(n_items)} Items", f"₹{n_value:,.0f} Locked Value")
        
        # Visuals
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Item Count by Movement Class")
            fig_fsn_donut = px.pie(
                full_analysis, names='FSN_Class', 
                hole=0.4,
                color='FSN_Class',
                color_discrete_map={'F': '#2ecc71', 'S': '#f1c40f', 'N': '#e74c3c'}
            )
            st.plotly_chart(fig_fsn_donut, use_container_width=True)
            
        with col2:
            st.subheader("Value Locked by Movement Class")
            fig_fsn_bar = px.bar(
                full_analysis.groupby('FSN_Class')[value_col].sum().reset_index(),
                x='FSN_Class', y=value_col,
                color='FSN_Class',
                color_discrete_map={'F': '#2ecc71', 'S': '#f1c40f', 'N': '#e74c3c'}
            )
            st.plotly_chart(fig_fsn_bar, use_container_width=True)
            
        # Action List
        st.subheader("⚠️ Dead Stock Alert (Non-Moving Items)")
        st.markdown("These items haven't sold in over 90 days. Consider clearance or discounting.")
        st.dataframe(
            n_items[[item_col, 'Last_Sale_Date', 'Days_Since_Last_Sale', value_col]].sort_values('Days_Since_Last_Sale', ascending=False),
            use_container_width=True
        )
        
    else:
        st.info("💡 Please select a valid **Date Column** in the configuration above to enable FSN Analysis.")

# --- TAB 5: STRATEGY MATRIX ---
with tab_matrix:
    if 'XYZ_Class' in full_analysis.columns and 'Unknown' not in full_analysis['XYZ_Class'].values:
        st.markdown("### 🧩 ABC-XYZ Strategic Matrix")
        st.markdown("Combine Value (ABC) and Volatility (XYZ) to determine inventory policies.")
        
        # Create Matrix Data
        matrix_counts = full_analysis.groupby(['Class', 'XYZ_Class']).size().unstack(fill_value=0)
        
        # Ensure all columns/rows exist
        for c in ['X', 'Y', 'Z']:
            if c not in matrix_counts.columns: matrix_counts[c] = 0
        for i in ['A', 'B', 'C']:
            if i not in matrix_counts.index: matrix_counts.loc[i] = 0
            
        matrix_counts = matrix_counts[['X', 'Y', 'Z']].loc[['A', 'B', 'C']]
        
        # Heatmap
        fig_matrix = px.imshow(
            matrix_counts,
            labels=dict(x="Volatility (XYZ)", y="Value (ABC)", color="Count"),
            x=['X', 'Y', 'Z'],
            y=['A', 'B', 'C'],
            text_auto=True,
            color_continuous_scale='Blues',
            title="Inventory Classification Matrix (Count of SKUs)"
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Strategic Recommendations
        st.subheader("💡 Inventory Management Policies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("### 🟢 Gold Standard (AX, BX)")
            st.markdown("""
            *   **Characteristics:** High/Med Value, Stable Demand.
            *   **Strategy:** Automate replenishment (JIT).
            *   **Safety Stock:** Low.
            *   **Focus:** Efficiency and cost reduction.
            """)
            
            st.warning("### 🟡 Risk Management (AY, AZ, BY, BZ)")
            st.markdown("""
            *   **Characteristics:** High/Med Value, Volatile Demand.
            *   **Strategy:** Tight inventory control, frequent reviews.
            *   **Safety Stock:** High (to prevent stockouts).
            *   **Focus:** Availability and forecasting accuracy.
            """)
            
        with col2:
            st.info("### 🔵 Bulk Management (CX)")
            st.markdown("""
            *   **Characteristics:** Low Value, Stable Demand.
            *   **Strategy:** Bulk purchasing, less frequent orders.
            *   **Safety Stock:** Moderate.
            *   **Focus:** Minimizing administrative costs.
            """)
            
            st.error("### 🔴 Candidates for Elimination (CZ)")
            st.markdown("""
            *   **Characteristics:** Low Value, Volatile Demand.
            *   **Strategy:** Make-to-order or drop-shipping.
            *   **Action:** Review for delisting or clearance.
            *   **Risk:** High risk of obsolescence.
            """)
            
        # Clearance Candidates
        if 'FSN_Class' in full_analysis.columns:
            clearance = full_analysis[
                (full_analysis['Class'] == 'C') & 
                (full_analysis['XYZ_Class'] == 'Z') & 
                (full_analysis['FSN_Class'] == 'N')
            ]
            if not clearance.empty:
                st.markdown("---")
                st.markdown(f"### 🗑️ Clearance Candidates (CZ + Non-Moving) - {len(clearance)} Items")
                st.dataframe(clearance[[item_col, value_col, 'Days_Since_Last_Sale']], use_container_width=True)
                
    else:
        st.info("💡 Enable XYZ Analysis (select Date column) to view the Strategy Matrix.")
