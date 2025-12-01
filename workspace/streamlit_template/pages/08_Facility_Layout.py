import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from utils.ui_components import render_sidebar

def load_css():
    with open("assets/custom.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()


st.set_page_config(page_title="Facility Layout", page_icon="🏭", layout="wide")

# Check authentication
if not st.session_state.get('authenticated', False):
    st.error("Please login first!")
    st.stop()

# Sidebar
render_sidebar()

st.title("🏭 Facility Layout & Activity Relationship Chart")
st.markdown("Optimize facility layout using Activity Relationship Chart (ARC) methodology")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Department Setup",
    "🔗 Relationship Matrix",
    "📊 Layout Optimization",
    "📈 Flow Analysis"
])

with tab1:
    st.markdown("### 📋 Department/Activity Setup")
    st.markdown("Define departments or activities for your facility")
    
    # Initialize session state for departments
    if 'departments' not in st.session_state:
        st.session_state.departments = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5, 6],
            'Name': ['Receiving', 'Storage', 'Assembly', 'Packaging', 'Shipping', 'Office'],
            'Area': [500, 2000, 1500, 800, 600, 400],
            'Type': ['Operations', 'Operations', 'Operations', 'Operations', 'Operations', 'Support']
        })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Current Departments")
        
        # Display existing departments
        edited_df = st.data_editor(
            st.session_state.departments,
            width='stretch',
            num_rows="dynamic",
            column_config={
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "Name": st.column_config.TextColumn("Department Name", required=True),
                "Area": st.column_config.NumberColumn("Area (sq ft)", min_value=0),
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["Operations", "Support", "Storage", "Production"],
                    required=True
                )
            }
        )
        
        if st.button("💾 Save Department Changes"):
            st.session_state.departments = edited_df
            st.success("Departments updated successfully!")
    
    with col2:
        st.markdown("#### Quick Stats")
        
        total_area = st.session_state.departments['Area'].sum()
        st.metric("Total Area", f"{total_area:,.0f} sq ft")
        
        dept_count = len(st.session_state.departments)
        st.metric("Total Departments", dept_count)
        
        # Department distribution
        dept_types = st.session_state.departments['Type'].value_counts()
        fig = px.pie(
            values=dept_types.values,
            names=dept_types.index,
            title="Department Types"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, width='stretch')

with tab2:
    st.markdown("### 🔗 Activity Relationship Chart (ARC)")
    st.markdown("Define relationships between departments using closeness ratings")
    
    if len(st.session_state.departments) < 2:
        st.warning("Please add at least 2 departments in the Department Setup tab first.")
    else:
        # Relationship importance scale
        st.markdown("""
        **Closeness Rating Scale:**
        - **A (Absolutely Necessary)**: Must be adjacent (e.g., Assembly & Packaging)
        - **E (Especially Important)**: Should be very close (e.g., Receiving & Storage)
        - **I (Important)**: Should be close (e.g., Storage & Assembly)
        - **O (Ordinary)**: Normal proximity (e.g., Office & Operations)
        - **U (Unimportant)**: No specific requirement
        - **X (Undesirable)**: Should be separated (e.g., Office & Noisy Operations)
        """)


        
        # Initialize relationship matrix
        dept_names = st.session_state.departments['Name'].tolist()
        n_depts = len(dept_names)
        
        if 'relationship_matrix' not in st.session_state:
            # Create default relationship matrix
            matrix_data = []
            for i in range(n_depts):
                for j in range(i+1, n_depts):
                    matrix_data.append({
                        'From': dept_names[i],
                        'To': dept_names[j],
                        'Closeness': 'U',
                        'Reason': ''
                    })
            st.session_state.relationship_matrix = pd.DataFrame(matrix_data)
        
        # Filter to only current departments
        current_pairs = []
        for i in range(n_depts):
            for j in range(i+1, n_depts):
                current_pairs.append((dept_names[i], dept_names[j]))
        
        # Update matrix to match current departments
        if len(st.session_state.relationship_matrix) != len(current_pairs):
            matrix_data = []
            for from_dept, to_dept in current_pairs:
                # Check if relationship already exists
                existing = None
                if 'From' in st.session_state.relationship_matrix.columns and 'To' in st.session_state.relationship_matrix.columns:
                    existing = st.session_state.relationship_matrix[
                        (st.session_state.relationship_matrix['From'] == from_dept) &
                        (st.session_state.relationship_matrix['To'] == to_dept)
                    ]
                if existing is not None and len(existing) > 0:
                    matrix_data.append(existing.iloc[0].to_dict())
                else:
                    matrix_data.append({
                        'From': from_dept,
                        'To': to_dept,
                        'Closeness': 'U',
                        'Reason': ''
                    })
            st.session_state.relationship_matrix = pd.DataFrame(matrix_data)
        
        # Edit relationship matrix
        st.markdown("#### Define Relationships")
        
        edited_matrix = st.data_editor(
            st.session_state.relationship_matrix,
            width='stretch',
            column_config={
                "From": st.column_config.TextColumn("From Department", disabled=True),
                "To": st.column_config.TextColumn("To Department", disabled=True),
                "Closeness": st.column_config.SelectboxColumn(
                    "Closeness Rating",
                    options=["A", "E", "I", "O", "U", "X"],
                    required=True
                ),
                "Reason": st.column_config.TextColumn("Reason (Optional)")
            },
            hide_index=True
        )
        
        if st.button("💾 Save Relationships"):
            st.session_state.relationship_matrix = edited_matrix
            st.success("Relationships updated successfully!")
        
        # Visualize relationship network
        st.markdown("#### Relationship Network Diagram")
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for dept in dept_names:
            G.add_node(dept)
        
        # Add edges with weights
        closeness_weights = {'A': 5, 'E': 4, 'I': 3, 'O': 2, 'U': 1, 'X': 0}
        closeness_colors = {
            'A': 'darkgreen',
            'E': 'green',
            'I': 'lightgreen',
            'O': 'gray',
            'U': 'lightgray',
            'X': 'red'
        }
        
        edges_data = []
        for _, row in st.session_state.relationship_matrix.iterrows():
            weight = closeness_weights.get(row['Closeness'], 1)
            if weight > 0:  # Don't show X relationships in network
                G.add_edge(row['From'], row['To'], weight=weight, closeness=row['Closeness'])
                edges_data.append({
                    'from': row['From'],
                    'to': row['To'],
                    'closeness': row['Closeness'],
                    'weight': weight
                })
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create plotly figure
        edge_trace = []
        for _, row in st.session_state.relationship_matrix.iterrows():
            if row['Closeness'] != 'X':
                x0, y0 = pos[row['From']]
                x1, y1 = pos[row['To']]
                
                edge_trace.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(
                        width=closeness_weights[row['Closeness']],
                        color=closeness_colors[row['Closeness']]
                    ),
                    hoverinfo='text',
                    text=f"{row['From']} - {row['To']}: {row['Closeness']}",
                    showlegend=False
                ))
        
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title="Department Relationship Network",
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            critical_count = len(edited_matrix[edited_matrix['Closeness'].isin(['A', 'E'])])
            st.metric("Critical Relationships (A/E)", critical_count)
        
        with col2:
            important_count = len(edited_matrix[edited_matrix['Closeness'] == 'I'])
            st.metric("Important Relationships (I)", important_count)
        
        with col3:
            undesirable_count = len(edited_matrix[edited_matrix['Closeness'] == 'X'])
            st.metric("Undesirable Relationships (X)", undesirable_count)

with tab3:
    st.markdown("### 📊 Layout Optimization")
    st.markdown("Generate optimized facility layouts based on relationship scores")
    
    if 'departments' not in st.session_state or 'relationship_matrix' not in st.session_state:
        st.warning("Please complete Department Setup and Relationship Matrix first.")
    else:
        # Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            facility_width = st.number_input("Facility Width (ft)", 50, 500, 100, 10)
        
        with col2:
            facility_length = st.number_input("Facility Length (ft)", 50, 500, 150, 10)
        
        with col3:
            layout_type = st.selectbox(
                "Layout Type",
                ["Grid", "Flow-based", "Custom"]
            )
        
        if st.button("🎯 Generate Optimal Layout", type="primary"):
            with st.spinner("Optimizing facility layout..."):
                # Simple grid-based layout algorithm
                depts = st.session_state.departments.copy()
                n_depts = len(depts)
                
                # Calculate grid dimensions
                cols = int(np.ceil(np.sqrt(n_depts)))
                rows = int(np.ceil(n_depts / cols))
                
                # Calculate cell dimensions
                cell_width = facility_width / cols
                cell_length = facility_length / rows
                
                # Assign positions
                positions = []
                for idx, row in depts.iterrows():
                    grid_row = idx // cols
                    grid_col = idx % cols
                    
                    x = grid_col * cell_width + cell_width / 2
                    y = grid_row * cell_length + cell_length / 2
                    
                    positions.append({
                        'Department': row['Name'],
                        'X': x,
                        'Y': y,
                        'Width': min(cell_width * 0.9, np.sqrt(row['Area'])),
                        'Height': min(cell_length * 0.9, np.sqrt(row['Area']))
                    })
                
                st.session_state.layout_positions = pd.DataFrame(positions)
                st.success("Layout generated successfully!")
        
        # Display layout
        if 'layout_positions' in st.session_state:
            st.markdown("#### Optimized Layout Visualization")
            
            fig = go.Figure()
            
            # Draw facility boundary
            fig.add_shape(
                type="rect",
                x0=0, y0=0,
                x1=facility_width, y1=facility_length,
                line=dict(color="black", width=3)
            )
            
            # Draw departments
            for _, dept in st.session_state.layout_positions.iterrows():
                # Rectangle for department
                fig.add_shape(
                    type="rect",
                    x0=dept['X'] - dept['Width']/2,
                    y0=dept['Y'] - dept['Height']/2,
                    x1=dept['X'] + dept['Width']/2,
                    y1=dept['Y'] + dept['Height']/2,
                    fillcolor='lightblue',
                    line=dict(color='darkblue', width=2),
                    opacity=0.7
                )
                
                # Label
                fig.add_annotation(
                    x=dept['X'],
                    y=dept['Y'],
                    text=dept['Department'],
                    showarrow=False,
                    font=dict(size=10, color='black')
                )
            
            # Draw relationship connections
            closeness_colors = {
                'A': 'darkgreen',
                'E': 'green',
                'I': 'lightgreen',
                'O': 'gray',
                'U': 'lightgray',
                'X': 'red'
            }
            
            for _, rel in st.session_state.relationship_matrix.iterrows():
                from_pos = st.session_state.layout_positions[
                    st.session_state.layout_positions['Department'] == rel['From']
                ]
                to_pos = st.session_state.layout_positions[
                    st.session_state.layout_positions['Department'] == rel['To']
                ]
                
                if len(from_pos) > 0 and len(to_pos) > 0:
                    fig.add_trace(go.Scatter(
                        x=[from_pos.iloc[0]['X'], to_pos.iloc[0]['X']],
                        y=[from_pos.iloc[0]['Y'], to_pos.iloc[0]['Y']],
                        mode='lines',
                        line=dict(
                            color=closeness_colors[rel['Closeness']],
                            width=1,
                            dash='dot'
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        text=f"{rel['From']} - {rel['To']}: {rel['Closeness']}"
                    ))
            
            fig.update_layout(
                title="Facility Layout with Department Relationships",
                xaxis=dict(title="Width (ft)", range=[0, facility_width]),
                yaxis=dict(title="Length (ft)", range=[0, facility_length]),
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Layout score
            st.markdown("#### Layout Quality Metrics")
            
            # Calculate layout score based on relationships
            closeness_scores = {'A': 10, 'E': 8, 'I': 6, 'O': 4, 'U': 2, 'X': -10}
            total_score = 0
            max_possible_score = 0
            
            for _, rel in st.session_state.relationship_matrix.iterrows():
                from_pos = st.session_state.layout_positions[
                    st.session_state.layout_positions['Department'] == rel['From']
                ]
                to_pos = st.session_state.layout_positions[
                    st.session_state.layout_positions['Department'] == rel['To']
                ]
                
                if len(from_pos) > 0 and len(to_pos) > 0:
                    # Calculate distance
                    distance = np.sqrt(
                        (from_pos.iloc[0]['X'] - to_pos.iloc[0]['X'])**2 +
                        (from_pos.iloc[0]['Y'] - to_pos.iloc[0]['Y'])**2
                    )
                    
                    # Score: higher closeness rating should have lower distance
                    importance = closeness_scores[rel['Closeness']]
                    max_possible_score += abs(importance) * 100
                    
                    if importance > 0:
                        # Closer is better for positive relationships
                        score = importance * (1 / (1 + distance / 10))
                    elif importance < 0:
                        # Further is better for negative relationships
                        score = abs(importance) * (distance / 100)
                    else:
                        score = 0
                    
                    total_score += score
            
            # Display score
            col1, col2, col3 = st.columns(3)
            
            with col1:
                layout_efficiency = (total_score / max(max_possible_score, 1)) * 100
                st.metric("Layout Efficiency", f"{layout_efficiency:.1f}%")
            
            with col2:
                avg_distance = st.session_state.relationship_matrix.shape[0]
                st.metric("Relationships Optimized", avg_distance)
            
            with col3:
                space_utilization = (st.session_state.departments['Area'].sum() / (facility_width * facility_length)) * 100
                st.metric("Space Utilization", f"{space_utilization:.1f}%")

with tab4:
    st.markdown("### 📈 Material Flow Analysis")
    st.markdown("Analyze material flow and identify bottlenecks")
    
    if 'departments' not in st.session_state:
        st.warning("Please complete Department Setup first.")
    else:
        st.markdown("#### Material Flow Matrix")
        st.info("Define the volume of material/product flow between departments")
        
        # Initialize flow matrix
        dept_names = st.session_state.departments['Name'].tolist()
        n_depts = len(dept_names)
        
        if 'flow_matrix' not in st.session_state:
            # Create default flow matrix (from-to chart)
            flow_data = []
            for from_dept in dept_names:
                for to_dept in dept_names:
                    if from_dept != to_dept:
                        flow_data.append({
                            'From': from_dept,
                            'To': to_dept,
                            'Flow Volume': 0,
                            'Unit': 'units/day'
                        })
            st.session_state.flow_matrix = pd.DataFrame(flow_data)
        
        # Edit flow matrix
        edited_flow = st.data_editor(
            st.session_state.flow_matrix,
            width='stretch',
            column_config={
                "From": st.column_config.TextColumn("From Department", disabled=True),
                "To": st.column_config.TextColumn("To Department", disabled=True),
                "Flow Volume": st.column_config.NumberColumn(
                    "Flow Volume",
                    min_value=0,
                    format="%d"
                ),
                "Unit": st.column_config.SelectboxColumn(
                    "Unit",
                    options=["units/day", "kg/day", "pallets/day", "trips/day"]
                )
            },
            hide_index=True
        )
        
        if st.button("💾 Save Flow Data"):
            st.session_state.flow_matrix = edited_flow
            st.success("Flow data updated successfully!")
        
        # Visualize flow
        if st.session_state.flow_matrix['Flow Volume'].sum() > 0:
            st.markdown("#### Material Flow Diagram")
            
            # Create Sankey diagram
            flow_df = st.session_state.flow_matrix[st.session_state.flow_matrix['Flow Volume'] > 0]
            
            # Prepare data for Sankey
            all_nodes = list(set(flow_df['From'].tolist() + flow_df['To'].tolist()))
            node_dict = {node: idx for idx, node in enumerate(all_nodes)}
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_nodes,
                    color="lightblue"
                ),
                link=dict(
                    source=[node_dict[row['From']] for _, row in flow_df.iterrows()],
                    target=[node_dict[row['To']] for _, row in flow_df.iterrows()],
                    value=flow_df['Flow Volume'].tolist(),
                    color="rgba(0,0,255,0.2)"
                )
            )])
            
            fig.update_layout(
                title="Material Flow Between Departments",
                height=500
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Flow statistics
            st.markdown("#### Flow Statistics")
            
            # Incoming and outgoing flow per department
            incoming = flow_df.groupby('To')['Flow Volume'].sum().reset_index()
            incoming.columns = ['Department', 'Incoming Flow']
            
            outgoing = flow_df.groupby('From')['Flow Volume'].sum().reset_index()
            outgoing.columns = ['Department', 'Outgoing Flow']
            
            flow_stats = pd.merge(incoming, outgoing, on='Department', how='outer').fillna(0)
            flow_stats['Net Flow'] = flow_stats['Outgoing Flow'] - flow_stats['Incoming Flow']
            
            st.dataframe(flow_stats, width='stretch')
            
            # Identify bottlenecks
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top Flow Connections")
                top_flows = flow_df.nlargest(5, 'Flow Volume')[['From', 'To', 'Flow Volume']]
                st.dataframe(top_flows, width='stretch')
            
            with col2:
                st.markdown("#### Potential Bottlenecks")
                # Departments with high incoming flow
                bottlenecks = flow_stats.nlargest(3, 'Incoming Flow')
                for _, dept in bottlenecks.iterrows():
                    st.warning(f"**{dept['Department']}**: High incoming flow ({dept['Incoming Flow']:.0f} units/day)")
        else:
            st.info("No flow data entered yet. Add flow volumes between departments to see analysis.")
        
        # Optimization recommendations
        st.markdown("#### 💡 Optimization Recommendations")
        
        if 'layout_positions' in st.session_state and st.session_state.flow_matrix['Flow Volume'].sum() > 0:
            st.markdown("""
            **Based on your layout and flow analysis:**
            
            1. **High-Flow Connections**: Ensure departments with high material flow are positioned close together
            2. **Bottleneck Management**: Consider expanding capacity or adding parallel paths for bottleneck departments
            3. **Flow Efficiency**: Minimize backtracking and cross-traffic in material flow
            4. **Buffer Zones**: Add staging areas between high-volume departments
            5. **Access Points**: Ensure adequate access for material handling equipment
            """)
        else:
            st.info("Complete both layout optimization and flow data entry to get personalized recommendations.")
